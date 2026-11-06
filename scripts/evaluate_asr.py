"""Objective intelligibility check without listening: synthesise held-out (validation) sentences with FastSpeech 2 + HiFi-GAN,
transcribe them with an off-the-shelf ASR model and report character / word error rate against the input text.
The same ASR on the REAL validation recordings gives the ceiling (ASR errors that are not the TTS model's fault).

python scripts/evaluate_asr.py --lang English --ckpts output/ckpt/English/model_best_fp16.pt output/ckpt/English/model_final_fp16.pt \
       --asr openai/whisper-small --n 80 --out Results/English/asr_eval.json
python scripts/evaluate_asr.py --lang Telugu --asr vasista22/whisper-telugu-large-v2 --asr_lang te ..."""
import argparse
import csv
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm(t, lang):
    t = unicodedata.normalize("NFC", t).lower()
    t = re.sub(r"[​-‍﻿]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def rates(refs, hyps, lang):
    ce = cn = we = wn = 0
    for r, h in zip(refs, hyps):
        r, h = norm(r, lang), norm(h, lang)
        rc, hc = r.replace(" ", ""), h.replace(" ", "")
        ce += lev(rc, hc); cn += len(rc)
        we += lev(r.split(), h.split()); wn += len(r.split())
    return dict(CER=round(100 * ce / cn, 2), WER=round(100 * we / wn, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True); ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--asr", required=True); ap.add_argument("--asr_lang", default="en")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None); ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    import random
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import pipeline
    from fs2 import synth
    sys.path.insert(0, os.path.join(ROOT, a.lang))
    from frontend import text_to_phones
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = a.manifest or f"prep/{a.lang.lower()}_manifest.csv"; raw = a.raw or f"raw_data/{a.lang}"
    rows = [r for r in csv.DictReader(open(man, encoding="utf-8")) if r["split"] == "val"]
    random.Random(0).shuffle(rows); rows = rows[: a.n]
    def make_asr(model_id):
        pl = pipeline("automatic-speech-recognition", model=model_id, device=0 if dev == "cuda" else -1, dtype=torch.float16 if dev == "cuda" else None)
        if "whisper" in model_id:
            # older community fine-tunes ship an outdated generation config that new transformers rejects with `language=`;
            # give them the standard multilingual Whisper-large generation settings (same tokenizer / architecture family)
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained("openai/whisper-large-v2")
            gc.forced_decoder_ids = None
            pl.model.generation_config = gc
            pl.model.config.forced_decoder_ids = None
        return pl
    gk_lang = {"language": a.asr_lang, "task": "transcribe"}
    asr, asr_used = make_asr(a.asr), a.asr
    def transcribe(wav):
        inp = {"raw": wav.astype(np.float32), "sampling_rate": 22050}
        return asr(inp, generate_kwargs=gk_lang)["text"] if "whisper" in asr_used else asr(inp)["text"]
    try:
        transcribe(np.zeros(22050, dtype=np.float32))
    except Exception as e:                                           # fall back to the generic multilingual Whisper
        print("primary ASR failed (%s: %s); falling back to openai/whisper-large-v3" % (type(e).__name__, str(e)[:120]), flush=True)
        del asr; torch.cuda.empty_cache()
        asr_used = "openai/whisper-large-v3"; asr = make_asr(asr_used)
    refs = [r["text"] for r in rows]
    res = {"n": len(rows), "asr": asr_used}
    real = [transcribe(sf.read(os.path.join(raw, r["speaker"], r["id"] + ".wav"), dtype="float32")[0]) for r in rows]
    res["real_recordings"] = rates(refs, real, a.lang)
    print("real recordings:", res["real_recordings"], flush=True)
    voc = synth.load_vocoder(dev)
    for c in a.ckpts:
        model, ck = synth.load_model(c, dev)
        counts = ck.get("speaker_counts") or {}
        hyps, bad = [], 0
        for r in rows:
            spk = ck["speakers"].get(r["speaker"], 0)          # the held-out sentence in its own speaker's voice
            try:
                w = synth.synthesize(model, ck, voc, text_to_phones(r["text"]), speaker=spk, dev=dev)["wav"]
                hyps.append(transcribe(w))
            except Exception as e:                              # unsupported symbol etc.: count as empty hypothesis
                bad += 1; hyps.append("")
        res[os.path.basename(c)] = dict(step=ck["step"], failed=bad, **rates(refs, hyps, a.lang), examples=[(refs[i], hyps[i]) for i in range(12)])
        print(os.path.basename(c), res[os.path.basename(c)], flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
