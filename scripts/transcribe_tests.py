"""Transcribe the generated test sentences with an ASR model (objective per-sentence pronunciation evidence).
python scripts/transcribe_tests.py --asr vasista22/whisper-telugu-large-v2 --asr_lang te --dirs <wav dir> ... --out x.json"""
import argparse, glob, json, os
import numpy as np, soundfile as sf, torch
from transformers import pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--asr", required=True); ap.add_argument("--asr_lang", default="en")
ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

def make(model_id):
    pl = pipeline("automatic-speech-recognition", model=model_id, device=0 if dev == "cuda" else -1, dtype=torch.float16 if dev == "cuda" else None)
    if "whisper" in model_id:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained("openai/whisper-large-v2"); gc.forced_decoder_ids = None
        pl.model.generation_config = gc; pl.model.config.forced_decoder_ids = None
    return pl

used = a.asr; asr = make(used)
tr = lambda w: asr({"raw": w.astype(np.float32), "sampling_rate": 22050}, generate_kwargs={"language": a.asr_lang, "task": "transcribe"})["text"]
try:
    tr(np.zeros(22050, dtype=np.float32))
except Exception as e:
    print("primary ASR failed (%s); falling back to openai/whisper-large-v3" % type(e).__name__, flush=True)
    del asr; torch.cuda.empty_cache(); used = "openai/whisper-large-v3"; asr = make(used)
res = {"asr": used}
for d in a.dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.wav"))):
        w, sr = sf.read(f, dtype="float32")
        res.setdefault(d, {})[os.path.basename(f)] = tr(w)
        print(d, os.path.basename(f), "->", res[d][os.path.basename(f)], flush=True)
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
