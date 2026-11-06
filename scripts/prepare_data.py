"""Data selection, utterance-level splitting and export for the English (Svarah) and Telugu (IndicTTS) models.

Outputs (per language) under <root>/raw_data/<Lang>/<speaker>/<id>.wav (+ .lab) and <root>/prep/<lang>_manifest.csv
All audio is converted to mono 22,050 Hz 16-bit WAV (matches the HiFi-GAN vocoder used by the reference code).
"""
import collections
import csv
import io
import json
import math
import os
import random
import re
import unicodedata

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 22050
SEED = 42
EN_META = ["gender", "age-group", "primary_language", "native_place_state", "native_place_district",
           "highest_qualification", "job_category", "occupation_domain"]


# ----------------------------------------------------------------------------- audio helpers
def read_audio(a):
    """a: dict with 'bytes' or 'path' (datasets Audio(decode=False)). Returns mono float32, sample rate."""
    src = io.BytesIO(a["bytes"]) if a.get("bytes") is not None else a["path"]
    x, sr = sf.read(src, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return x, sr


def quality(x, sr):
    pk = float(np.abs(x).max())
    clip = float((np.abs(x) >= 0.999).mean())
    n = max(1, len(x) // max(1, sr // 100))
    fr = np.array([np.sqrt(np.mean(f ** 2)) + 1e-9 for f in np.array_split(x, n)])
    snr = 20 * np.log10(np.percentile(fr, 90)) - 20 * np.log10(np.percentile(fr, 10))
    return pk, clip, float(snr)


def resample(x, sr, target=TARGET_SR):
    if sr == target:
        return x
    g = math.gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


# ----------------------------------------------------------------------------- English cleaning
_EN_OK = re.compile(r"^[A-Za-z' ,.?!\-]+$")


def clean_en(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = t.replace("\r", " ").replace("\n", " ")
    for a, b in (("’", "'"), ("‘", "'"), ("“", ""), ("”", ""), ('"', "")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def en_lab(t):
    """Transcript for the aligner: lower-case words, apostrophes kept, no punctuation."""
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------- splitting
def split_by_text_group(items, val_hours, seed=SEED, max_val_share_per_speaker=0.4):
    """Utterance-level split, speaker-stratified, with ALL clips sharing the same text on the same side.
    items: dicts with keys spk, dur, key (normalised text). Sets item['split'] to 'train' or 'val'."""
    rng = random.Random(seed)
    total = sum(i["dur"] for i in items)
    frac = val_hours * 3600.0 / total
    by_spk = collections.defaultdict(list)
    for it in items:
        by_spk[it["spk"]].append(it)
    side = {}
    for spk in sorted(by_spk):
        its = by_spk[spk][:]
        rng.shuffle(its)
        spk_total = sum(i["dur"] for i in its)
        target = frac * spk_total
        group_dur = collections.defaultdict(float)  # this speaker's total duration per text group
        for it in its:
            group_dur[it["key"]] += it["dur"]
        acc = sum(d for k, d in group_dur.items() if side.get(k) == "val")
        for k in list(dict.fromkeys(it["key"] for it in its)):  # unique keys in shuffled order
            if k in side:
                continue
            d = group_dur[k]
            if acc + 0.5 * d <= target and acc + d <= max_val_share_per_speaker * spk_total:
                side[k] = "val"
                acc += d
            else:
                side[k] = "train"
    for it in items:
        it["split"] = side[it["key"]]
    return items


def summarize(items):
    out = {}
    for sp in ("train", "val"):
        s = [i for i in items if i["split"] == sp]
        out[sp] = {"clips": len(s), "hours": round(sum(i["dur"] for i in s) / 3600, 3),
                   "speakers": len({i["spk"] for i in s}), "unique_texts": len({i["key"] for i in s})}
    tr = {i["key"] for i in items if i["split"] == "train"}
    va = {i["key"] for i in items if i["split"] == "val"}
    out["text_overlap_train_val"] = len(tr & va)
    out["val_speakers_not_in_train"] = len({i["spk"] for i in items if i["split"] == "val"} -
                                           {i["spk"] for i in items if i["split"] == "train"})
    return out


# ----------------------------------------------------------------------------- English selection
def select_english(ds, max_hours=6.1, val_hours=1.0):
    from datasets import Audio
    acol = "audio_filepath"
    A = ds.cast_column(acol, Audio(decode=False))
    dur = list(ds["duration"])
    txt = [clean_en(t) for t in ds["text"]]
    meta = {c: list(ds[c]) for c in EN_META}
    combos = [" | ".join(str(meta[c][i]) for c in EN_META) for i in range(len(ds))]
    spk_ids = {c: "spk%03d" % k for k, c in enumerate(sorted(set(combos)))}
    stats = collections.Counter()
    cand = []
    for i in range(len(ds)):
        t, d = txt[i], dur[i]
        if not (1.5 <= d <= 15):
            stats["drop_duration"] += 1; continue
        if re.search(r"\d", t) or not _EN_OK.match(t):
            stats["drop_text_chars_or_digits"] += 1; continue
        if len(t.split()) < 3:
            stats["drop_lt3_words"] += 1; continue
        cps = len(t) / d
        if not (5 <= cps <= 22):
            stats["drop_chars_per_sec"] += 1; continue
        cand.append(i)
    # audio quality on candidates
    rows = []
    for i in cand:
        x, sr = read_audio(A[i][acol])
        pk, clip, snr = quality(x, sr)
        if pk < 0.03 or clip > 0.005:
            stats["drop_quiet_or_clipped"] += 1; continue
        rows.append({"src": i, "dur": float(len(x) / sr), "snr": snr})
    # drop the lowest-SNR clips until we are at the target size
    rows.sort(key=lambda r: -r["snr"])
    kept, hrs = [], 0.0
    for r in rows:
        if hrs + r["dur"] / 3600 > max_hours:
            stats["drop_lowest_snr_to_reach_target"] += 1; continue
        kept.append(r); hrs += r["dur"] / 3600
    items = []
    for r in kept:
        i = r["src"]
        items.append({"src": i, "spk": spk_ids[combos[i]], "dur": r["dur"], "text": txt[i], "lab": en_lab(txt[i]),
                      "key": en_lab(txt[i]), "audio_col": acol})
    split_by_text_group(items, val_hours)
    return items, {"drops": dict(stats), "pool_hours_after_filters": round(sum(r["dur"] for r in rows) / 3600, 3),
                   "selected_hours": round(hrs, 3), "n_speakers_total": len(spk_ids),
                   "speaker_definition": "pseudo-speaker = unique combination of 8 metadata fields (no speaker id in dataset)",
                   "split": summarize(items)}


# ----------------------------------------------------------------------------- Telugu selection
def select_telugu(ds, hours=6.1, val_hours=1.0, voice="male"):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Telugu'))
    import telugu_text as TT
    from datasets import Audio
    A = ds.cast_column("audio", Audio(decode=False))
    gid = ds.features["gender"].str2int(voice)
    genders = list(ds["gender"])
    txt_raw = list(ds["text"])
    stats = collections.Counter()
    cand = []
    for i in range(len(ds)):
        if genders[i] != gid:
            continue
        b = A[i]["audio"]
        info = sf.info(io.BytesIO(b["bytes"])) if b.get("bytes") is not None else sf.info(b["path"])
        d = info.duration
        if not (1.5 <= d <= 15):
            stats["drop_duration"] += 1; continue
        t = TT.normalize_script(txt_raw[i])
        if re.search(r"\d", t) or "ఙ" in t:
            stats["drop_digits_or_rare_letter"] += 1; continue
        if any(not (("ఀ" <= c <= "౿") or c in " ,.?!") for c in t):
            stats["drop_non_telugu_chars"] += 1; continue
        cps = len(t) / d
        if not (8 <= cps <= 16.5):
            stats["drop_chars_per_sec"] += 1; continue
        try:
            TT.text_to_symbols(t)
        except Exception:
            stats["drop_unmapped_symbol"] += 1; continue
        cand.append((i, d, t))
    rng = random.Random(SEED)
    rng.shuffle(cand)
    items, hrs = [], 0.0
    for i, d, t in cand:
        if hrs >= hours:
            break
        x, sr = read_audio(A[i]["audio"])
        pk, clip, snr = quality(x, sr)
        if pk < 0.03 or clip > 0.005:
            stats["drop_quiet_or_clipped"] += 1; continue
        dd = len(x) / sr
        items.append({"src": i, "spk": "te_" + voice, "dur": float(dd), "text": t, "lab": TT.text_to_lab(t),
                      "key": t, "audio_col": "audio"})
        hrs += dd / 3600
    split_by_text_group(items, val_hours)
    return items, {"drops": dict(stats), "pool_clips_after_filters": len(cand), "selected_hours": round(hrs, 3),
                   "voice": voice, "split": summarize(items)}


# ----------------------------------------------------------------------------- export
def export(ds, items, root, lang):
    """Write wav (22.05 kHz mono) + .lab per clip under <root>/raw_data/<lang>/<spk>/ and a manifest CSV."""
    from datasets import Audio
    acol = items[0]["audio_col"]
    A = ds.cast_column(acol, Audio(decode=False))
    base = os.path.join(root, "raw_data", lang)
    os.makedirs(os.path.join(root, "prep"), exist_ok=True)
    rows = []
    for n, it in enumerate(items):
        x, sr = read_audio(A[it["src"]][acol])
        y = resample(x, sr)
        uid = "%s_%05d" % (it["spk"], n)
        d = os.path.join(base, it["spk"])
        os.makedirs(d, exist_ok=True)
        sf.write(os.path.join(d, uid + ".wav"), y, TARGET_SR, subtype="PCM_16")
        with open(os.path.join(d, uid + ".lab"), "w", encoding="utf-8") as f:
            f.write(it["lab"] + "\n")
        rows.append({"id": uid, "speaker": it["spk"], "split": it["split"], "duration_s": round(len(y) / TARGET_SR, 3),
                     "text": it["text"], "lab": it["lab"], "source_row": it["src"]})
    path = os.path.join(root, "prep", "%s_manifest.csv" % lang.lower())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return path
