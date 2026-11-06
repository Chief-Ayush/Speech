"""Feature extraction: forced-alignment durations + mel + phoneme-level pitch (F0) and energy.
Port of ming024/FastSpeech2 preprocessor.py (modernised; TextGrid parsed directly, split taken from our manifest)."""
import csv
import glob
import json
import os
import re
from multiprocessing import Pool

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.interpolate import interp1d

import torch

SIL = {"", "sil", "sp", "spn"}          # MFA 3.x writes silence as an EMPTY label (older MFA wrote sil/sp/spn)


class MelExtractor:
    def __init__(self, sr=22050, n_fft=1024, hop=256, win=1024, n_mels=80, fmin=0, fmax=8000):
        self.sr, self.n_fft, self.hop, self.win = sr, n_fft, hop, win
        self.mel_basis = torch.from_numpy(
            librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)).float()
        self.window = torch.hann_window(win)

    def __call__(self, wav):
        """wav: float32 numpy in [-1, 1]. Returns (mel [n_mels, T], energy [T])."""
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32))[None]
        spec = torch.stft(x, self.n_fft, self.hop, self.win, window=self.window, center=True,
                          pad_mode="reflect", return_complex=True)
        mag = spec.abs()[0]                                   # [n_fft/2+1, T]
        mel = torch.log(torch.clamp(self.mel_basis @ mag, min=1e-5))
        energy = torch.norm(mag, dim=0)                       # frame-level energy = L2 norm of the magnitude spectrum
        return mel.numpy(), energy.numpy()


def read_phone_tier(path):
    s = open(path, encoding="utf-8").read()
    for blk in re.split(r"item \[\d+\]:", s)[1:]:
        if re.search(r'name = "phones"', blk):
            return [(float(a), float(b), t) for a, b, t in
                    re.findall(r'xmin = ([\d.eE+-]+)\s+xmax = ([\d.eE+-]+)\s+text = "([^"]*)"', blk)]
    raise ValueError("no phones tier in " + path)


def get_alignment(tier, sr, hop):
    """Trim leading/trailing silence; inner silences become the 'sp' token. Durations are in mel frames."""
    phones, durs, start, end, end_idx = [], [], 0.0, 0.0, 0
    for s, e, p in tier:
        if not phones:
            if p in SIL:
                continue
            start = s
        if p not in SIL:
            phones.append(p); end = e; end_idx = len(phones)
        else:
            phones.append("sp")
        durs.append(int(np.round(e * sr / hop) - np.round(s * sr / hop)))
    return phones[:end_idx], durs[:end_idx], start, end


def _remove_outlier(v):
    v = np.asarray(v)
    q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
    return v[(v > q1 - 1.5 * (q3 - q1)) & (v < q3 + 1.5 * (q3 - q1))]


_G = {}


def _init(cfg, tg_dir):
    pc = cfg["preprocess"]
    _G.update(cfg=cfg, mel=MelExtractor(pc["sampling_rate"], pc["filter_length"], pc["hop_length"], pc["win_length"],
                                        pc["n_mels"], pc["mel_fmin"], pc["mel_fmax"]),
              tg={os.path.basename(f)[:-9]: f for f in glob.glob(os.path.join(tg_dir, "**", "*.TextGrid"), recursive=True)})


def _process(args):
    spk, uid, wav_path = args
    cfg = _G["cfg"]; pc = cfg["preprocess"]; sr, hop = pc["sampling_rate"], pc["hop_length"]
    tg_path = _G["tg"].get(uid)
    if tg_path is None:
        return ("missing_textgrid", uid)
    phones, dur, start, end = get_alignment(read_phone_tier(tg_path), sr, hop)
    if not phones or start >= end:
        return ("empty_alignment", uid)
    wav, _ = librosa.load(wav_path, sr=sr)
    wav = wav[int(sr * start): int(sr * end)].astype(np.float32)
    f0, t = pw.dio(wav.astype(np.float64), sr, f0_floor=pc["f0_floor"], f0_ceil=pc["f0_ceil"], frame_period=hop / sr * 1000)
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, sr)[: sum(dur)]
    if np.sum(f0 != 0) <= 1:
        return ("unvoiced", uid)
    mel, energy = _G["mel"](wav)
    mel, energy = mel[:, : sum(dur)], energy[: sum(dur)]
    if mel.shape[1] < sum(dur) or len(f0) < sum(dur):                       # alignment longer than audio: skip
        return ("short_audio", uid)
    nz = np.where(f0 != 0)[0]
    f0 = interp1d(nz, f0[nz], fill_value=(f0[nz[0]], f0[nz[-1]]), bounds_error=False)(np.arange(len(f0)))
    p_ph, e_ph, pos = np.zeros(len(dur)), np.zeros(len(dur)), 0                # phoneme-level averages
    for i, d in enumerate(dur):
        if d > 0:
            p_ph[i], e_ph[i] = f0[pos: pos + d].mean(), energy[pos: pos + d].mean()
        pos += d
    return ("ok", uid, spk, phones, np.array(dur), p_ph, e_ph, mel.T.astype(np.float32))


def run(cfg, phone_ok, workers=4):
    """phone_ok: set of legal phone symbols (anything else -> utterance dropped and reported)."""
    P = cfg["path"]; out = P["preprocessed_path"]; raw = P["raw_path"]
    for d in ("mel", "pitch", "energy", "duration"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    man = list(csv.DictReader(open(P["manifest"], encoding="utf-8")))
    jobs = [(r["speaker"], r["id"], os.path.join(raw, r["speaker"], r["id"] + ".wav")) for r in man]
    split = {r["id"]: r["split"] for r in man}
    text = {r["id"]: r["text"] for r in man}
    with Pool(workers, initializer=_init, initargs=(cfg, os.path.join(out, "TextGrid"))) as pool:
        res = pool.map(_process, jobs, chunksize=16)
    dropped, ok = {}, []
    for r in res:
        if r[0] != "ok":
            dropped.setdefault(r[0], []).append(r[1]); continue
        bad = set(r[3]) - phone_ok
        if bad:
            dropped.setdefault("unknown_phone:" + ",".join(sorted(bad)), []).append(r[1]); continue
        ok.append(r)
    # pitch / energy statistics from TRAIN utterances only (outliers removed), then normalise everything with them
    tr = [r for r in ok if split[r[1]] == "train"]
    stat = {}
    for name, ix in (("pitch", 5), ("energy", 6)):
        allv = _remove_outlier(np.concatenate([r[ix] for r in tr]))
        stat[name] = [float(allv.mean()), float(allv.std())]
    mins = {"pitch": 1e9, "energy": 1e9}; maxs = {"pitch": -1e9, "energy": -1e9}
    spks = sorted({r[2] for r in ok})
    lines = {"train": [], "val": []}
    n_frames = {"train": 0, "val": 0}
    for uid, spk, phones, dur, p, e, mel in [(r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in ok]:
        p = (p - stat["pitch"][0]) / stat["pitch"][1]; e = (e - stat["energy"][0]) / stat["energy"][1]
        if split[uid] == "train":
            mins["pitch"] = min(mins["pitch"], p.min()); maxs["pitch"] = max(maxs["pitch"], p.max())
            mins["energy"] = min(mins["energy"], e.min()); maxs["energy"] = max(maxs["energy"], e.max())
        for d, arr in (("duration", dur), ("pitch", p), ("energy", e), ("mel", mel)):
            np.save(os.path.join(out, d, f"{spk}-{d}-{uid}.npy"), arr)
        lines[split[uid]].append("|".join([uid, spk, "{" + " ".join(phones) + "}", text[uid]]))
        n_frames[split[uid]] += mel.shape[0]
    for k in lines:
        open(os.path.join(out, k + ".txt"), "w", encoding="utf-8").write("\n".join(sorted(lines[k])) + "\n")
    json.dump({s: i for i, s in enumerate(spks)}, open(os.path.join(out, "speakers.json"), "w"))
    json.dump({"pitch": [float(mins["pitch"]), float(maxs["pitch"]), *stat["pitch"]],
               "energy": [float(mins["energy"]), float(maxs["energy"]), *stat["energy"]]},
              open(os.path.join(out, "stats.json"), "w"))
    hop = cfg["preprocess"]["hop_length"]; sr = cfg["preprocess"]["sampling_rate"]
    rep = {"utterances_ok": len(ok), "dropped": {k: len(v) for k, v in dropped.items()},
           "train_clips": len(lines["train"]), "val_clips": len(lines["val"]),
           "train_hours_after_trim": round(n_frames["train"] * hop / sr / 3600, 3),
           "val_hours_after_trim": round(n_frames["val"] * hop / sr / 3600, 3), "speakers": len(spks)}
    json.dump({"report": rep, "dropped_ids": dropped}, open(os.path.join(out, "preprocess_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return rep
