"""Inference: text symbols -> FastSpeech 2 -> mel -> HiFi-GAN waveform, plus variance-adaptor plots."""
import json
import os
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from .data import phones_to_ids
from .model import FastSpeech2
from .vocoder import AttrDict, Generator

VOCODER_URL = "https://github.com/ming024/FastSpeech2/raw/master/hifigan/generator_universal.pth.tar.zip"
VOC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocoder")


def load_model(ckpt_path, dev="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = FastSpeech2(ck["cfg"], len(ck["symbols"]) + 1, ck["stats"], ck["n_speakers"])
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return m.to(dev).eval(), ck


def load_vocoder(dev="cpu"):
    wdir = VOC_DIR
    pt = os.path.join(wdir, "generator_universal.pth.tar")
    if not os.path.exists(pt):
        z = pt + ".zip"
        print("downloading HiFi-GAN universal vocoder (from the reference repository) ...")
        req = urllib.request.Request(VOCODER_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(z, "wb") as f:
            f.write(r.read())
        zipfile.ZipFile(z).extractall(wdir)
    cfg = AttrDict(json.load(open(os.path.join(wdir, "config.json"))))
    g = Generator(cfg)
    g.load_state_dict(torch.load(pt, map_location="cpu")["generator"])
    g.eval(); g.remove_weight_norm()
    return g.to(dev)


@torch.no_grad()
def synthesize(model, ck, vocoder, phones, speaker=0, ctrl=(1.0, 1.0, 1.0), dev="cpu"):
    ids = torch.tensor([phones_to_ids(phones, ck["symbols"])], device=dev)
    out = model(ids, torch.tensor([ids.shape[1]], device=dev), torch.tensor([speaker], device=dev), ctrl=ctrl)
    mel = out["post"]                                                       # [1, T, 80]
    wav = vocoder(mel.transpose(1, 2)).squeeze().cpu().numpy()
    wav = np.clip(wav, -1, 1)
    st = ck["stats"]
    return dict(wav=wav, mel=mel[0].cpu().numpy(), dur=out["dur"][0].cpu().numpy(),
                pitch=out["pitch"][0].cpu().numpy() * st["pitch"][3] + st["pitch"][2],      # back to Hz
                energy=out["energy"][0].cpu().numpy() * st["energy"][3] + st["energy"][2])


def save_wav(path, wav, sr=22050):
    sf.write(path, (wav * 32767).astype(np.int16), sr, subtype="PCM_16")


def variance_plots(res, phones, title, outdir, name, hop=256, sr=22050, font=None):
    """Three figures per sentence: predicted duration (ms per phoneme), pitch (Hz) and energy over time."""
    dur = res["dur"]; ms = dur * hop / sr * 1000
    edges = np.concatenate([[0], np.cumsum(dur)]) * hop / sr
    x = np.arange(len(phones))
    fp = {"fontproperties": font} if font is not None else {}
    colors = ["#bbbbbb" if p == "sp" else "#2a6f97" for p in phones]
    n = len(phones)
    w = min(max(8, 0.28 * n), 24)                 # cap the width so long sentences stay readable
    fs = 8 if n < 60 else (6.5 if n < 100 else 5.5)

    def finish(fig, kind):
        os.makedirs(os.path.join(outdir, kind), exist_ok=True)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, kind, name + ".png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(w, 3.6))
    ax.bar(x, ms, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(phones, rotation=90, fontsize=fs, **fp)
    ax.set_ylabel("predicted duration (ms)"); ax.set_title("Duration: " + title, fontsize=9, **fp)
    ax.text(0.99, 0.95, "total %.2f s" % (dur.sum() * hop / sr), transform=ax.transAxes, ha="right", va="top", fontsize=9)
    finish(fig, "duration")

    for kind, vals, unit, c in (("pitch", res["pitch"], "F0 (Hz)", "#c1121f"), ("energy", res["energy"], "energy (STFT L2 norm)", "#2d6a4f")):
        fig, ax = plt.subplots(figsize=(w, 3.6))
        ax.stairs(vals, edges, baseline=None, color=c, linewidth=1.8)
        mid = (edges[:-1] + edges[1:]) / 2
        ax.plot(mid, vals, "o", color=c, markersize=3)
        for e in edges[1:-1]:
            ax.axvline(e, color="#dddddd", linewidth=0.5, zorder=0)
        ax.set_xticks(mid); ax.set_xticklabels(phones, rotation=90, fontsize=fs, **fp)
        ax.set_xlim(0, edges[-1]); ax.set_ylabel("predicted " + unit); ax.set_xlabel("time (phoneme boundaries from predicted durations)")
        ax.set_title(kind.capitalize() + " (phoneme level): " + title, fontsize=9, **fp)
        finish(fig, kind)
