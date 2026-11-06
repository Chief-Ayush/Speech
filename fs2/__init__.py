"""Compact, modernised FastSpeech 2 implementation (port of ming024/FastSpeech2, MIT licence).

model.py     network + loss          data.py   dataset / batching        features.py  mel, F0, energy, duration extraction
train.py     training loop           synth.py  inference + plots         vocoder/     HiFi-GAN (pre-trained weights downloaded)
"""
import argparse
import os

import yaml


def load_cfg(here):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(here, "config.yaml"))
    ap.add_argument("--max_steps", type=int, default=None, help="override train.total_step (benchmarks / smoke tests)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--speaker", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return yaml.safe_load(open(a.config)), a
