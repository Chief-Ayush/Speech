"""Step 3: train FastSpeech 2 (English).  python English/train.py  (resumes from last.pt automatically)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.dirname(HERE), HERE]
from fs2 import load_cfg, train
from frontend import SYMBOLS

if __name__ == "__main__":
    cfg, a = load_cfg(HERE)
    train.run(cfg, SYMBOLS, max_steps=a.max_steps)
