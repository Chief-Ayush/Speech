"""Step 2: durations (from the forced-aligner TextGrids), mel, phoneme-level pitch and energy, train/val statistics.
Run from the repository root, which holds raw_data/ and preprocessed_data/English/TextGrid:  python English/preprocess.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.dirname(HERE), HERE]
from fs2 import load_cfg, features
from frontend import SYMBOLS

if __name__ == "__main__":
    cfg, a = load_cfg(HERE)
    features.run(cfg, set(SYMBOLS), workers=a.workers)
