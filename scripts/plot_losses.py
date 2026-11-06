"""Training / validation loss figures from output/log/<Lang>/{train,val}_log.csv.
python scripts/plot_losses.py --log output/log/English --out Results/English --title English"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLS = [("total", "total", "#111111"), ("mel", "mel L1", "#2a6f97"), ("postnet", "post-net mel L1", "#7b2cbf"),
        ("pitch", "pitch MSE", "#c1121f"), ("energy", "energy MSE", "#2d6a4f"), ("duration", "log-duration MSE", "#e09f3e")]


def draw(df, kind, title, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, which in zip(axes, ("total", "components")):
        for key, lab, c in COLS:
            if (which == "total") != (key == "total"):
                continue
            ax.plot(df["step"], df[key], color=c, label=lab, linewidth=1.5, marker="o" if kind == "validation" else None, markersize=3)
        ax.set_xlabel("training step"); ax.set_ylabel(f"{kind} loss"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        ax.set_title(f"{title}: {kind} loss ({which})", fontsize=10)
        if which == "components":
            ax.set_yscale("log")
    if kind == "validation":
        b = df.loc[df["total"].idxmin()]
        axes[0].axvline(b["step"], color="#888", linestyle="--", linewidth=1)
        axes[0].annotate(f"best step {int(b['step'])}\n{b['total']:.3f}", (b["step"], b["total"]), textcoords="offset points", xytext=(8, 12), fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{kind}_loss.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--log"); ap.add_argument("--out"); ap.add_argument("--title", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tr = pd.read_csv(os.path.join(a.log, "train_log.csv")); va = pd.read_csv(os.path.join(a.log, "val_log.csv"))
    draw(tr, "training", a.title, a.out); draw(va, "validation", a.title, a.out)
    both = pd.merge(tr[["step", "total"]], va[["step", "total"]], on="step", suffixes=("_train", "_val"))
    both.to_csv(os.path.join(a.out, "train_vs_val_total.csv"), index=False)
    print(both.iloc[::max(1, len(both) // 12)].to_string(index=False))
