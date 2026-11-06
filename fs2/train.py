"""Training loop (Adam + Noam warm-up/anneal schedule of the reference code, optional fp16 autocast, resumable)."""
import csv
import json
import os
import time

import torch
import yaml

from .data import TTSData, collate
from .model import FastSpeech2, fs2_loss

class BucketSampler:
    """Length-bucketed batches: shuffle, cut into chunks of 40 batches, sort each chunk by mel length, shuffle the batches.
    Cuts the padding waste of random batches (clips are 1.5-15 s) without making batches deterministic."""
    def __init__(self, lens, bs, seed=0):
        self.lens, self.bs, self.seed, self.ep = lens, bs, seed, 0

    def __len__(self):
        return len(self.lens) // self.bs

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self.ep); self.ep += 1
        idx = list(range(len(self.lens))); rng.shuffle(idx)
        batches, ch = [], self.bs * 40
        for i in range(0, len(idx), ch):
            c = sorted(idx[i:i + ch], key=lambda j: self.lens[j])
            batches += [c[k:k + self.bs] for k in range(0, len(c) - self.bs + 1, self.bs)]
        rng.shuffle(batches)
        return iter(batches)


COMPONENTS = ["total", "mel", "postnet", "pitch", "energy", "duration"]


def lr_at(step, cfg, d_model):
    o = cfg["train"]
    lr = d_model ** -0.5 * min(step ** -0.5, step * o["warm_up_step"] ** -1.5)
    for a in o["anneal_steps"]:
        if step > a:
            lr *= o["anneal_rate"]
    return lr


@torch.no_grad()
def validate(model, loader, dev, amp):
    model.eval()
    sums, n = {k: 0.0 for k in COMPONENTS}, 0
    for b in loader:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(b["text"], b["src_len"], b["spk"], b["mel_len"], int(b["mel_len"].max()), b["pitch"], b["energy"], b["dur"])
        _, parts = fs2_loss(out, b["mel"], b["pitch"], b["energy"], b["dur"])
        w = len(b["ids"])
        for k in COMPONENTS:
            sums[k] += parts[k] * w
        n += w
    model.train()
    return {k: v / n for k, v in sums.items()}


def run(cfg, symbols, max_steps=None):
    torch.manual_seed(cfg["train"].get("seed", 1234))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, P = cfg["train"], cfg["path"]
    pre = P["preprocessed_path"]
    stats = json.load(open(os.path.join(pre, "stats.json")))
    tr, va = TTSData(pre, "train", symbols), TTSData(pre, "val", symbols)
    n_spk = len(tr.spk_map)
    counts = {}
    for it in tr.items:
        counts[it["spk"]] = counts.get(it["spk"], 0) + 1
    tl = torch.utils.data.DataLoader(tr, batch_sampler=BucketSampler([len(it["mel"]) for it in tr.items], T["batch_size"], T.get("seed", 1234)),
                                     collate_fn=collate, num_workers=2, persistent_workers=True)
    vl = torch.utils.data.DataLoader(va, batch_size=T["batch_size"], shuffle=False, collate_fn=collate)
    model = FastSpeech2(cfg, len(symbols) + 1, stats, n_spk).to(dev)
    print("parameters: %.2fM | train clips %d | val clips %d | speakers %d | device %s" % (
        sum(p.numel() for p in model.parameters()) / 1e6, len(tr), len(va), n_spk, dev), flush=True)
    opt = torch.optim.Adam(model.parameters(), betas=tuple(T["betas"]), eps=T["eps"], weight_decay=T["weight_decay"])
    amp = bool(T.get("amp", False)) and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    os.makedirs(P["ckpt_path"], exist_ok=True); os.makedirs(P["log_path"], exist_ok=True)
    step, best_resume = 0, float("inf")
    last = os.path.join(P["ckpt_path"], "last.pt")
    if os.path.exists(last):                                     # resume (Kaggle sessions can die)
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print("resumed from step", step, flush=True)
        bp = os.path.join(P["ckpt_path"], "best.pt")
        if os.path.exists(bp):                                    # keep the best-validation record across restarts
            best_resume = torch.load(bp, map_location="cpu")["val"]["total"]
    new = not os.path.exists(os.path.join(P["log_path"], "train_log.csv"))
    ftr = open(os.path.join(P["log_path"], "train_log.csv"), "a", newline=""); wtr = csv.writer(ftr)
    fva = open(os.path.join(P["log_path"], "val_log.csv"), "a", newline=""); wva = csv.writer(fva)
    if new:
        wtr.writerow(["step"] + COMPONENTS); wva.writerow(["step"] + COMPONENTS)
    total = max_steps or T["total_step"]
    best = best_resume; t0 = time.time(); acc = {k: 0.0 for k in COMPONENTS}; na = 0
    model.train()
    while step < total:
        for b in tl:
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg, cfg["model"]["encoder_hidden"])
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(b["text"], b["src_len"], b["spk"], b["mel_len"], int(b["mel_len"].max()), b["pitch"], b["energy"], b["dur"])
            loss, parts = fs2_loss(out, b["mel"], b["pitch"], b["energy"], b["dur"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip_thresh"])
            scaler.step(opt); scaler.update()
            for k in COMPONENTS:
                acc[k] += parts[k]
            na += 1
            if step % T["log_step"] == 0:
                wtr.writerow([step] + [round(acc[k] / na, 5) for k in COMPONENTS]); ftr.flush()
                print("step %d/%d  loss %.4f  (mel %.3f post %.3f pitch %.3f energy %.3f dur %.3f)  lr %.2e  %.1f it/s" % (
                    step, total, acc["total"] / na, acc["mel"] / na, acc["postnet"] / na, acc["pitch"] / na,
                    acc["energy"] / na, acc["duration"] / na, opt.param_groups[0]["lr"], T["log_step"] / max(1e-9, time.time() - t0)), flush=True)
                acc = {k: 0.0 for k in COMPONENTS}; na = 0; t0 = time.time()
            if step % T["val_step"] == 0 or step == total:
                v = validate(model, vl, dev, amp)
                wva.writerow([step] + [round(v[k], 5) for k in COMPONENTS]); fva.flush()
                print("  [val] step %d  loss %.4f  (mel %.3f post %.3f pitch %.3f energy %.3f dur %.3f)" % (
                    step, v["total"], v["mel"], v["postnet"], v["pitch"], v["energy"], v["duration"]), flush=True)
                if v["total"] < best:
                    best = v["total"]
                    torch.save({"model": model.state_dict(), "step": step, "val": v}, os.path.join(P["ckpt_path"], "best.pt"))
            if step % T["save_step"] == 0 or step == total:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
            if step >= total:
                break
    # inference-only checkpoints (fp16 weights; the 256-bin quantiser buffers stay fp32)
    export_inference(os.path.join(P["ckpt_path"], "best.pt"), os.path.join(P["ckpt_path"], "model_best_fp16.pt"), cfg, symbols, stats, n_spk, tr.spk_map, counts)
    export_inference(last, os.path.join(P["ckpt_path"], "model_final_fp16.pt"), cfg, symbols, stats, n_spk, tr.spk_map, counts)


def export_inference(src, dst, cfg, symbols, stats, n_spk, spk_map, counts=None):
    if not os.path.exists(src):
        return
    ck = torch.load(src, map_location="cpu")
    sd = {k: (v.half() if v.is_floating_point() and "bins" not in k else v) for k, v in ck["model"].items()}
    torch.save({"model": sd, "step": ck["step"], "cfg": cfg, "symbols": symbols, "stats": stats, "n_speakers": n_spk,
                "speakers": spk_map, "speaker_counts": counts}, dst)
    print("saved", dst, "%.1f MB" % (os.path.getsize(dst) / 1e6), "(step %d)" % ck["step"])
