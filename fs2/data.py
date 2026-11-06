import json
import os

import numpy as np
import torch


def phones_to_ids(phones, symbols):
    table = {s: i + 1 for i, s in enumerate(symbols)}   # 0 = padding
    return [table[p] for p in phones]


class TTSData(torch.utils.data.Dataset):
    """Whole split is cached in RAM (about 0.5 GB for 5 h of 80-bin mels)."""
    def __init__(self, pre_dir, split, symbols):
        self.spk_map = json.load(open(os.path.join(pre_dir, "speakers.json")))
        self.items = []
        for line in open(os.path.join(pre_dir, split + ".txt"), encoding="utf-8"):
            uid, spk, ph, raw = line.rstrip("\n").split("|")
            ids = np.array(phones_to_ids(ph.strip("{}").split(), symbols))
            ld = lambda k: np.load(os.path.join(pre_dir, k, f"{spk}-{k}-{uid}.npy"))
            self.items.append(dict(id=uid, spk=self.spk_map[spk], text=ids, mel=ld("mel"), pitch=ld("pitch"),
                                   energy=ld("energy"), dur=ld("duration"), raw=raw))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _pad(arrs, dtype):
    n = max(len(a) for a in arrs)
    out = np.zeros((len(arrs), n) + arrs[0].shape[1:], dtype=dtype)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return torch.from_numpy(out)


def collate(batch):
    return dict(
        ids=[b["id"] for b in batch], spk=torch.tensor([b["spk"] for b in batch]),
        text=_pad([b["text"] for b in batch], np.int64), src_len=torch.tensor([len(b["text"]) for b in batch]),
        mel=_pad([b["mel"] for b in batch], np.float32), mel_len=torch.tensor([len(b["mel"]) for b in batch]),
        pitch=_pad([b["pitch"] for b in batch], np.float32), energy=_pad([b["energy"] for b in batch], np.float32),
        dur=_pad([b["dur"] for b in batch], np.int64))
