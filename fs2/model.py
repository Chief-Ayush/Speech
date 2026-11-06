"""FastSpeech 2 (Ren et al., 2021) -- compact PyTorch port of the ming024/FastSpeech2 architecture.

Components kept from the reference: FFT blocks (multi-head self-attention + 1D-conv feed-forward), variance adaptor
(duration / pitch / energy predictors, phoneme-level pitch and energy, 256-bin quantised embeddings, length regulator),
mel decoder, 5-layer convolutional post-net, optional speaker-embedding table.
Modernised: scaled_dot_product_attention (fp16-safe), no custom STFT, vectorised length regulator.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid_table(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d)[None]
    ang = pos / np.power(10000, 2 * (i // 2) / d)
    t = np.zeros((n_pos, d), dtype=np.float32)
    t[:, 0::2] = np.sin(ang[:, 0::2])
    t[:, 1::2] = np.cos(ang[:, 1::2])
    return torch.from_numpy(t)


def length_mask(lengths, max_len=None):
    """True at PADDING positions."""
    max_len = int(max_len or lengths.max())
    ids = torch.arange(max_len, device=lengths.device)[None]
    return ids >= lengths[:, None]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        super().__init__()
        self.h, self.dk, self.dv = n_head, d_k, d_v
        self.wq, self.wk, self.wv = nn.Linear(d_model, n_head * d_k), nn.Linear(d_model, n_head * d_k), nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, pad):                                  # pad: [B, L] True = padding
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.wk(x).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.wv(x).view(B, L, self.h, self.dv).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=(~pad)[:, None, None, :])
        o = self.drop(self.fc(o.transpose(1, 2).reshape(B, L, -1)))
        return self.ln(o + x)


class ConvFFN(nn.Module):
    def __init__(self, d_model, d_inner, kernel, dropout):
        super().__init__()
        self.c1 = nn.Conv1d(d_model, d_inner, kernel[0], padding=(kernel[0] - 1) // 2)
        self.c2 = nn.Conv1d(d_inner, d_model, kernel[1], padding=(kernel[1] - 1) // 2)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.c2(F.relu(self.c1(x.transpose(1, 2)))).transpose(1, 2)
        return self.ln(self.drop(y) + x)


class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_inner, kernel, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, d_model, d_model // n_head, d_model // n_head, dropout)
        self.ffn = ConvFFN(d_model, d_inner, kernel, dropout)

    def forward(self, x, pad):
        x = self.attn(x, pad).masked_fill(pad[..., None], 0)
        return self.ffn(x).masked_fill(pad[..., None], 0)


class Stack(nn.Module):
    """Encoder / decoder: positional encoding + FFT blocks."""
    def __init__(self, d, layers, heads, d_inner, kernel, dropout, max_len):
        super().__init__()
        self.layers = nn.ModuleList([FFTBlock(d, heads, d_inner, kernel, dropout) for _ in range(layers)])
        self.register_buffer("pos", sinusoid_table(max_len, d), persistent=False)
        self.d = d

    def forward(self, x, pad):
        pos = self.pos if x.shape[1] <= self.pos.shape[0] else sinusoid_table(x.shape[1], self.d).to(x.device)
        x = x + pos[: x.shape[1]][None].to(x.dtype)
        for l in self.layers:
            x = l(x, pad)
        return x


class VariancePredictor(nn.Module):
    def __init__(self, d_in, d_f, k, dropout):
        super().__init__()
        self.c1 = nn.Conv1d(d_in, d_f, k, padding=(k - 1) // 2)
        self.l1 = nn.LayerNorm(d_f)
        self.c2 = nn.Conv1d(d_f, d_f, k, padding=1)
        self.l2 = nn.LayerNorm(d_f)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_f, 1)

    def forward(self, x, pad):
        h = F.relu(self.c1(x.transpose(1, 2))).transpose(1, 2)
        h = self.drop(self.l1(h))
        h = F.relu(self.c2(h.transpose(1, 2))).transpose(1, 2)
        h = self.drop(self.l2(h))
        return self.out(h).squeeze(-1).masked_fill(pad, 0)


def length_regulate(x, dur, max_len=None):
    """x [B,L,D], dur [B,L] (int) -> [B,T,D], mel lengths [B]."""
    lens = dur.sum(1)
    T = int(max_len or lens.max().clamp(min=1))
    cum = dur.cumsum(1)                                          # [B,L]
    t = torch.arange(T, device=x.device)[None, :, None]          # [1,T,1]
    idx = (t >= cum[:, None, :]).sum(-1).clamp(max=x.shape[1] - 1)   # phoneme index per frame
    out = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
    out = out.masked_fill((torch.arange(T, device=x.device)[None] >= lens[:, None])[..., None], 0)
    return out, lens


class VarianceAdaptor(nn.Module):
    def __init__(self, cfg, stats):
        super().__init__()
        m = cfg["model"]
        d, vp = m["encoder_hidden"], m["variance_predictor"]
        self.dur = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])
        self.pitch = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])
        self.energy = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])
        nb = m["n_bins"]
        # linear quantisation over the (normalised) value range seen in the training set
        self.register_buffer("pitch_bins", torch.linspace(stats["pitch"][0], stats["pitch"][1], nb - 1), persistent=True)
        self.register_buffer("energy_bins", torch.linspace(stats["energy"][0], stats["energy"][1], nb - 1), persistent=True)
        self.pitch_emb = nn.Embedding(nb, d)
        self.energy_emb = nn.Embedding(nb, d)

    def forward(self, x, src_pad, mel_pad, max_mel_len, p_t=None, e_t=None, d_t=None, ctrl=(1.0, 1.0, 1.0)):
        log_d = self.dur(x, src_pad)
        p = self.pitch(x, src_pad)
        e = self.energy(x, src_pad)
        if p_t is not None:                                      # training: ground-truth variances condition the decoder
            x = x + self.pitch_emb(torch.bucketize(p_t, self.pitch_bins))
            x = x + self.energy_emb(torch.bucketize(e_t, self.energy_bins))
            dur = d_t
        else:                                                    # inference: use predictions
            x = x + self.pitch_emb(torch.bucketize(p * ctrl[0], self.pitch_bins))
            x = x + self.energy_emb(torch.bucketize(e * ctrl[1], self.energy_bins))
            dur = torch.clamp(torch.round(torch.exp(log_d.float()) - 1) * ctrl[2], min=0).long()
            dur = dur.masked_fill(src_pad, 0)
        x, mel_len = length_regulate(x, dur, max_mel_len)
        return x, p, e, log_d, dur, mel_len


class PostNet(nn.Module):
    def __init__(self, n_mel=80, ch=512, k=5, n=5):
        super().__init__()
        chans = [n_mel] + [ch] * (n - 1) + [n_mel]
        self.convs = nn.ModuleList()
        for i in range(n):
            self.convs.append(nn.Sequential(nn.Conv1d(chans[i], chans[i + 1], k, padding=(k - 1) // 2), nn.BatchNorm1d(chans[i + 1])))

    def forward(self, x):
        x = x.transpose(1, 2)
        for i, c in enumerate(self.convs):
            x = c(x)
            if i < len(self.convs) - 1:
                x = torch.tanh(x)
        return x.transpose(1, 2)


class FastSpeech2(nn.Module):
    def __init__(self, cfg, n_symbols, stats, n_speakers=1):
        super().__init__()
        m = cfg["model"]
        d = m["encoder_hidden"]
        self.emb = nn.Embedding(n_symbols, d, padding_idx=0)
        k = tuple(m["conv_kernel_size"])
        self.encoder = Stack(d, m["encoder_layer"], m["encoder_head"], m["conv_filter_size"], k, m["encoder_dropout"], m["max_seq_len"])
        self.adaptor = VarianceAdaptor(cfg, stats)
        self.decoder = Stack(m["decoder_hidden"], m["decoder_layer"], m["decoder_head"], m["conv_filter_size"], k, m["decoder_dropout"], m["max_seq_len"])
        self.mel_out = nn.Linear(m["decoder_hidden"], cfg["preprocess"]["n_mels"])
        self.postnet = PostNet(cfg["preprocess"]["n_mels"])
        self.spk = nn.Embedding(n_speakers, d) if n_speakers > 1 else None

    def forward(self, ids, src_len, spk=None, mel_len=None, max_mel_len=None, p_t=None, e_t=None, d_t=None, ctrl=(1.0, 1.0, 1.0)):
        src_pad = length_mask(src_len, ids.shape[1])
        x = self.encoder(self.emb(ids), src_pad)
        if self.spk is not None:
            x = x + self.spk(spk)[:, None, :]
        mel_pad = length_mask(mel_len, max_mel_len) if mel_len is not None else None
        x, p, e, log_d, dur, mel_len = self.adaptor(x, src_pad, mel_pad, max_mel_len, p_t, e_t, d_t, ctrl)
        mel_pad = length_mask(mel_len, x.shape[1])
        x = self.decoder(x, mel_pad)
        mel = self.mel_out(x)
        post = (mel + self.postnet(mel)).masked_fill(mel_pad[..., None], 0)
        mel = mel.masked_fill(mel_pad[..., None], 0)
        return dict(mel=mel, post=post, pitch=p, energy=e, log_dur=log_d, dur=dur, src_pad=src_pad, mel_pad=mel_pad, mel_len=mel_len)


def fs2_loss(out, mel_t, p_t, e_t, d_t):
    """Mel L1 (before and after post-net) + MSE on pitch, energy and log(duration+1), masked over padding."""
    sp, mp = ~out["src_pad"], ~out["mel_pad"]
    T = out["mel"].shape[1]
    mel_t = mel_t[:, :T]
    mm = mp[..., None].expand_as(mel_t)
    l_mel = F.l1_loss(out["mel"].float()[mm], mel_t[mm])
    l_post = F.l1_loss(out["post"].float()[mm], mel_t[mm])
    l_p = F.mse_loss(out["pitch"].float()[sp], p_t[sp])
    l_e = F.mse_loss(out["energy"].float()[sp], e_t[sp])
    l_d = F.mse_loss(out["log_dur"].float()[sp], torch.log(d_t.float()[sp] + 1))
    tot = l_mel + l_post + l_p + l_e + l_d
    return tot, dict(total=tot.item(), mel=l_mel.item(), postnet=l_post.item(), pitch=l_p.item(), energy=l_e.item(), duration=l_d.item())
