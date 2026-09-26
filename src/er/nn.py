"""Character-level neural pair matcher (decomposable attention), trained from scratch.

For a (record, Source-1) pair the normalised name and address strings of both sides
are read as bytes -> embedding -> two 1-D convolutions.  Each side then attends over
the other (soft alignment, Parikh et al. 2016 "decomposable attention"); the
[x, aligned, x - aligned, x * aligned] comparison is projected and pooled, for name
and address separately, and an MLP produces the match logit.  ~0.3M parameters,
no pretrained weights or external data.  Its probability is stacked into XGBoost as
feature ``nn_p``; it is trained on records reserved for it (``tid % 10 == NN_FOLD``)
so the XGBoost training rows never see in-sample NN scores.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.er.gpu_features import DEV, to_bytes

NN_FOLD = 7
L_NAME, L_ADDR = 48, 96


class Encoder(nn.Module):
    def __init__(self, emb: nn.Embedding, d: int):
        super().__init__()
        self.emb = emb
        self.c1 = nn.Conv1d(emb.embedding_dim, d, 3, padding=1)
        self.c2 = nn.Conv1d(d, d, 3, padding=1)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        h = F.gelu(self.c1(h))
        h = h + F.gelu(self.c2(h))
        return h.transpose(1, 2)  # [B, L, d]


class Matcher(nn.Module):
    def __init__(self, d: int = 96):
        super().__init__()
        self.emb = nn.Embedding(256, 32, padding_idx=0)
        self.enc_n = Encoder(self.emb, d)
        self.enc_a = Encoder(self.emb, d)
        self.cmp_n = nn.Linear(4 * d, d)
        self.cmp_a = nn.Linear(4 * d, d)
        self.head = nn.Sequential(nn.Linear(8 * d + 4, 128), nn.GELU(), nn.Linear(128, 1))
        self.d = d

    def _side(self, A, B, ma, mb, cmp):
        e = torch.bmm(A, B.transpose(1, 2)) / math.sqrt(self.d)
        e = e.masked_fill(~mb[:, None, :], -1e4)
        al = torch.bmm(torch.softmax(e, -1), B)
        v = F.gelu(cmp(torch.cat([A, al, A - al, A * al], -1)))
        v = v.masked_fill(~ma[:, :, None], 0.0)
        n = ma.sum(1, keepdim=True).clamp(min=1)
        mx = v.masked_fill(~ma[:, :, None], -1e4).max(1).values
        mx = torch.where(ma.any(1, keepdim=True), mx, torch.zeros_like(mx))
        return torch.cat([v.sum(1) / n, mx], -1)

    def _pair(self, xa, xb, enc, cmp):
        ma, mb = xa > 0, xb > 0
        A, B = enc(xa), enc(xb)
        return torch.cat([self._side(A, B, ma, mb, cmp), self._side(B, A, mb, ma, cmp)], -1), ma, mb

    def forward(self, nt, ns, at, as_):
        vn, mnt, mns = self._pair(nt, ns, self.enc_n, self.cmp_n)
        va, mat, mas = self._pair(at, as_, self.enc_a, self.cmp_a)
        flags = torch.stack([mat.any(1), mas.any(1), mnt.any(1), mns.any(1)], 1).float()
        return self.head(torch.cat([vn, va, flags], -1)).squeeze(-1)


def encode_pairs(df: pl.DataFrame):
    """Byte tensors (uint8 on GPU) for columns nm, nm_s, ad, ad_s."""
    out = []
    for col, L in (("nm", L_NAME), ("nm_s", L_NAME), ("ad", L_ADDR), ("ad_s", L_ADDR)):
        x, _ = to_bytes(df[col], L)
        out.append(x.to(torch.uint8))
    return out


@torch.no_grad()
def predict(model: Matcher, df: pl.DataFrame, batch: int = 8192) -> np.ndarray:
    model.eval()
    nt, ns, at, as_ = encode_pairs(df)
    out = torch.empty(nt.shape[0], device=DEV)
    for i in range(0, nt.shape[0], batch):
        sl = slice(i, i + batch)
        with torch.autocast("cuda", dtype=torch.float16):
            out[sl] = model(nt[sl].long(), ns[sl].long(), at[sl].long(), as_[sl].long()).float()
    return torch.sigmoid(out).cpu().numpy()


def train_model(df: pl.DataFrame, epochs: int = 2, batch: int = 1024, lr: float = 2e-3,
                valid: pl.DataFrame | None = None, log=print) -> Matcher:
    """``df``: nm, nm_s, ad, ad_s, y.  Encodes on the GPU in chunks, trains with AMP."""
    torch.manual_seed(0)
    model = Matcher().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    y_all = torch.from_numpy(df["y"].to_numpy().astype(np.float32)).to(DEV)
    enc = encode_pairs(df)
    n = y_all.numel()
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0, step = time.time(), 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(*(e[idx].long() for e in enc))
                loss = F.binary_cross_entropy_with_logits(logit.float(), y_all[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * idx.numel()
            step += 1
            if step % 2000 == 0:
                log(f"  nn epoch {ep} step {step}/{steps} loss {tot / min(i + batch, n):.4f} {time.time() - t0:.0f}s")
        msg = f"  nn epoch {ep} train loss {tot / n:.4f}"
        if valid is not None:
            p = predict(model, valid)
            y = valid["y"].to_numpy()
            eps = 1e-6
            ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            msg += f" | valid logloss {ll:.4f}"
        log(msg + f"  {time.time() - t0:.0f}s")
    return model


def save(model: Matcher, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path: str | Path) -> Matcher:
    m = Matcher().to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV))
    m.eval()
    return m
