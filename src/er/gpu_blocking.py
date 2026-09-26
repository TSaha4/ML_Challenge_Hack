"""CUDA candidate generation: same semantics as ``blocking.generate_candidates``.

CPU work is limited to hashing each record's blocking keys (``make_keys`` on
small slices).  The inverted-index lookup, fan-out, per-(target, Source-1) IDF
score aggregation and per-target top-k all run on the GPU with torch:

1. Source-1 keys (df <= cap) are sorted by hash; unique hashes, their start
   offsets and counts live on the GPU.
2. A target slice's key hashes are located with ``searchsorted``; matched index
   ranges are expanded with ``repeat_interleave``.
3. Pairs are grouped with ``unique(return_inverse)`` on ``tid_local * S + s1_row``
   and IDF weights summed with ``index_add_`` (name and address separately).
4. Per-target top-k via a stable two-key sort and rank-within-group.

Slices are split automatically when the fan-out would exceed ``max_pairs``,
so VRAM stays bounded (~2 GB peak at the default).
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import polars as pl
import torch

from src.er.blocking import make_keys
from src.er.resources import wait_for_ram


class GpuIndex:
    def __init__(self, s1: pl.DataFrame, cap: int = 500, device: str = "cuda", slice_rows: int = 400_000):
        self.device = torch.device(device)
        self.s1_ids = s1["id"].to_numpy()
        row_of = pl.DataFrame({"id": s1["id"], "row": np.arange(s1.height, dtype=np.int32)})
        parts = []
        for start in range(0, s1.height, slice_rows):
            wait_for_ram()
            k = make_keys(s1.slice(start, slice_rows))
            parts.append(k.join(row_of, on="id").select("h", "row", "kind"))
        keys = pl.concat(parts)
        del parts
        df_ = keys.group_by("h").agg(pl.len().alias("df"))
        keys = keys.join(df_.filter(pl.col("df") <= cap), on="h").sort("h")
        del df_
        n = float(s1.height)
        w = (math.log(n) - np.log(keys["df"].to_numpy().astype(np.float32))).astype(np.float32)

        # u64 hashes -> signed int64 preserves ordering after the same shift; use a bit-flip
        h = keys["h"].to_numpy().view(np.int64) ^ np.int64(-2**63)
        order_ok = np.all(h[1:] >= h[:-1]) if len(h) > 1 else True
        if not order_ok:  # sort again in signed space
            o = np.argsort(h, kind="stable")
            h, rows, kinds, w = h[o], keys["row"].to_numpy()[o], keys["kind"].to_numpy()[o], w[o]
        else:
            rows, kinds = keys["row"].to_numpy(), keys["kind"].to_numpy()
        del keys
        uh, starts, counts = np.unique(h, return_index=True, return_counts=True)
        dev = self.device
        self.uh = torch.from_numpy(uh).to(dev)
        self.starts = torch.from_numpy(starts.astype(np.int64)).to(dev)
        self.counts = torch.from_numpy(counts.astype(np.int64)).to(dev)
        self.rows = torch.from_numpy(rows.astype(np.int64)).to(dev)
        self.w = torch.from_numpy(w).to(dev)
        self.is_name = torch.from_numpy(kinds == 0).to(dev)
        self.n_rows = len(h)
        self.n_s1 = s1.height

    @staticmethod
    def signed_hash(h: pl.Series) -> np.ndarray:
        return h.to_numpy().view(np.int64) ^ np.int64(-2**63)

    # ------------------------------------------------------------------
    def _topk(self, tl: torch.Tensor, hh: torch.Tensor, n_t: int, k: int):
        dev = self.device
        pos = torch.searchsorted(self.uh, hh).clamp_(max=self.uh.numel() - 1)
        hit = self.uh[pos] == hh
        tl, pos = tl[hit], pos[hit]
        cnt = self.counts[pos]
        total = int(cnt.sum())
        if total == 0:
            return None
        rep = torch.repeat_interleave(torch.arange(cnt.numel(), device=dev), cnt)
        offs = torch.arange(total, device=dev) - (torch.cumsum(cnt, 0) - cnt)[rep]
        idx = self.starts[pos][rep] + offs
        del offs, cnt
        pair = tl[rep] * self.n_s1 + self.rows[idx]
        w, name = self.w[idx], self.is_name[idx]
        del rep, idx
        up, inv = torch.unique(pair, return_inverse=True)
        del pair
        m = up.numel()
        sn = torch.zeros(m, device=dev).index_add_(0, inv, torch.where(name, w, 0.0))
        sa = torch.zeros(m, device=dev).index_add_(0, inv, torch.where(name, 0.0, w))
        nk = torch.zeros(m, device=dev, dtype=torch.int32).index_add_(
            0, inv, torch.ones_like(inv, dtype=torch.int32))
        del inv, w, name
        tloc = up // self.n_s1
        srow = up % self.n_s1
        score = sn + sa
        o = torch.argsort(score, descending=True, stable=True)
        o = o[torch.argsort(tloc[o], stable=True)]
        tloc, srow, sn, sa, nk = tloc[o], srow[o], sn[o], sa[o], nk[o]
        first = torch.ones_like(tloc, dtype=torch.bool)
        first[1:] = tloc[1:] != tloc[:-1]
        grp_start = torch.cummax(torch.where(first, torch.arange(tloc.numel(), device=dev), 0), 0).values
        rk = torch.arange(tloc.numel(), device=dev) - grp_start
        keep = rk < k
        return (tloc[keep].cpu().numpy(), srow[keep].cpu().numpy(), sn[keep].cpu().numpy(),
                sa[keep].cpu().numpy(), nk[keep].cpu().numpy(), rk[keep].cpu().numpy())

    def candidates(self, targets: pl.DataFrame, k: int = 10, slice_targets: int = 50_000,
                   max_pairs: int = 60_000_000) -> pl.DataFrame:
        """Top-``k`` Source-1 candidates per target -> ``tid, s1, sn, sa, nk, rk``."""
        out = []
        stack = [(s, min(slice_targets, targets.height - s)) for s in range(0, targets.height, slice_targets)]
        while stack:
            start, n = stack.pop(0)
            part = targets.slice(start, n)
            keys = make_keys(part).rename({"id": "tid"})
            tid_row = pl.DataFrame({"tid": part["id"], "tl": np.arange(part.height, dtype=np.int64)})
            keys = keys.join(tid_row, on="tid")
            hh = torch.from_numpy(self.signed_hash(keys["h"])).to(self.device)
            tl = torch.from_numpy(keys["tl"].to_numpy()).to(self.device)
            # estimate fan-out before expanding; split the slice if too large
            pos = torch.searchsorted(self.uh, hh).clamp_(max=self.uh.numel() - 1)
            est = int(torch.where(self.uh[pos] == hh, self.counts[pos], 0).sum())
            if est > max_pairs and n > 1000:
                half = n // 2
                stack[:0] = [(start, half), (start + half, n - half)]
                continue
            res = self._topk(tl, hh, part.height, k)
            del hh, tl, pos
            if res is None:
                continue
            tloc, srow, sn, sa, nk, rk = res
            out.append(pl.DataFrame({
                "tid": part["id"].to_numpy()[tloc],
                "s1": self.s1_ids[srow],
                "sn": sn.astype(np.float32),
                "sa": sa.astype(np.float32),
                "nk": nk.astype(np.int16),
                "rk": rk.astype(np.int16),
            }))
        torch.cuda.empty_cache()
        return pl.concat(out) if out else pl.DataFrame(
            schema={"tid": pl.Int64, "s1": pl.Int64, "sn": pl.Float32, "sa": pl.Float32,
                    "nk": pl.Int16, "rk": pl.Int16})
