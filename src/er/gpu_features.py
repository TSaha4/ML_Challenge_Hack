"""CUDA pair features: exact char n-gram and hashed-token overlaps computed in torch.

Strings are moved to the GPU as padded byte matrices built directly from the
Arrow buffers (no per-string Python work).  For each (target, Source-1) pair:

* char 2-gram / 3-gram multiset overlap -> Jaccard and both containments
  (exact: rows of the Source-1 side are sorted and probed with a batched
  ``searchsorted``), for ``nm``, ``core``, ``sq`` and ``ad``;
* tokens hashed on the GPU (polynomial hash per token via segment scatter) ->
  token Jaccard and IDF-weighted overlap for ``core`` and ``ad``; numeric address
  tokens -> number overlap and first-number equality.

A few edit-distance scores that have no cheap GPU analogue come from rapidfuzz on
a small, fixed number of CPU workers (``ER_WORKERS``, default 2).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl
import pyarrow as pa
import torch
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.er.features import add_context
from src.er.siblings import SIB_FEATURES, SiblingLookup

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = int(os.environ.get("ER_WORKERS", "2"))
_MOD = 2_147_483_647
_BASE = 1_000_003

FIELD_LEN = {"nm": 64, "core": 56, "sq": 48, "ad": 112, "nm_pre": 40}
TOKENS = {"core": 12, "ad": 24}

BLOCK_FEATURES = ["sn", "sa", "nk", "rk", "qs", "rq", "score", "score_top", "score_rel", "score_gap",
                  "n_cand", "s1_top_cnt", "s1_cand_cnt"]
GRAM_FEATURES = [f"{f}_{q}_{m}" for f in ("nm", "core", "sq", "ad") for q in ("g2", "g3")
                 for m in ("jac", "cin", "cout")] + ["pre_core_g3_cin"]
TOKEN_FEATURES = ["core_tok_jac", "core_tok_wjac", "core_tok_wcov_t", "core_tok_wcov_s",
                  "ad_tok_jac", "ad_tok_wjac", "ad_tok_wcov_t", "ad_tok_wcov_s",
                  "num_jac", "num_first_eq", "num_missing", "ntok_t", "ntok_s", "ad_ntok_t", "ad_ntok_s"]
CPU_FEATURES = ["rf_nm_ratio", "rf_core_tset", "rf_ad_tset"]
FLAG_FEATURES = ["src", "n_alias", "n_domain", "n_indic", "ad_null", "ad_null_s",
                 "len_nm_t", "len_nm_s", "len_ad_t", "len_ad_s",
                 # name rarity: how many Source-1 records carry exactly this core name
                 "core_freq_s", "core_freq_t", "core_eq", "sq_eq"]
NUM_FEATURES = ["num_absdiff", "num_logratio"]
FEATURES = (BLOCK_FEATURES + GRAM_FEATURES + TOKEN_FEATURES + CPU_FEATURES + FLAG_FEATURES
            + NUM_FEATURES + SIB_FEATURES)


# ---------------------------------------------------------------------------
# strings -> padded byte tensors
# ---------------------------------------------------------------------------
def to_bytes(s: pl.Series, L: int):
    """Return (``[n, L]`` int64 byte codes, 0-padded; ``[n]`` lengths clipped to L)."""
    arr = s.fill_null("").to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    arr = arr.cast(pa.large_string())
    n = len(arr)
    bufs = arr.buffers()
    offsets = np.frombuffer(bufs[1], dtype=np.int64)[arr.offset: arr.offset + n + 1]
    data = np.frombuffer(bufs[2], dtype=np.uint8) if bufs[2] is not None else np.zeros(1, np.uint8)
    off = torch.from_numpy(offsets.copy()).to(DEV)
    dat = torch.from_numpy(data[offsets[0]: offsets[-1]].copy() if offsets[-1] > offsets[0]
                           else np.zeros(1, np.uint8)).to(DEV)
    off = off - off[0]
    lens = (off[1:] - off[:-1]).clamp(max=L)
    ar = torch.arange(L, device=DEV)
    pos = (off[:-1, None] + ar[None, :]).clamp(max=max(dat.numel() - 1, 0))
    mask = ar[None, :] < lens[:, None]
    x = torch.where(mask, dat[pos].long(), torch.zeros((), dtype=torch.long, device=DEV))
    return x, lens


# ---------------------------------------------------------------------------
# char n-gram multiset overlap
# ---------------------------------------------------------------------------
def _grams(x, lens, q: int, sentinel: int):
    L = x.shape[1]
    if q == 2:
        g = x[:, :-1] * 257 + x[:, 1:]
    else:
        g = (x[:, :-2] * 257 + x[:, 1:-1]) * 257 + x[:, 2:]
    valid = torch.arange(L - q + 1, device=DEV)[None, :] < (lens[:, None] - q + 1)
    return torch.where(valid, g, torch.full_like(g, sentinel)), valid.sum(1)


def gram_overlap(xa, la, xb, lb, q: int):
    """Jaccard, |A∩B|/|A|, |A∩B|/|B| over char q-gram multisets (row-wise)."""
    ga, na = _grams(xa, la, q, -1)
    gb, nb = _grams(xb, lb, q, -2)
    ga = ga.sort(dim=1).values.contiguous()
    gb = gb.sort(dim=1).values.contiguous()
    # The nth occurrence in A matches only if B contains at least n copies.
    # A presence-only lookup overcounts repeated grams and is asymmetric.
    first_a = torch.searchsorted(ga, ga, right=False)
    occurrence = torch.arange(ga.shape[1], device=ga.device)[None, :] - first_a
    count_b = torch.searchsorted(gb, ga, right=True) - torch.searchsorted(gb, ga, right=False)
    inter = ((ga >= 0) & (occurrence < count_b)).sum(1).float()
    naf, nbf = na.float(), nb.float()
    jac = inter / (naf + nbf - inter).clamp(min=1)
    return jac, inter / naf.clamp(min=1), inter / nbf.clamp(min=1)


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------
_POW = None


def _powtab(L: int):
    global _POW
    if _POW is None or _POW.numel() < L:
        p = [1]
        for _ in range(L):
            p.append(p[-1] * _BASE % _MOD)
        _POW = torch.tensor(p, dtype=torch.long, device=DEV)
    return _POW[:L]


def token_hashes(x, lens, T: int):
    """Per-row token hashes ``[n, T]`` (-1 = empty) and numeric-token mask."""
    n, L = x.shape
    ar = torch.arange(L, device=DEV)[None, :].expand(n, L)
    inside = ar < lens[:, None]
    ch = inside & (x != 32)
    prev = torch.zeros_like(ch)
    prev[:, 1:] = ch[:, :-1]
    start = ch & ~prev
    tok = torch.cumsum(start.long(), 1) - 1
    sidx = torch.cummax(torch.where(start, ar, torch.zeros_like(ar)), 1).values
    pos = ar - sidx
    contrib = torch.where(ch, (x + 1) * _powtab(L)[pos] % _MOD, torch.zeros_like(x))
    keep = ch & (tok < T)
    rows = torch.arange(n, device=DEV)[:, None].expand(n, L)
    flat = (rows * T + tok.clamp(min=0, max=T - 1))[keep]
    h = torch.zeros(n * T, dtype=torch.long, device=DEV).index_add_(0, flat, contrib[keep])
    ln = torch.zeros(n * T, dtype=torch.long, device=DEV).index_add_(0, flat, torch.ones_like(flat))
    nd = torch.zeros(n * T, dtype=torch.long, device=DEV).index_add_(
        0, flat, (((x >= 48) & (x <= 57)).long())[keep])
    h, ln, nd = h.view(n, T), ln.view(n, T), nd.view(n, T)
    exists = ln > 0
    h = torch.where(exists, (h % _MOD) * 64 + ln.clamp(max=63), torch.full_like(h, -1))
    return h, exists & (nd == ln)


class TokenIdf:
    """IDF of hashed tokens over a reference corpus (Source-1)."""

    def __init__(self, s: pl.Series, L: int, T: int, chunk: int = 400_000):
        hs = []
        for start in range(0, s.len(), chunk):
            x, l = to_bytes(s.slice(start, chunk), L)
            h, _ = token_hashes(x, l, T)
            hs.append(h[h >= 0])  # tokens are already de-duplicated per record -> doc freq
        allh = torch.cat(hs)
        self.keys, cnt = torch.unique(allh, return_counts=True)
        self.idf = torch.log(torch.tensor(float(s.len()), device=DEV) / cnt.float()).clamp(min=0.1)
        self.default = float(np.log(max(s.len(), 1)))

    def __call__(self, h):
        if not self.keys.numel():
            return torch.full(h.shape, self.default, dtype=torch.float32, device=h.device)
        pos = torch.searchsorted(self.keys, h).clamp(max=self.keys.numel() - 1)
        found = self.keys[pos] == h
        return torch.where(found, self.idf[pos], torch.full_like(self.idf[pos], self.default))


def token_overlap(ha, hb, idf: Optional[TokenIdf]):
    va, vb = ha >= 0, hb >= 0
    eq = (ha[:, :, None] == hb[:, None, :]) & va[:, :, None] & vb[:, None, :]
    in_b = eq.any(2)
    na, nb, inter = va.sum(1).float(), vb.sum(1).float(), in_b.sum(1).float()
    jac = inter / (na + nb - inter).clamp(min=1)
    if idf is None:
        return jac, None, None, None
    wa = torch.where(va, idf(ha), 0.0)
    wb = torch.where(vb, idf(hb), 0.0)
    wi = torch.where(in_b, wa, 0.0).sum(1)
    sa, sb = wa.sum(1), wb.sum(1)
    return jac, wi / (sa + sb - wi).clamp(min=1e-6), wi / sa.clamp(min=1e-6), wi / sb.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
class GpuFeaturizer:
    """Holds Source-1 text + corpus IDF tables; featurises candidate chunks."""

    T_COLS = ["id", "cty", "nm", "core", "sq", "nm_pre", "ad", "ad_nums", "ad_num",
              "src", "n_alias", "n_domain", "n_indic", "ad_null"]
    S_COLS = ["id", "nm", "core", "sq", "ad", "ad_num", "ad_null"]

    def __init__(self, s1: pl.DataFrame, stats: pl.DataFrame, sib_dir=None, nn_path=None):
        self.sibs = SiblingLookup(sib_dir) if sib_dir else None
        self.nn = None
        if nn_path:
            if not Path(nn_path).is_file():
                raise FileNotFoundError(nn_path)
            from src.er import nn as NN
            self.nn = NN.load(nn_path)
        self.freq = s1.group_by("cty", "core").agg(pl.len().cast(pl.Int32).alias("core_freq"))
        self.s1 = (s1.select(*self.S_COLS, "cty")
                   .join(self.freq, on=["cty", "core"], how="left")
                   .rename({"core_freq": "core_freq_s"}).drop("cty"))
        self.S_COLS = self.S_COLS + ["core_freq_s"]
        self.stats = stats
        self.idf_core = TokenIdf(s1["core"], FIELD_LEN["core"], TOKENS["core"])
        self.idf_ad = TokenIdf(s1["ad"], FIELD_LEN["ad"], TOKENS["ad"])

    def featurize(self, cands: pl.DataFrame, tg: pl.DataFrame, gpu_chunk: int = 250_000) -> pl.DataFrame:
        """``cands`` must contain all candidates of each of its targets."""
        c = add_context(cands, self.stats)
        tgt = (tg.select(self.T_COLS).join(self.freq, on=["cty", "core"], how="left")
               .with_columns(pl.col("core_freq").fill_null(0).alias("core_freq_t")).drop("core_freq", "cty"))
        ren = {k: f"{k}_s" for k in self.S_COLS if k not in ("id", "core_freq_s")}
        df = (c.join(tgt.rename({"id": "tid"}), on="tid", how="left")
              .join(self.s1.rename(ren).rename({"id": "s1"}), on="s1", how="left"))
        # house-number distance (numeric part of the first number)
        n_t = pl.col("ad_num").str.extract(r"^(\d+)", 1).cast(pl.Int64, strict=False)
        n_s = pl.col("ad_num_s").str.extract(r"^(\d+)", 1).cast(pl.Int64, strict=False)
        df = df.with_columns(
            (n_t - n_s).abs().fill_null(-1).cast(pl.Float32).alias("num_absdiff"),
            ((n_t + 1).cast(pl.Float64).log() - (n_s + 1).cast(pl.Float64).log()).abs()
            .fill_null(-1).cast(pl.Float32).alias("num_logratio"),
        )
        if self.sibs is not None:
            sf = self.sibs.features(df.select(
                "tid", "s1", "rk", "src", "core", "ad", "ad_num", "ad_nums",
                ((pl.col("ad_num") == pl.col("ad_num_s")) & (pl.col("ad_num") != "")).cast(pl.Int32)
                .alias("num_eq_s_self")))
            df = df.join(sf, on=["tid", "s1"], how="left")
        else:
            df = df.with_columns([pl.lit(0, pl.Int32).alias(c) for c in SIB_FEATURES])
        outs = [self._chunk(df.slice(i, gpu_chunk)) for i in range(0, df.height, gpu_chunk)]
        if not outs:
            cols = FEATURES + (["nn_p"] if self.nn is not None else [])
            return pl.DataFrame(schema={"tid": pl.Int64, "s1": pl.Int64, **{c: pl.Float32 for c in cols}})
        return pl.concat(outs)

    def _chunk(self, df: pl.DataFrame) -> pl.DataFrame:
        f: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            enc = {}
            for fld in ("nm", "core", "sq", "ad"):
                L = FIELD_LEN[fld]
                enc[fld] = (to_bytes(df[fld], L), to_bytes(df[f"{fld}_s"], L))
            for fld, ((xa, la), (xb, lb)) in enc.items():
                for q in (2, 3):
                    j, ci, co = gram_overlap(xa, la, xb, lb, q)
                    f[f"{fld}_g{q}_jac"], f[f"{fld}_g{q}_cin"], f[f"{fld}_g{q}_cout"] = j, ci, co
            xp, lp = to_bytes(df["nm_pre"], FIELD_LEN["core"])
            _, ci, _ = gram_overlap(xp, lp, *enc["core"][1], 3)
            f["pre_core_g3_cin"] = ci

            (xa, la), (xb, lb) = enc["core"]
            ha, _ = token_hashes(xa, la, TOKENS["core"])
            hb, _ = token_hashes(xb, lb, TOKENS["core"])
            j, wj, wt, ws = token_overlap(ha, hb, self.idf_core)
            f.update(core_tok_jac=j, core_tok_wjac=wj, core_tok_wcov_t=wt, core_tok_wcov_s=ws)
            f["ntok_t"], f["ntok_s"] = (ha >= 0).sum(1), (hb >= 0).sum(1)

            (xa, la), (xb, lb) = enc["ad"]
            ha, numa = token_hashes(xa, la, TOKENS["ad"])
            hb, numb = token_hashes(xb, lb, TOKENS["ad"])
            j, wj, wt, ws = token_overlap(ha, hb, self.idf_ad)
            f.update(ad_tok_jac=j, ad_tok_wjac=wj, ad_tok_wcov_t=wt, ad_tok_wcov_s=ws)
            f["ad_ntok_t"], f["ad_ntok_s"] = (ha >= 0).sum(1), (hb >= 0).sum(1)
            na_h = torch.where(numa, ha, torch.full_like(ha, -1))
            nb_h = torch.where(numb, hb, torch.full_like(hb, -2))
            f["num_jac"], *_ = token_overlap(na_h, torch.where(nb_h == -2, -1, nb_h), None)
            for k in ("nm", "ad"):
                f[f"len_{k}_t"], f[f"len_{k}_s"] = enc[k][0][1], enc[k][1][1]
            f = {k: v.float().cpu().numpy() for k, v in f.items()}
            del enc

        rf = {
            "rf_nm_ratio": (df["nm"], df["nm_s"], fuzz.ratio),
            "rf_core_tset": (df["core"], df["core_s"], fuzz.token_set_ratio),
            "rf_ad_tset": (df["ad"], df["ad_s"], fuzz.token_set_ratio),
        }
        for k, (a, b, scorer) in rf.items():
            f[k] = np.asarray(cpdist(a.fill_null("").to_list(), b.fill_null("").to_list(),
                                     scorer=scorer, workers=WORKERS, dtype=np.float32))
        out = df.select(
            "tid", "s1", *BLOCK_FEATURES, "src", "n_alias", "n_domain", "n_indic", "ad_null", "ad_null_s",
            ((pl.col("ad_num") == pl.col("ad_num_s")) & (pl.col("ad_num") != "")).cast(pl.Int8).alias("num_first_eq"),
            ((pl.col("ad_num") == "") | (pl.col("ad_num_s") == "")).cast(pl.Int8).alias("num_missing"),
            "core_freq_s", "core_freq_t", *NUM_FEATURES, *SIB_FEATURES,
            (pl.col("core") == pl.col("core_s")).cast(pl.Int8).alias("core_eq"),
            (pl.col("sq") == pl.col("sq_s")).cast(pl.Int8).alias("sq_eq"),
        )
        cols = FEATURES
        if self.nn is not None:
            from src.er import nn as NN
            f["nn_p"] = NN.predict(self.nn, df)
            cols = FEATURES + ["nn_p"]
        return out.with_columns([pl.Series(k, v) for k, v in f.items()]).select("tid", "s1", *cols)
