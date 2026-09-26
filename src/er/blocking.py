"""Rare-key inverted-index candidate generation (target-centric, chunked polars joins).

Each record emits hashed blocking keys, always namespaced by country:

* ``n``  core-name tokens, plus the space-less name (so ``reliableanchorone.com``
         meets "Reliable Anchor One")
* ``a``  address tokens
* ``b``  adjacent address-token bigrams ("15277 spencer", "spencer st")

Source-1 is indexed.  A key shared by more than ``cap`` Source-1 records carries
little identity and is skipped, which bounds the join fan-out.  Every
(target, Source-1) pair sharing at least one key is scored by the summed IDF
(``ln(N / df)``) of its shared name keys and address keys, and the top-``k``
Source-1 records per target are kept.  Targets are processed in chunks so peak
memory is bounded by the chunk size, not by the ~10M-record target corpus.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import polars as pl


def make_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Return unique ``(id, h, kind)`` rows; ``kind`` 0 = name key, 1 = address key."""
    cty = pl.col("cty")
    name_tok = (
        df.select("id", "cty", pl.col("core").str.split(" ").alias("t"))
        .explode("t")
        .filter((pl.col("t").str.len_chars() >= 2) | pl.col("t").str.contains(r"^\d$"))
        .select("id", (cty + "|n|" + pl.col("t")).alias("k"), pl.lit(0, pl.Int8).alias("kind"))
    )
    name_sq = (
        df.filter(pl.col("core").str.contains(" "))
        .select("id", (cty + "|n|" + pl.col("sq")).alias("k"), pl.lit(0, pl.Int8).alias("kind"))
    )
    addr_tok = (
        df.select("id", "cty", pl.col("ad").str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
        .select("id", (cty + "|a|" + pl.col("t")).alias("k"), pl.lit(1, pl.Int8).alias("kind"))
    )
    # overlapping adjacent bigrams = non-overlapping pairs of s and of s-minus-first-token
    shifted = pl.col("ad").str.replace(r"^\S+\s*", "")
    addr_bi = (
        df.select(
            "id", "cty",
            pl.concat_list(
                pl.col("ad").str.extract_all(r"\S+ \S+"),
                shifted.str.extract_all(r"\S+ \S+"),
            ).alias("t"),
        )
        .explode("t")
        .filter(pl.col("t").is_not_null())
        .select("id", (cty + "|b|" + pl.col("t")).alias("k"), pl.lit(1, pl.Int8).alias("kind"))
    )
    keys = pl.concat([name_tok, name_sq, addr_tok, addr_bi])
    return keys.select("id", pl.col("k").hash(seed=17).alias("h"), "kind").unique(["id", "h"])


def build_index(s1: pl.DataFrame, cap: int) -> pl.DataFrame:
    """Source-1 inverted index: ``(h, s1, w, kind)`` for keys with ``df <= cap``."""
    keys = make_keys(s1).rename({"id": "s1"})
    n_by_cty = s1.group_by("cty").len()
    df_ = keys.group_by("h").agg(pl.len().alias("df"))
    keys = keys.join(df_, on="h").filter(pl.col("df") <= cap)
    # IDF relative to the whole Source-1 corpus (country sizes are similar)
    n = float(s1.height)
    keys = keys.with_columns((math.log(n) - pl.col("df").cast(pl.Float32).log()).cast(pl.Float32).alias("w"))
    return keys.select("h", "s1", "w", "kind").sort("h")


def generate_candidates(
    targets: pl.DataFrame,
    index: pl.DataFrame,
    k: int = 10,
    chunk: int = 200_000,
    verbose: bool = True,
) -> pl.DataFrame:
    """Top-``k`` Source-1 candidates per target.

    Returns ``tid, s1, sn, sa, nk, rk`` where ``sn``/``sa`` are the summed IDF of
    shared name/address keys, ``nk`` the number of shared keys and ``rk`` the
    rank (0 = best) of this Source-1 record among the target's candidates.
    """
    idx = index.lazy()
    out = []
    t0 = time.time()
    for start in range(0, targets.height, chunk):
        part = targets.slice(start, chunk)
        tk = make_keys(part).rename({"id": "tid"}).drop("kind")
        res = (
            tk.lazy()
            .join(idx, on="h", how="inner")
            .group_by("tid", "s1")
            .agg(
                pl.col("w").filter(pl.col("kind") == 0).sum().alias("sn"),
                pl.col("w").filter(pl.col("kind") == 1).sum().alias("sa"),
                pl.len().cast(pl.Int16).alias("nk"),
            )
            .with_columns((pl.col("sn") + pl.col("sa")).alias("_s"))
            .with_columns(
                pl.col("_s").rank("ordinal", descending=True).over("tid").cast(pl.Int16).alias("rk")
            )
            .filter(pl.col("rk") <= k)
            .with_columns(pl.col("rk") - 1)
            .drop("_s")
            .collect()
        )
        out.append(res)
        if verbose and (start // chunk) % 5 == 0:
            done = min(start + chunk, targets.height)
            print(f"  blocking {done:,}/{targets.height:,} targets  "
                  f"{sum(o.height for o in out):,} pairs  {time.time() - t0:.0f}s", flush=True)
    return pl.concat(out)
