"""Pairwise features for (target, Source-1) candidate pairs.

String similarities are computed with ``rapidfuzz.process.cpdist`` (element-wise,
C++), so tens of millions of pairs featurise without a Python loop.  Nothing is
country-specific: the same features apply to US, India and the unseen France.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

WORKERS = int(os.environ.get("ER_WORKERS", "4"))

#: (output name, left col, right col, scorer)
_STRING_SIMS = [
    ("nm_ratio", "nm", "nm_s", fuzz.ratio),
    ("core_ratio", "core", "core_s", fuzz.ratio),
    ("core_tset", "core", "core_s", fuzz.token_set_ratio),
    ("core_tsort", "core", "core_s", fuzz.token_sort_ratio),
    ("core_partial", "core", "core_s", fuzz.partial_ratio),
    ("core_jw", "core", "core_s", JaroWinkler.normalized_similarity),
    ("sq_ratio", "sq", "sq_s", fuzz.ratio),
    ("sq_partial", "sq", "sq_s", fuzz.partial_ratio),
    ("pre_tset", "nm_pre", "core_s", fuzz.token_set_ratio),
    ("ad_ratio", "ad", "ad_s", fuzz.ratio),
    ("ad_tset", "ad", "ad_s", fuzz.token_set_ratio),
    ("ad_tsort", "ad", "ad_s", fuzz.token_sort_ratio),
    ("ad_partial", "ad", "ad_s", fuzz.partial_ratio),
    ("nums_tset", "ad_nums", "ad_nums_s", fuzz.token_set_ratio),
]

FEATURES = [
    # blocking statistics
    "sn", "sa", "nk", "rk", "score", "score_top", "score_rel", "score_gap", "n_cand",
    "s1_top_cnt", "s1_cand_cnt",
    # string similarities
    *[name for name, *_ in _STRING_SIMS],
    # structured comparisons
    "num_eq", "num_missing", "core_tok_jacc", "ntok_t", "ntok_s", "ad_ntok_t", "ad_ntok_s",
    # record flags
    "src", "n_alias", "n_domain", "n_indic", "ad_null", "ad_null_s", "pre_nonempty",
]


def _sim(a: pl.Series, b: pl.Series, scorer) -> np.ndarray:
    out = cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=WORKERS, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def s1_stats(cands: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Global per-Source-1 candidate statistics (computed once over all targets)."""
    return (
        cands.lazy()
        .group_by("s1")
        .agg(
            (pl.col("rk") == 0).cast(pl.Int32).sum().alias("s1_top_cnt"),
            pl.len().cast(pl.Int32).alias("s1_cand_cnt"),
        )
        .collect()
    )


def add_context(cands: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    """Per-target competition statistics plus the global per-Source-1 ``stats``.

    ``cands`` must contain *all* candidates of each target it contains.
    """
    c = cands.with_columns((pl.col("sn") + pl.col("sa")).alias("score"))
    c = c.with_columns(
        pl.col("score").max().over("tid").alias("score_top"),
        pl.len().over("tid").cast(pl.Int16).alias("n_cand"),
        pl.col("score").sort(descending=True).slice(1, 1).first().over("tid").fill_null(0.0).alias("_second"),
    )
    c = c.with_columns(
        (pl.col("score") / pl.col("score_top").clip(1e-6)).alias("score_rel"),
        # margin over the best competitor (negative when another S1 scores higher)
        pl.when(pl.col("rk") == 0)
        .then(pl.col("score") - pl.col("_second"))
        .otherwise(pl.col("score") - pl.col("score_top"))
        .alias("score_gap"),
    )
    return c.drop("_second").join(stats, on="s1", how="left")


def featurize(cands: pl.DataFrame, stats: pl.DataFrame, tg: pl.DataFrame, s1: pl.DataFrame,
              chunk_targets: int = 200_000):
    """Yield feature frames for ``cands`` in target-aligned chunks (bounded memory)."""
    tids = cands["tid"].unique(maintain_order=True)
    for start in range(0, tids.len(), chunk_targets):
        sub = cands.join(tids.slice(start, chunk_targets).to_frame(), on="tid", how="semi")
        yield build_features(add_context(sub, stats), tg, s1)


def build_features(cands: pl.DataFrame, tg: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """Join record text onto candidate pairs and compute the feature matrix.

    ``cands`` must already carry :func:`add_context` columns.  Returns ``tid, s1``
    plus :data:`FEATURES` (float32 / small ints), in the same row order.
    """
    t_cols = ["id", "nm", "core", "sq", "nm_pre", "ad", "ad_nums", "ad_num",
              "src", "n_alias", "n_domain", "n_indic", "ad_null"]
    s_cols = ["id", "nm", "core", "sq", "ad", "ad_nums", "ad_num", "ad_null"]
    df = (
        cands.join(tg.select(t_cols).rename({"id": "tid"}), on="tid", how="left")
        .join(s1.select(s_cols).rename({c: f"{c}_s" for c in s_cols if c != "id"}).rename({"id": "s1"}),
              on="s1", how="left")
    )
    sims = {name: _sim(df[a], df[b], scorer) for name, a, b, scorer in _STRING_SIMS}

    core_t = pl.col("core").str.split(" ")
    core_s = pl.col("core_s").str.split(" ")
    df = df.with_columns(
        *[pl.Series(k, v) for k, v in sims.items()],
        ((pl.col("ad_num") == pl.col("ad_num_s")) & (pl.col("ad_num") != "")).cast(pl.Int8).alias("num_eq"),
        ((pl.col("ad_num") == "") | (pl.col("ad_num_s") == "")).cast(pl.Int8).alias("num_missing"),
        (core_t.list.set_intersection(core_s).list.len()
         / core_t.list.set_union(core_s).list.len().clip(1)).cast(pl.Float32).alias("core_tok_jacc"),
        core_t.list.len().cast(pl.Int16).alias("ntok_t"),
        core_s.list.len().cast(pl.Int16).alias("ntok_s"),
        pl.col("ad").str.split(" ").list.len().cast(pl.Int16).alias("ad_ntok_t"),
        pl.col("ad_s").str.split(" ").list.len().cast(pl.Int16).alias("ad_ntok_s"),
        pl.col("ad_null_s").alias("ad_null_s"),
        (pl.col("nm_pre") != "").cast(pl.Int8).alias("pre_nonempty"),
    )
    return df.select("tid", "s1", *FEATURES)
