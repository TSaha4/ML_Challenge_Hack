"""Collective (cluster-aware) evidence from records competing for the same Source-1 entity.

A source perturbs a business's address/name *once*, and all of that source's copies
share the perturbation; unrelated noise records do not.  So for a candidate pair
(record t, Source-1 s) we look at the *siblings*: the other records whose top-ranked
candidate is also ``s``.  On train, a record whose house number disagrees with ``s``
is a true match 13% of the time when no sibling shares its number, but 64% / 87%
with one / two agreeing siblings.

``build_sibling_tables`` aggregates, per split, counts of rank-0 records per Source-1
entity keyed by hashed record values; ``sibling_features`` turns them into pair
features (excluding the record itself).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

KEYS = {
    "num": "ad_num",   # first house number
    "ad": "ad",        # full normalised address
    "core": "core",    # core name
    "nums": "ad_nums", # all numbers of the address
}
SIB_FEATURES = [
    "n_sib", "sib_same_num", "sib_same_ad", "sib_same_core", "sib_same_nums",
    "sib_same_src", "sib_same_src_ad", "sib_num_eq_s", "sib_frac_num_eq_s",
    "sib_same_num_frac", "sib_same_ad_frac",
]


def _hashes(df: pl.DataFrame) -> pl.DataFrame:
    """Hashed record values; empty strings -> null (never count as agreement)."""
    return df.with_columns([
        pl.when(pl.col(c) != "").then(pl.col(c).hash(seed=7)).otherwise(None).alias(f"h_{k}")
        for k, c in KEYS.items()
    ])


def build_sibling_tables(cands_glob: str, tg_path: str, s1_path: str, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    top = pl.scan_parquet(cands_glob).filter(pl.col("rk") == 0).select("tid", "s1")
    tg = pl.scan_parquet(tg_path).select(pl.col("id").alias("tid"), "src", *KEYS.values())
    sib = _hashes(top.join(tg, on="tid").collect(engine="streaming")).select(
        "tid", "s1", "src", *[f"h_{k}" for k in KEYS])
    s1 = _hashes(pl.read_parquet(s1_path, columns=["id", *KEYS.values()])).select(
        pl.col("id").alias("s1"), pl.col("h_num").alias("h_num_s"))

    sib.group_by("s1").agg(pl.len().cast(pl.Int32).alias("c")).write_parquet(out / "s1.parquet")
    sib.group_by("s1", "src").agg(pl.len().cast(pl.Int32).alias("c")).write_parquet(out / "s1_src.parquet")
    for k in KEYS:
        (sib.drop_nulls(f"h_{k}").group_by("s1", f"h_{k}").agg(pl.len().cast(pl.Int32).alias("c"))
         .write_parquet(out / f"s1_{k}.parquet"))
    (sib.drop_nulls("h_ad").group_by("s1", "src", "h_ad").agg(pl.len().cast(pl.Int32).alias("c"))
     .write_parquet(out / "s1_src_ad.parquet"))
    (sib.join(s1, on="s1").filter(pl.col("h_num") == pl.col("h_num_s"))
     .group_by("s1").agg(pl.len().cast(pl.Int32).alias("c")).write_parquet(out / "s1_numeq.parquet"))


class SiblingLookup:
    def __init__(self, table_dir: str | Path):
        d = Path(table_dir)
        self.t = {p.stem: pl.read_parquet(p) for p in d.glob("*.parquet")}

    def features(self, pairs: pl.DataFrame) -> pl.DataFrame:
        """``pairs``: tid, s1, rk, src, the KEYS columns of the record and ``num_eq_s_self``
        (1 when the record's house number equals the Source-1 one). Returns SIB_FEATURES."""
        p = _hashes(pairs).with_columns((pl.col("rk") == 0).cast(pl.Int32).alias("_self"))
        t = self.t

        def cnt(df, table, on, name):
            return df.join(t[table].rename({"c": name}), on=on, how="left").with_columns(pl.col(name).fill_null(0))

        p = cnt(p, "s1", ["s1"], "_n")
        p = cnt(p, "s1_src", ["s1", "src"], "_nsrc")
        for k in KEYS:
            p = cnt(p, f"s1_{k}", ["s1", f"h_{k}"], f"_{k}")
        p = cnt(p, "s1_src_ad", ["s1", "src", "h_ad"], "_srcad")
        p = cnt(p, "s1_numeq", ["s1"], "_numeq")
        me = pl.col("_self")
        has = lambda k: pl.col(f"h_{k}").is_not_null().cast(pl.Int32)  # self only counted if value present
        n_sib = (pl.col("_n") - me).clip(0)
        p = p.with_columns(
            n_sib.alias("n_sib"),
            (pl.col("_num") - me * has("num")).clip(0).alias("sib_same_num"),
            (pl.col("_ad") - me * has("ad")).clip(0).alias("sib_same_ad"),
            (pl.col("_core") - me * has("core")).clip(0).alias("sib_same_core"),
            (pl.col("_nums") - me * has("nums")).clip(0).alias("sib_same_nums"),
            (pl.col("_nsrc") - me).clip(0).alias("sib_same_src"),
            (pl.col("_srcad") - me * has("ad")).clip(0).alias("sib_same_src_ad"),
            # siblings agreeing with the Source-1 house number (self excluded if it agrees)
            (pl.col("_numeq") - me * pl.col("num_eq_s_self")).clip(0).alias("sib_num_eq_s"),
        )
        denom = pl.col("n_sib").clip(1).cast(pl.Float32)
        p = p.with_columns(
            (pl.col("sib_num_eq_s") / denom).alias("sib_frac_num_eq_s"),
            (pl.col("sib_same_num") / denom).alias("sib_same_num_frac"),
            (pl.col("sib_same_ad") / denom).alias("sib_same_ad_frac"),
        )
        return p.select("tid", "s1", *SIB_FEATURES)
