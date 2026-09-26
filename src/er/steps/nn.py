"""Step 3a (optional): train the character-level neural matcher on reserved records.

    python -m src.er.steps.nn [--n-targets 400000 --epochs 2]

Uses training records with ``tid % 10 == NN_FOLD`` whose candidates contain no
validation Source-1 entity; XGBoost excludes that fold, so its ``nn_p`` feature is
always out-of-sample.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from src.er.resources import lower_priority, rss_gb

ART = Path("artifacts/er")
VAL_MOD = 100


def pair_text(c: pl.DataFrame, tg: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    return (c.join(tg.select(pl.col("id").alias("tid"), "nm", "ad"), on="tid")
            .join(s1.select(pl.col("id").alias("s1"), pl.col("nm").alias("nm_s"), pl.col("ad").alias("ad_s")), on="s1"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=400_000)
    ap.add_argument("--n-valid", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default=str(ART / "nn" / "matcher.pt"))
    args = ap.parse_args(argv)
    lower_priority()
    from src.er import nn as NN
    from src.er.nn import NN_FOLD
    from src.er.io import DATA_DIR, load_ground_truth

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s rss {rss_gb():.1f}GB] {m}", flush=True)
    gt = load_ground_truth(DATA_DIR).select("tid", "s1").with_columns(pl.lit(1, pl.Int8).alias("y"))
    cands = pl.scan_parquet(ART / "train_cands" / "part-*.parquet").select("tid", "s1")
    val_t = cands.filter(pl.col("s1") % VAL_MOD == 0).select("tid").unique()
    fold = (cands.filter(pl.col("tid") % 10 == NN_FOLD).select("tid").unique()
            .join(val_t, on="tid", how="anti").collect().sample(args.n_targets + args.n_valid, seed=7))
    s1 = pl.read_parquet(ART / "train_s1.parquet", columns=["id", "nm", "ad"])

    def build(tids: pl.DataFrame) -> pl.DataFrame:
        c = cands.join(tids.lazy(), on="tid", how="semi").collect()
        tg = pl.scan_parquet(ART / "train_tg.parquet").select("id", "nm", "ad").join(
            tids.lazy().rename({"tid": "id"}), on="id", how="semi").collect()
        return (pair_text(c, tg, s1).join(gt, on=["tid", "s1"], how="left")
                .with_columns(pl.col("y").fill_null(0)).sample(fraction=1.0, shuffle=True, seed=1))

    train_df = build(fold.head(args.n_targets))
    valid_df = build(fold.tail(args.n_valid))
    log(f"nn train pairs {train_df.height:,} (pos {train_df['y'].sum():,}) | valid pairs {valid_df.height:,}")
    model = NN.train_model(train_df, epochs=args.epochs, valid=valid_df, log=log)
    NN.save(model, args.out)
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
