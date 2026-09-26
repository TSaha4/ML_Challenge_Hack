"""Step 3: featurise (GPU), train XGBoost (GPU) and tune the macro-F0.5 threshold.

    python -m src.er.steps.train [--n-train 500000]

Validation Source-1 entities are ``s1 % 100 == 0`` (a subset of the fold held out
from transliteration learning).  Every target that has one of them among its
candidates is scored, so false merges into validation entities are counted and
the macro F0.5 matches the leaderboard definition (singletons included).

RAM stays bounded: features are written to parquet in chunks, XGBoost streams them
into a GPU QuantileDMatrix, and validation is featurised + scored chunk by chunk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.er.resources import lower_priority, rss_gb, wait_for_ram

ART = Path("artifacts/er")
VAL_MOD = 100


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=500_000, help="training targets")
    ap.add_argument("--n-es", type=int, default=100_000, help="early-stopping targets")
    ap.add_argument("--chunk-targets", type=int, default=100_000)
    ap.add_argument("--model-dir", default=str(ART / "model"))
    ap.add_argument("--reuse-features", action="store_true")
    ap.add_argument("--train-files", type=int, default=None,
                    help="use only the first N training feature files (100k targets each)")
    ap.add_argument("--max-bin", type=int, default=128)
    ap.add_argument("--nn", default=None, help="neural matcher weights -> adds feature nn_p")
    args = ap.parse_args(argv)
    lower_priority()
    import xgboost as xgb
    from src.er import model as M
    from src.er.gpu_features import FEATURES, GpuFeaturizer
    from src.er.io import DATA_DIR, load_ground_truth

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s rss {rss_gb():.1f}GB] {m}", flush=True)

    gt = load_ground_truth(DATA_DIR)
    s1 = pl.read_parquet(ART / "train_s1.parquet")
    stats = pl.read_parquet(ART / "train_s1stats.parquet")
    cands = pl.scan_parquet(ART / "train_cands" / "part-*.parquet")
    tg_scan = pl.scan_parquet(ART / "train_tg.parquet")
    fz = GpuFeaturizer(s1, stats, sib_dir=ART / "train_sibs", nn_path=args.nn)
    pos = gt.select("tid", "s1").with_columns(pl.lit(1, pl.Int8).alias("y"))

    # --- target split --------------------------------------------------------
    val_s1 = s1.filter(pl.col("id") % VAL_MOD == 0)["id"]
    val_tids = pl.concat([
        cands.filter(pl.col("s1") % VAL_MOD == 0).select("tid").collect(),
        gt.filter(pl.col("s1") % VAL_MOD == 0).select("tid"),
    ]).unique()
    pool = tg_scan.select(pl.col("id").alias("tid")).collect().join(val_tids, on="tid", how="anti")
    if args.nn:  # records reserved for the neural matcher never train XGBoost
        from src.er.nn import NN_FOLD
        pool = pool.filter(pl.col("tid") % 10 != NN_FOLD)
    pool = pool.sample(args.n_train + args.n_es, seed=42)
    splits = {"train": pool.head(args.n_train), "es": pool.tail(args.n_es)}
    log(f"val S1 {val_s1.len():,} | val targets {val_tids.height:,} | "
        f"train targets {args.n_train:,} | es targets {args.n_es:,}")

    def feature_chunks(tids: pl.DataFrame):
        for i in range(0, tids.height, args.chunk_targets):
            wait_for_ram()
            sub = tids.slice(i, args.chunk_targets).lazy()
            c = cands.join(sub, on="tid", how="semi").collect()
            tg = tg_scan.join(sub.rename({"tid": "id"}), on="id", how="semi").collect()
            yield fz.featurize(c, tg)

    # --- features for train / early-stopping -> parquet ------------------------
    fdir = ART / "features"
    if not args.reuse_features:
        shutil.rmtree(fdir, ignore_errors=True)
        for name, tids in splits.items():
            d = fdir / name
            d.mkdir(parents=True)
            n = 0
            for j, f in enumerate(feature_chunks(tids)):
                f = f.join(pos, on=["tid", "s1"], how="left").with_columns(pl.col("y").fill_null(0))
                f.write_parquet(d / f"{j:04d}.parquet")
                n += f.height
            log(f"features[{name}]: {n:,} pairs")

    feat_names = FEATURES + (["nn_p"] if args.nn else [])

    class ParquetIter(xgb.DataIter):
        def __init__(self, files):
            self.files, self.i = files, 0
            super().__init__()

        def next(self, input_data):
            if self.i == len(self.files):
                return False
            f = pl.read_parquet(self.files[self.i])
            input_data(data=f.select([pl.col(c).cast(pl.Float32) for c in feat_names]).to_numpy(),
                       label=f["y"].to_numpy(), feature_names=feat_names)
            self.i += 1
            return True

        def reset(self):
            self.i = 0

    # free the featuriser (Source-1 text + sibling tables) while XGBoost builds its matrices
    del fz
    import gc
    gc.collect()
    train_files = sorted((fdir / "train").glob("*.parquet"))[: args.train_files]
    dtr = xgb.QuantileDMatrix(ParquetIter(train_files), max_bin=args.max_bin)
    des = xgb.QuantileDMatrix(ParquetIter(sorted((fdir / "es").glob("*.parquet"))), ref=dtr, max_bin=args.max_bin)
    log(f"train rows {dtr.num_row():,}  es rows {des.num_row():,}")
    booster = xgb.train({**M.XGB_PARAMS, "max_bin": args.max_bin}, dtr, 2000, evals=[(dtr, "train"), (des, "es")],
                        early_stopping_rounds=60, verbose_eval=100)
    del dtr, des
    log(f"trained on GPU: best iteration {booster.best_iteration}")
    imp = booster.get_score(importance_type="gain")
    log("top gain: " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(imp.items(), key=lambda x: -x[1])[:15]))

    # --- validation: featurise + score in chunks -------------------------------
    fz = GpuFeaturizer(s1, stats, sib_dir=ART / "train_sibs", nn_path=args.nn)
    scored = []
    for f in feature_chunks(val_tids):
        scored.append(f.select("tid", "s1").with_columns(pl.Series("p", M.predict(booster, f))))
    scored = pl.concat(scored)
    scored.write_parquet(ART / "val_scored.parquet")
    truth = gt.filter(pl.col("s1") % VAL_MOD == 0)
    thr, curve = M.tune_threshold(scored, truth, val_s1)
    best = next(r for r in curve if r["threshold"] == thr)
    oracle = scored.join(truth, on=["tid", "s1"], how="semi").with_columns(pl.lit(1.0).alias("p"))
    ceiling = M.macro_f05(M.assign(oracle, 0.5), truth, val_s1)
    log("threshold curve:\n" + "\n".join(
        f"  thr={r['threshold']:.3f}  F0.5={r['macro_f05']:.4f}  single={r['singleton_f05']:.4f}  "
        f"matched={r['matched_f05']:.4f}  P={r['pair_precision']:.4f}  R={r['pair_recall']:.4f}"
        for r in curve))
    log(f"BEST thr={thr}  macro F0.5={best['macro_f05']:.4f}  (candidate-set ceiling {ceiling['macro_f05']:.4f})")
    M.save(booster, {"threshold": thr, "val": best, "ceiling": ceiling, "val_mod": VAL_MOD,
                     "n_train_targets": args.n_train}, args.model_dir)
    (ART / "logs" / "threshold_curve.json").write_text(json.dumps(curve, indent=2))
    log(f"saved model -> {args.model_dir}")


if __name__ == "__main__":
    main()
