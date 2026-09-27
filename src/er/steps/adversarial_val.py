"""Adversarial validation: how different are test Source-1 entities from train ones?

    python -m src.er.steps.adversarial_val [--model-dir artifacts/er_fixed/model_fixed]

Entity-level (the metric is a per-Source-1 macro average). Features use no ground
truth: candidate density around the entity, how generic its name is within its
country, address completeness, and the share of its top-ranked records that lack an
address. A shallow XGBoost separates train (0) from test (1) entities; AUC near 0.5
means no shift. The classifier's odds p/(1-p) then reweight the calibration entities
so their mix matches test, giving a test-like macro F0.5 for the current model.
Only US/India are compared (France has no train entities to weight).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.er.artifacts import ART
from src.er.resources import lower_priority

FEATS = ["s1_top_cnt", "s1_cand_cnt", "name_ntok", "name_freq", "ad_ntok", "has_num",
         "ad_null_s", "top_null_share", "top_mean_score", "india"]


def entity_features(split: str) -> pl.DataFrame:
    s1 = pl.read_parquet(ART / f"{split}_s1.parquet", columns=["id", "cty", "core", "ad", "ad_num", "ad_null"])
    freq = s1.group_by("cty", "core").agg(pl.len().alias("name_freq"))
    stats = pl.read_parquet(ART / f"{split}_s1stats.parquet")
    top = (pl.scan_parquet(ART / f"{split}_cands" / "part-*.parquet").filter(pl.col("rk") == 0)
           .select("tid", "s1", (pl.col("sn") + pl.col("sa")).alias("score"))
           .join(pl.scan_parquet(ART / f"{split}_tg.parquet").select(pl.col("id").alias("tid"), "ad_null"), on="tid")
           .group_by("s1").agg(pl.col("ad_null").cast(pl.Float32).mean().alias("top_null_share"),
                               pl.col("score").mean().alias("top_mean_score"))
           .collect(engine="streaming"))
    return (s1.join(freq, on=["cty", "core"]).rename({"id": "s1"})
            .join(stats, on="s1", how="left").join(top, on="s1", how="left")
            .select("s1", "cty",
                    pl.col("s1_top_cnt").fill_null(0), pl.col("s1_cand_cnt").fill_null(0),
                    pl.col("core").str.split(" ").list.len().alias("name_ntok"), "name_freq",
                    pl.col("ad").str.split(" ").list.len().alias("ad_ntok"),
                    (pl.col("ad_num") != "").cast(pl.Int8).alias("has_num"),
                    pl.col("ad_null").alias("ad_null_s"),
                    pl.col("top_null_share").fill_null(-1), pl.col("top_mean_score").fill_null(0),
                    (pl.col("cty") == "india").cast(pl.Int8).alias("india")))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ART / "model_fixed"))
    ap.add_argument("--n", type=int, default=300_000, help="entities sampled per split")
    args = ap.parse_args(argv)
    lower_priority()
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from src.er import model as M
    from src.er.io import DATA_DIR, load_ground_truth

    tr = entity_features("train")
    te = entity_features("test").filter(pl.col("cty").is_in(["us", "india"]))
    both = pl.concat([tr.sample(args.n, seed=1).with_columns(pl.lit(0).alias("y")),
                      te.sample(args.n, seed=1).with_columns(pl.lit(1).alias("y"))]).sample(fraction=1.0, seed=2)
    X, y = both.select(FEATS).to_numpy().astype(np.float32), both["y"].to_numpy()
    cut = int(0.7 * len(y))
    params = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.1, "device": "cuda",
              "tree_method": "hist", "nthread": 2, "eval_metric": "auc"}
    bst = xgb.train(params, xgb.DMatrix(X[:cut], y[:cut], feature_names=FEATS), 300)
    p = bst.predict(xgb.DMatrix(X[cut:], feature_names=FEATS))
    auc = roc_auc_score(y[cut:], p)
    gain = bst.get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    print(f"ADVERSARIAL AUC (train vs test entities, US+India): {auc:.4f}")
    print("feature importance (share of gain):")
    for k, v in sorted(gain.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v / total:6.3f}")
    print("\nfeature means  train | test:")
    for f in FEATS:
        print(f"  {f:16s} {tr[f].mean():9.3f} | {te[f].mean():9.3f}")

    # --- test-like reweighting of the calibration entities --------------------------------
    booster, meta = M.load(args.model_dir)
    thr = meta["threshold"]
    scored = pl.read_parquet(Path(args.model_dir) / "validation_scored.parquet")
    truth = load_ground_truth(DATA_DIR).filter(pl.col("s1") % 100 == 0)
    cal = tr.filter(pl.col("s1") % 100 == 0)
    w_p = bst.predict(xgb.DMatrix(cal.select(FEATS).to_numpy().astype(np.float32), feature_names=FEATS))
    w = np.clip(w_p / (1 - w_p), 0.05, 20.0)
    pred = M.assign(scored, thr).select("tid", "s1")
    u = pl.DataFrame({"s1": cal["s1"]})
    tp = pred.join(truth, on=["s1", "tid"]).group_by("s1").len("tp")
    npd = pred.group_by("s1").len("np")
    nt = truth.group_by("s1").len("nt")
    e = u.join(tp, on="s1", how="left").join(npd, on="s1", how="left").join(nt, on="s1", how="left").fill_null(0)
    P = pl.col("tp") / pl.col("np").clip(1)
    R = pl.col("tp") / pl.col("nt").clip(1)
    f = (pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * P * R / (0.25 * P + R)))
    fe = e.with_columns(f.alias("f"), pl.Series("w", w))
    plain = fe["f"].mean()
    rew = float((fe["f"] * fe["w"]).sum() / fe["w"].sum())
    ess = float(fe["w"].sum() ** 2 / (fe["w"] ** 2).sum())
    print(f"\ncalibration macro F0.5 (current model, thr {thr:.3f}): plain {plain:.4f} | "
          f"TEST-LIKE reweighted {rew:.4f}  (effective sample size {ess:,.0f} of {len(w):,})")
    out = {"auc": auc, "importance": {k: v / total for k, v in gain.items()},
           "calibration_f05": plain, "reweighted_f05": rew, "ess": ess, "threshold": thr}
    Path(ART / "adversarial_val.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
