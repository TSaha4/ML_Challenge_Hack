"""Prior (label) shift estimation, Saerens et al. 2002 EM, at the record level.
A record's 'label' = its best-scoring S1 candidate is a true match. The model's
probabilities are calibrated to the training prior; EM re-estimates the prior on a
new population from the probabilities alone."""
import numpy as np, polars as pl
from src.er.resources import lower_priority; lower_priority()
from src.er.io import load_ground_truth

def em_prior(p, pi_train, iters=200):
    p = np.clip(p, 1e-6, 1 - 1e-6); pi = pi_train
    for _ in range(iters):
        a = (pi / pi_train) * p; b = ((1 - pi) / (1 - pi_train)) * (1 - p)
        new = float(np.mean(a / (a + b)))
        if abs(new - pi) < 1e-7: break
        pi = new
    return pi

def adjust(p, pi_train, pi_new):
    a = (pi_new / pi_train) * p; b = ((1 - pi_new) / (1 - pi_train)) * (1 - p)
    return a / (a + b)

gt = load_ground_truth("student_resource/dataset")
v = pl.read_parquet("artifacts/er_fixed/model_fixed/validation_scored.parquet")
vb = v.sort("p", descending=True).unique("tid", keep="first").join(
    gt.with_columns(pl.lit(1).alias("y")), on=["tid", "s1"], how="left").with_columns(pl.col("y").fill_null(0))
pi_true_val = vb["y"].mean()
print(f"VALIDATION records: true prior (best candidate is a true match) = {pi_true_val:.4f}; mean p = {vb['p'].mean():.4f}")
# pretend the training prior is the val truth; EM should keep it (calibration check)
print(f"  EM on validation starting from its own prior -> {em_prior(vb['p'].to_numpy(), pi_true_val):.4f}")
# robustness check: simulate a prior shift on validation by dropping 50% of true records
rng = np.random.default_rng(0)
keep = (vb["y"] == 0).to_numpy() | (rng.random(vb.height) < 0.5)
vs = vb.filter(pl.Series(keep))
print(f"  simulated shift: true prior {vs['y'].mean():.4f} | EM estimate {em_prior(vs['p'].to_numpy(), pi_true_val):.4f}")

import glob
SD = "artifacts/er_fixed/test_scored/b0ce54191a07c8e1921bea1f1fcaf490568fe022378f50d5f6cad75dc5cf58c4"
tb = pl.concat([pl.read_parquet(f).sort("p", descending=True).unique("tid", keep="first")
                for f in sorted(glob.glob(SD + "/part-*.parquet"))])
cty = pl.read_parquet("artifacts/er_fixed/test_tg.parquet", columns=["id", "cty"]).rename({"id": "tid"})
tb = tb.join(cty, on="tid")
print(f"\nTEST records: {tb.height:,}")
for c in (None, "us", "india", "france"):
    d = tb if c is None else tb.filter(pl.col("cty") == c)
    pi = em_prior(d["p"].to_numpy(), pi_true_val)
    f = (pi / pi_true_val) / ((1 - pi) / (1 - pi_true_val))          # odds multiplier
    thr_equiv = (0.697 / 0.303) / f; thr_equiv = thr_equiv / (1 + thr_equiv)
    print(f"  {c or 'ALL':7s} EM prior {pi:.4f} (train {pi_true_val:.4f}) | odds x{f:.3f} | "
          f"0.697 on adjusted p == raw threshold {thr_equiv:.3f}")
