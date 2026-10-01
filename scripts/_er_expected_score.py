"""Label-free expected macro-F0.5 of a submission, from (prior-shift-corrected) probabilities.
Per S1: records whose best candidate is that S1 are Bernoulli(p'). Kept = p >= thr.
E[F] ~ 1.25 E[TP] / (0.25 E[#true] + #kept) for kept>0; P(no true) for kept==0.
A constant share `miss` of true matches is never a candidate (measured on validation)."""
import glob, numpy as np, polars as pl
from src.er.resources import lower_priority; lower_priority()
from src.er.io import load_ground_truth
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ps", "scripts/_er_prior_shift.py")

def em_prior(p, pi_train, iters=200):
    p = np.clip(p, 1e-6, 1 - 1e-6); pi = pi_train
    for _ in range(iters):
        a = (pi / pi_train) * p; b = ((1 - pi) / (1 - pi_train)) * (1 - p)
        new = float(np.mean(a / (a + b)))
        if abs(new - pi) < 1e-7: break
        pi = new
    return pi

def adj(p, pt, pn):
    a = (pn / pt) * p; b = ((1 - pn) / (1 - pt)) * (1 - p); return a / (a + b)

def expected_f(best, universe, thr, miss):
    b = best.with_columns((pl.col("p") >= thr).alias("keep"))
    g = b.group_by("s1").agg(pl.col("q").sum().alias("et"),
                             pl.col("q").filter(pl.col("keep")).sum().alias("etp"),
                             pl.col("keep").sum().alias("k"),
                             (1 - pl.col("q")).clip(1e-9).log().sum().alias("lp0"))
    g = universe.join(g, on="s1", how="left").fill_null(0)
    et = pl.col("et") / (1 - miss)
    f = (pl.when(pl.col("k") == 0).then((pl.col("lp0").exp() * (-et * miss).exp()))
         .otherwise(1.25 * pl.col("etp") / (0.25 * et + pl.col("k"))))
    return g.select(f.mean()).item()

PT = 0.7233
gt = load_ground_truth("student_resource/dataset")
v = pl.read_parquet("artifacts/er_fixed/model_fixed/validation_scored.parquet")
vb = v.sort("p", descending=True).unique("tid", keep="first").with_columns(pl.col("p").alias("q"))
vu = pl.read_parquet("artifacts/er_fixed/train_s1.parquet", columns=["id"]).filter(pl.col("id") % 100 == 0).rename({"id": "s1"})
MISS = 0.0195   # true pairs never retrieved or lost to another S1 (validation measurement)
print("VALIDATION check: actual macro F0.5 at 0.697 = 0.9841 (from training log)")
for thr in (0.5, 0.697, 0.8, 0.9):
    print(f"  expected F0.5 at thr {thr}: {expected_f(vb, vu, thr, MISS):.4f}")

SD = "artifacts/er_fixed/test_scored/b0ce54191a07c8e1921bea1f1fcaf490568fe022378f50d5f6cad75dc5cf58c4"
tb = pl.concat([pl.read_parquet(f).sort("p", descending=True).unique("tid", keep="first") for f in sorted(glob.glob(SD + "/part-*.parquet"))])
tb = tb.join(pl.read_parquet("artifacts/er_fixed/test_tg.parquet", columns=["id", "cty"]).rename({"id": "tid"}), on="tid")
parts = []
for c in ("us", "india", "france"):
    d = tb.filter(pl.col("cty") == c); pn = em_prior(d["p"].to_numpy(), PT)
    parts.append(d.with_columns(pl.Series("q", adj(d["p"].to_numpy(), PT, pn))))
tq = pl.concat(parts)
tu = pl.read_parquet("artifacts/er_fixed/test_s1.parquet", columns=["id", "cty"]).rename({"id": "s1"})
print("\nTEST (leaderboard said 0.970 for thr 0.697):")
print("  without shift correction, expected F at 0.697:", round(expected_f(tb.with_columns(pl.col("p").alias("q")), tu.select("s1"), 0.697, MISS), 4))
for thr in (0.5, 0.6, 0.697, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97):
    row = [f"thr {thr:<5}: ALL {expected_f(tq, tu.select('s1'), thr, MISS):.4f}"]
    for c in ("us", "india", "france"):
        row.append(f"{c} {expected_f(tq.filter(pl.col('cty') == c), tu.filter(pl.col('cty') == c).select('s1'), thr, MISS):.4f}")
    print("  " + " | ".join(row))
