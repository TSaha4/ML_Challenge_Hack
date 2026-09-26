"""Where does validation macro-F0.5 get lost?  (reads artifacts/er/val_scored.parquet)"""
import polars as pl
from src.er.io import load_ground_truth
from src.er import model as M
THR = 0.70
sc = pl.read_parquet("artifacts/er/val_scored.parquet")
gt = load_ground_truth("student_resource/dataset")
truth = gt.filter(pl.col("s1") % 100 == 0)
u = pl.read_parquet("artifacts/er/train_s1.parquet", columns=["id"]).filter(pl.col("id") % 100 == 0)["id"]
pred = M.assign(sc, THR)
base = M.macro_f05(pred, truth, u)
print("baseline", {k: round(v, 4) for k, v in base.items()})
n = u.len()
# 1) ids truly matched but never candidates
cand_true = sc.join(truth, on=["tid", "s1"], how="semi")
print("true pairs", truth.height, "in candidates", cand_true.height, f"({cand_true.height/truth.height:.4f})")
# error types on predicted pairs restricted to val S1
p = pred.join(pl.DataFrame({"s1": u}), on="s1", how="semi")
tp = p.join(truth, on=["tid", "s1"], how="semi")
fp = p.join(truth, on=["tid", "s1"], how="anti")
fp_t = fp.join(gt, on="tid", how="left", suffix="_true")
print("pred pairs", p.height, "TP", tp.height, "FP", fp.height,
      "| FP whose target is truly unmatched", fp_t["s1_true"].null_count(),
      "| FP whose target belongs to another S1", fp_t["s1_true"].is_not_null().sum())
# FN breakdown
fn = truth.join(p, on=["tid", "s1"], how="anti")
best = sc.sort("p", descending=True).unique("tid", keep="first")
fnb = fn.join(best.rename({"s1": "best_s1", "p": "best_p"}), on="tid", how="left").join(
      sc.rename({"p": "true_p"}), on=["tid", "s1"], how="left")
print("FN", fn.height,
      "| not a candidate", fnb["true_p"].null_count(),
      "| true S1 is argmax but p<thr", ((fnb["best_s1"] == fnb["s1"]) & (fnb["best_p"] < THR)).sum(),
      "| another S1 wins argmax", (fnb["best_s1"] != fnb["s1"]).sum())
# per-S1 score loss attribution
ps = (pl.DataFrame({"s1": u}).join(truth.group_by("s1").len("nt"), on="s1", how="left")
      .join(p.group_by("s1").len("np"), on="s1", how="left")
      .join(tp.group_by("s1").len("tp"), on="s1", how="left").fill_null(0))
b2 = 0.25
ps = ps.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
     .otherwise((1 + b2) * (pl.col("tp") / pl.col("np")) * (pl.col("tp") / pl.col("nt")) /
                (b2 * pl.col("tp") / pl.col("np") + pl.col("tp") / pl.col("nt"))).alias("f"))
ps = ps.with_columns(pl.when(pl.col("nt") == 0).then(pl.lit("singleton_with_FP"))
      .when(pl.col("np") == 0).then(pl.lit("matched_but_empty_pred"))
      .when(pl.col("tp") < pl.col("np")).then(pl.lit("has_FP"))
      .otherwise(pl.lit("FN_only")).alias("cat"))
print(ps.filter(pl.col("f") < 1).group_by("cat").agg(pl.len(), ((1 - pl.col("f")).sum() / n).round(4).alias("F_lost"))
      .sort("F_lost", descending=True))
print("singletons", (ps["nt"] == 0).sum(), "of", n)
