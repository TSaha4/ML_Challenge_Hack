"""Research: per-entity expected-F0.5 decision vs a global threshold (on saved val scores)."""
import sys
import numpy as np
import polars as pl
from src.er.resources import lower_priority; lower_priority()
from src.er import model as M
from src.er.io import load_ground_truth

path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/er_v2/val_scored.parquet"
sc = pl.read_parquet(path)
gt = load_ground_truth("student_resource/dataset")
truth = gt.filter(pl.col("s1") % 100 == 0)
u = pl.read_parquet("artifacts/er/train_s1.parquet", columns=["id"]).filter(pl.col("id") % 100 == 0)["id"]
best = sc.sort("p", descending=True).unique("tid", keep="first")          # argmax S1 per record
print("global threshold 0.70:", round(M.macro_f05(M.assign(sc, 0.70), truth, u)["macro_f05"], 4))

def expected_f_select(best, miss=0.0, beta=0.5, pmin=0.01):
    b2 = beta * beta
    g = (best.filter(pl.col("p") >= pmin).sort(["s1", "p"], descending=[False, True])
         .with_columns(pl.col("p").cum_sum().over("s1").alias("ctp"),
                       pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                       pl.col("p").sum().over("s1").alias("et"),
                       (1 - pl.col("p")).log().sum().over("s1").alias("lp0")))
    g = g.with_columns(((1 + b2) * pl.col("ctp") / (b2 * (pl.col("et") + miss) + pl.col("k"))).alias("ef"))
    # expected F of predicting nothing = P(no true match among candidates) * P(no missed match)
    ent = g.group_by("s1").agg(pl.col("ef").max().alias("ef_best"),
                               pl.col("k").filter(pl.col("ef") == pl.col("ef").max()).first().alias("kbest"),
                               (pl.col("lp0").first().exp() * np.exp(-miss)).alias("ef0"))
    keep = ent.filter(pl.col("ef_best") > pl.col("ef0")).select("s1", "kbest")
    return g.join(keep, on="s1").filter(pl.col("k") <= pl.col("kbest")).select("tid", "s1", "p")

for miss in (0.0, 0.05, 0.1, 0.2, 0.4):
    for pmin in (0.01, 0.05, 0.2):
        m = M.macro_f05(expected_f_select(best, miss, pmin=pmin), truth, u)
        print(f"expected-F  miss={miss:<4} pmin={pmin:<4} F0.5={m['macro_f05']:.4f}  P={m['pair_precision']:.4f} R={m['pair_recall']:.4f} single={m['singleton_f05']:.4f}")
