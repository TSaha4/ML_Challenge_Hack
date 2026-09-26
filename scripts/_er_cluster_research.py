"""Research: do noise records form their own clusters that disagree with the S1 record?"""
import polars as pl
from src.er.resources import lower_priority, rss_gb; lower_priority()
from src.er.io import load_ground_truth
gt = load_ground_truth("student_resource/dataset")
top = pl.scan_parquet("artifacts/er/train_cands/part-*.parquet").filter(pl.col("rk") == 0).select("tid", "s1").collect(engine="streaming")
tg = pl.read_parquet("artifacts/er/train_tg.parquet", columns=["id", "core", "ad_num", "ad", "ad_null"]).rename({"id": "tid"})
s1 = pl.read_parquet("artifacts/er/train_s1.parquet", columns=["id", "core", "ad_num"]).rename({"id": "s1", "core": "core_s", "ad_num": "num_s"})
d = (top.join(tg, on="tid").join(s1, on="s1")
     .join(gt.with_columns(pl.lit(1).alias("y")), on=["tid", "s1"], how="left").with_columns(pl.col("y").fill_null(0)))
del tg, top
print(f"rows {d.height:,}  rss {rss_gb():.1f}GB  base P(true|rank0)={d['y'].mean():.3f}")
d = d.with_columns(
    (pl.len().over("s1", "ad_num") - 1).alias("sib_same_num"),
    (pl.len().over("s1", "core") - 1).alias("sib_same_core"),
    (pl.len().over("s1") - 1).alias("n_sib"),
    (pl.col("ad_num") == pl.col("num_s")).alias("num_eq"),
    (pl.col("core") == pl.col("core_s")).alias("core_eq"),
)
has_num = d.filter((pl.col("ad_num") != "") & (pl.col("num_s") != ""))
print("\n--- house number disagrees with S1: does agreement with siblings predict noise? ---")
print(has_num.filter(~pl.col("num_eq")).group_by(pl.col("sib_same_num").clip(0, 3)).agg(pl.len(), pl.col("y").mean().round(3).alias("P_true")).sort("sib_same_num"))
print("--- house number agrees with S1 ---")
print(has_num.filter(pl.col("num_eq")).group_by(pl.col("sib_same_num").clip(0, 3)).agg(pl.len(), pl.col("y").mean().round(3).alias("P_true")).sort("sib_same_num"))
print("\n--- core name differs from S1: does agreement with siblings predict noise? ---")
print(d.filter(~pl.col("core_eq")).group_by(pl.col("sib_same_core").clip(0, 3)).agg(pl.len(), pl.col("y").mean().round(3).alias("P_true")).sort("sib_same_core"))
print("\n--- number of records competing for the same S1 (rank-0) ---")
print(d.group_by(pl.col("n_sib").clip(0, 8)).agg(pl.len(), pl.col("y").mean().round(3).alias("P_true")).sort("n_sib"))
