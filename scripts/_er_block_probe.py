import polars as pl, time, sys
from src.er.io import load_ground_truth
from src.er.blocking import build_index, generate_candidates
pl.Config.set_tbl_rows(30)
s1=pl.read_parquet("artifacts/er/train_s1.parquet")
tg=pl.read_parquet("artifacts/er/train_tg.parquet").sample(int(__import__("os").environ.get("NS","300000")),seed=0)
gt=load_ground_truth("student_resource/dataset").join(tg.select(pl.col("id").alias("tid")),on="tid",how="semi")
print("gt pairs in sample",gt.height)
for cap in [int(x) for x in sys.argv[1:]]:
    t=time.time(); idx=build_index(s1,cap); print(f"cap={cap} index rows {idx.height:,} build {time.time()-t:.0f}s")
    t=time.time(); c=generate_candidates(tg,idx,k=20,chunk=100_000,verbose=False); print(f"  cands {c.height:,} ({c.height/tg.height:.1f}/target) {time.time()-t:.0f}s")
    j=gt.join(c,on=["tid","s1"],how="left")
    print("  " + "  ".join(f"R@{k}={(j['rk'].fill_null(99)<k).mean():.4f}" for k in (1,2,3,5,10,20)))
    tj=j.join(tg.select(pl.col("id").alias("tid"),"n_indic","ad_null","cty"),on="tid")
    print(tj.group_by("cty","n_indic","ad_null").agg(pl.len(),(pl.col("rk").fill_null(99)<10).mean().alias("r10"),(pl.col("rk").fill_null(99)<1).mean().alias("r1")).sort("len",descending=True))
    del idx
