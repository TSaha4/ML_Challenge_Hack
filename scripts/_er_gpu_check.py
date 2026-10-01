"""Equivalence + recall check: GPU blocker vs CPU blocker on a 20k-target slice."""
import polars as pl, time, torch
from src.er.resources import lower_priority, rss_gb
lower_priority()
from src.er.gpu_blocking import GpuIndex
from src.er.blocking import build_index, generate_candidates
from src.er.io import load_ground_truth
cols=["id","cty","core","sq","ad"]
s1=pl.read_parquet("artifacts/er/train_s1.parquet",columns=cols)
tg=pl.scan_parquet("artifacts/er/train_tg.parquet").select(cols).slice(3_000_000,20_000).collect()
t=time.time(); gi=GpuIndex(s1,cap=500); print(f"gpu index {gi.n_rows:,} rows {time.time()-t:.0f}s  rss {rss_gb():.2f}GB  vram {torch.cuda.memory_allocated()/2**30:.2f}GB",flush=True)
t=time.time(); g=gi.candidates(tg,k=10); print(f"gpu cands {g.height:,} {time.time()-t:.1f}s  peak vram {torch.cuda.max_memory_allocated()/2**30:.2f}GB",flush=True)
del gi; torch.cuda.empty_cache()
idx=build_index(s1,500)
t=time.time(); c=generate_candidates(tg,idx,k=10,verbose=False); print(f"cpu cands {c.height:,} {time.time()-t:.1f}s",flush=True)
del idx
j=c.join(g,on=["tid","s1"],how="full",suffix="_g")
print("pairs only cpu",j["tid_g"].null_count(),"only gpu",j["tid"].null_count())
jj=c.join(g,on=["tid","s1"]); print("max |sn diff|",(jj["sn"]-jj["sn_g"]).abs().max(),"max |sa diff|",(jj["sa"]-jj["sa_g"]).abs().max(), "nk eq",(jj["nk"]==jj["nk_g"]).mean())
gt=load_ground_truth("student_resource/dataset").join(tg.select(pl.col("id").alias("tid")),on="tid",how="semi")
for name,x in (("cpu",c),("gpu",g)):
    r=gt.join(x,on=["tid","s1"],how="left")["rk"].fill_null(99)
    print(name, "R@1",(r<1).mean(),"R@10",(r<10).mean())
print("rss",rss_gb())
