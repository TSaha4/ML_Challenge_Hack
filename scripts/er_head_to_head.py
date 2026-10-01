"""Head-to-head on hard-mode (test-like) validation: current model vs hard-mode model.

Both models score the same hard-mode calibration + holdout records with identical
features (hard-mode candidates and sibling tables). Each model's threshold is tuned on
the calibration fold; the decision uses the independent holdout fold only.

    ER_DATA_DIR=student_resource_sim/dataset ER_ARTIFACT_DIR=artifacts/er_sim \
    python scripts/er_head_to_head.py
"""
import json
import time
from pathlib import Path

import polars as pl

from src.er.artifacts import ART
from src.er.resources import lower_priority, rss_gb, wait_for_ram

lower_priority()
from src.er import model as M
from src.er.gpu_features import GpuFeaturizer
from src.er.io import DATA_DIR, load_ground_truth

MODELS = {"current (0.970 on LB)": "artifacts/er_fixed/model_fixed", "hard-mode": str(ART / "model_hard")}
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s rss {rss_gb():.1f}GB] {m}", flush=True)

truth = load_ground_truth(DATA_DIR)
s1 = pl.read_parquet(ART / "train_s1.parquet")
cal_u = s1.filter(pl.col("id") % 100 == 0)["id"]
hold_u = s1.filter(pl.col("id") % 100 == 4)["id"]
cands = pl.scan_parquet(ART / "train_cands" / "part-*.parquet")
reserved = (pl.col("s1") % 100).is_in([0, 4])
tids = pl.concat([cands.filter(reserved).select("tid").collect(),
                  truth.filter(reserved).select("tid")]).unique()
log(f"hard-mode eval records: {tids.height:,} | calibration S1 {cal_u.len():,} | holdout S1 {hold_u.len():,}")

boosters = {k: M.load(v) for k, v in MODELS.items()}
fz = GpuFeaturizer(s1, pl.read_parquet(ART / "train_s1stats.parquet"), sib_dir=ART / "train_sibs",
                   nn_path=ART / "nn" / "matcher.pt")
tg_scan = pl.scan_parquet(ART / "train_tg.parquet")
scored = {k: [] for k in MODELS}
for i in range(0, tids.height, 100_000):
    wait_for_ram()
    sub = tids.slice(i, 100_000)
    c = cands.join(sub.lazy(), on="tid", how="semi").collect()
    tg = tg_scan.join(sub.lazy().rename({"tid": "id"}), on="id", how="semi").collect()
    f = fz.featurize(c, tg)
    for k, (b, _) in boosters.items():
        scored[k].append(f.select("tid", "s1").with_columns(pl.Series("p", M.predict(b, f))))
log("scored")
del fz

report = {}
for k in MODELS:
    sc = pl.concat(scored[k])
    cal = sc.join(pl.DataFrame({"s1": cal_u}), on="s1", how="semi")
    # threshold tuned on hard-mode calibration (records of other S1s kept so argmax is honest)
    thr, _ = M.tune_threshold(sc, truth.filter(pl.col("s1") % 100 == 0), cal_u)
    own = boosters[k][1]["threshold"]
    res = {}
    for name, t in (("own_threshold", own), ("hardmode_tuned", thr)):
        res[name] = {"threshold": t,
                     "holdout": M.macro_f05(M.assign(sc, t), truth.filter(pl.col("s1") % 100 == 4), hold_u),
                     "calibration": M.macro_f05(M.assign(sc, t), truth.filter(pl.col("s1") % 100 == 0), cal_u)}
    report[k] = res
    for name, r in res.items():
        h = r["holdout"]
        log(f"{k:24s} {name:15s} thr={r['threshold']:.3f} | HOLDOUT F0.5 {h['macro_f05']:.4f} "
            f"(P {h['pair_precision']:.4f} R {h['pair_recall']:.4f} single {h['singleton_f05']:.4f}) | "
            f"calib {r['calibration']['macro_f05']:.4f}")
(ART / "head_to_head.json").write_text(json.dumps(report, indent=2, default=float))
log(f"saved {ART / 'head_to_head.json'}")
