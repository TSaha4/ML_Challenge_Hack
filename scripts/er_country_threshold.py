"""Write a submission from cached test scores with per-country thresholds.

    ER_ARTIFACT_DIR=artifacts/er_fixed python scripts/er_country_threshold.py \
        --scores <test_scored dir> --default 0.697 --france 0.50 --out-dir output_fr050
"""
import argparse, glob, json
from pathlib import Path
import polars as pl
from src.er.artifacts import ART
from src.er.resources import lower_priority
from src.er.steps.predict import write_lists

lower_priority()
ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True); ap.add_argument("--default", type=float, required=True)
ap.add_argument("--france", type=float, required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--candidate-from", default=None, help="reuse this identical, already-validated candidate_pairs.tsv (cheap)")
a = ap.parse_args()
cty = pl.read_parquet(ART / "test_tg.parquet", columns=["id", "cty"]).rename({"id": "tid"})
best = []
for f in sorted(glob.glob(a.scores + "/part-*.parquet")):
    b = pl.read_parquet(f).sort(["p", "s1"], descending=[True, False]).unique("tid", keep="first").join(cty, on="tid")
    thr = pl.when(pl.col("cty") == "france").then(a.france).otherwise(a.default)
    best.append(b.filter(pl.col("p") >= thr).select("tid", "s1"))
best = pl.concat(best)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
s1 = pl.read_parquet(ART / "test_s1.parquet", columns=["id", "entity_id"])
tg = pl.scan_parquet(ART / "test_tg.parquet").select("id", "entity_id")
write_lists(s1, best.lazy(), "source1_entity_id\tmatched_entity_ids\n", out / "matching_results.tsv", target_ids=tg)
if a.candidate_from:  # candidates depend only on the scores, not the threshold
    import shutil
    shutil.copyfile(a.candidate_from, out / "candidate_pairs.tsv")
else:
    scored = pl.scan_parquet(a.scores + "/part-*.parquet").select("tid", "s1")
    write_lists(s1, scored, "source1_entity_id\tcandidate_entity_ids\n", out / "candidate_pairs.tsv", target_ids=tg)
json.dump({"default_threshold": a.default, "france_threshold": a.france, "links": best.height,
           "scores": a.scores}, open(out / "submission_manifest.json", "w"), indent=2)
print("links", best.height)
