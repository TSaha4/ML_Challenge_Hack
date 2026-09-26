"""Temporary audit helper: exact dataset scale + ground-truth statistics."""
import collections
import time
from pathlib import Path

import polars as pl

TRAIN = Path("student_resource/dataset/train")
TEST = Path("student_resource/dataset/test")


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(1 << 22)
            if not buf:
                break
            n += buf.count(b"\n")
    return max(0, n - 1)  # minus header


for folder in (TRAIN, TEST):
    for p in sorted(folder.glob("*.tsv")):
        t0 = time.time()
        print(f"{p.name:28s} rows={count_lines(p):>10,}  ({time.time()-t0:.1f}s)", flush=True)

print("\n=== ground truth profiling ===")
t0 = time.time()
gt = pl.read_csv(TRAIN / "train_ground_truth.tsv", separator="\t")
print(f"gt rows: {len(gt):,} ({time.time()-t0:.1f}s)")
ids = gt["matched_entity_ids"].fill_null("")
n_matches = ids.str.count_matches(",").sum() + (ids != "").sum()
non_singleton = (ids != "").sum()
print(f"non-singleton S1 : {non_singleton:,} ({non_singleton/len(gt)*100:.2f}%)")
print(f"total GT matches : {n_matches:,}  (avg {n_matches/max(1,non_singleton):.2f} per non-singleton)")

exploded = gt.select(
    pl.col("source1_entity_id"),
    pl.col("matched_entity_ids").fill_null("").str.split(","),
).explode("matched_entity_ids")
exploded = exploded.filter(pl.col("matched_entity_ids") != "")
exploded = exploded.with_columns(pl.col("matched_entity_ids").str.strip_chars())
print(f"exploded pair rows: {len(exploded):,}")
print("source prefix counts:",
      exploded["matched_entity_ids"].str.slice(0, 3).value_counts().to_dict(as_series=False))

# target reuse (hubs) + multi-source coverage
per_target = exploded.group_by("matched_entity_ids").len()
print("targets referenced:", f"{len(per_target):,}", "| max reuse:", int(per_target["len"].max()))
print("reuse histogram (top 8):", sorted(collections.Counter(per_target["len"].to_list()).items())[:8])

both_sources = exploded.with_columns(
    pl.col("matched_entity_ids").str.slice(0, 3).alias("src")
).group_by("source1_entity_id").agg(pl.col("src").n_unique().alias("n_src"))
print("S1 with both S2 and S3:", int((both_sources["n_src"] > 1).sum()))
print("S1 with only one source:", int((both_sources["n_src"] == 1).sum()))

card = gt.select(
    pl.col("matched_entity_ids").fill_null("").str.count_matches(",").add(
        (pl.col("matched_entity_ids").fill_null("") != "").cast(pl.Int32)
    ).alias("card")
)
print("cardinality histogram (top 10):", sorted(collections.Counter(card["card"].to_list()).items())[:10])
print(f"cardinality: max={card['card'].max()} p95={card['card'].quantile(0.95)} mean={card['card'].mean():.3f}")

uniq_s1 = gt["source1_entity_id"].n_unique()
print(f"unique S1 in GT: {uniq_s1:,} (rows {len(gt):,}) -> duplicate S1 rows? {'YES' if uniq_s1 != len(gt) else 'no'}")

s1_ids = set(pl.read_csv(TRAIN / "train_source1.tsv", separator="\t", columns=["entity_id"])["entity_id"].to_list())
print(f"S1 rows in source1: {len(s1_ids):,} | GT-only ids: {len(set(gt['source1_entity_id'].to_list()) - s1_ids):,} | "
      f"source1-only ids: {len(s1_ids - set(gt['source1_entity_id'].to_list())):,}")
