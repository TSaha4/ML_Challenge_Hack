"""Step 4: score the test candidates and write the two submission files.

    python -m scripts.er_predict [--threshold 0.6]

Streams the test candidate parts (1M targets each): GPU featurisation -> GPU
XGBoost scoring -> per-target argmax assignment above the tuned threshold.
Writes ``output/matching_results.tsv`` and ``output/candidate_pairs.tsv`` (the exact
pair set the model scored), one row per test Source-1 entity in file order, then
runs the official validator.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from src.er.resources import lower_priority, rss_gb, wait_for_ram

ART = Path("artifacts/er")


def write_lists(s1_order: pl.DataFrame, pairs: pl.LazyFrame, header: str, path: Path, chunk: int = 200_000):
    """One TSV row per Source-1 id (file order); ids comma-joined, sorted, no duplicates."""
    from src.er.io import TARGET_BASE

    grouped = (pairs.group_by("s1").agg(pl.col("tid").unique().sort()).collect(engine="streaming"))
    as_id = (pl.lit("S") + (pl.element() // TARGET_BASE).cast(pl.Utf8) + pl.lit("-")
             + (pl.element() % TARGET_BASE).cast(pl.Utf8))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        for i in range(0, s1_order.height, chunk):
            sub = s1_order.slice(i, chunk).join(grouped, left_on="id", right_on="s1", how="left")
            sub = sub.with_columns(pl.col("tid").list.eval(as_id).list.join(",").fill_null("").alias("ids"))
            lines = sub.select((pl.lit("S1-") + pl.col("id").cast(pl.Utf8) + pl.lit("\t") + pl.col("ids")).alias("l"))
            fh.write("\n".join(lines["l"].to_list()) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ART / "model"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--chunk-targets", type=int, default=100_000)
    args = ap.parse_args(argv)
    lower_priority()
    from src.er import model as M
    from src.er.gpu_features import GpuFeaturizer

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s rss {rss_gb():.1f}GB] {m}", flush=True)
    booster, meta = M.load(args.model_dir)
    thr = args.threshold if args.threshold is not None else meta["threshold"]
    log(f"model best_iteration={meta['best_iteration']}  threshold={thr}")

    s1 = pl.read_parquet(ART / "test_s1.parquet")
    fz = GpuFeaturizer(s1, pl.read_parquet(ART / "test_s1stats.parquet"))
    s1_order = s1.select("id")
    tg_scan = pl.scan_parquet(ART / "test_tg.parquet")
    parts = sorted((ART / "test_cands").glob("part-*.parquet"))

    scored_dir = ART / "test_scored"
    scored_dir.mkdir(exist_ok=True)
    for i, part_file in enumerate(parts):
        out = scored_dir / f"part-{i:03d}.parquet"
        if out.exists():
            continue  # resumable
        cands = pl.read_parquet(part_file)
        tids = cands["tid"].unique(maintain_order=True).to_frame()
        res = []
        for j in range(0, tids.height, args.chunk_targets):
            wait_for_ram()
            sub = tids.slice(j, args.chunk_targets)
            c = cands.join(sub, on="tid", how="semi")
            tg = tg_scan.join(sub.lazy().rename({"tid": "id"}), on="id", how="semi").collect()
            f = fz.featurize(c, tg)
            res.append(f.select("tid", "s1").with_columns(pl.Series("p", M.predict(booster, f))))
        pl.concat(res).write_parquet(out)
        log(f"scored part {i + 1}/{len(parts)}: {cands.height:,} pairs")
        del cands, res

    scored = pl.scan_parquet(scored_dir / "part-*.parquet")
    best = (scored.sort("p", descending=True).group_by("tid").first()
            .filter(pl.col("p") >= thr).select("tid", "s1"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wait_for_ram()
    write_lists(s1_order, best, "source1_entity_id\tmatched_entity_ids\n", out_dir / "matching_results.tsv")
    log("wrote matching_results.tsv")
    wait_for_ram()
    write_lists(s1_order, scored.select("tid", "s1"), "source1_entity_id\tcandidate_entity_ids\n",
                out_dir / "candidate_pairs.tsv")
    log("wrote candidate_pairs.tsv")

    m = pl.read_csv(out_dir / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    nonempty = m["matched_entity_ids"].fill_null("") != ""
    log(f"S1 rows {m.height:,}; with matches {nonempty.sum():,} ({nonempty.mean():.1%}); "
        f"matched ids {m['matched_entity_ids'].fill_null('').str.count_matches(',').sum() + nonempty.sum():,}")

    cmd = [sys.executable, "student_resource/utils/validate_submission.py",
           "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "candidate_pairs.tsv"),
           "--test-dir", "student_resource/dataset/test"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    log(f"validator exit code {r.returncode}")


if __name__ == "__main__":
    main()
