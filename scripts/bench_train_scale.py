"""
Real-scale audit benchmark for the Source-1 <-> Source-2/3 training pipeline.

Runs the *real* pipeline logic on bounded slices to measure, before spending hours
on the full run:

  counts        exact dataset scale and ground-truth statistics;
  equivalence   DuckDBBlocker vs AdaptiveBlocker candidate equality;
  ram           in-memory inverted-index cost per indexed target;
  scale         full-size on-disk index, querying the same Source-1 sample while
                the index grows (250k -> 10.3M targets) to measure candidate
                fan-out, positive recall and throughput;
  e2e           Phase 12-14 end to end on a Source-1 slice: dataset build,
                entity-aware split, LightGBM fit, F_0.5 threshold sweep, bundle
                persistence and inference-compatibility check.

Usage (each stage writes its results into the same JSON report):

    python scripts/bench_train_scale.py --stage counts
    python scripts/bench_train_scale.py --stage equivalence
    python scripts/bench_train_scale.py --stage ram
    python scripts/bench_train_scale.py --stage scale
    python scripts/bench_train_scale.py --stage e2e --max-s1 3000
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gc
import json
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import psutil

from src.blocking import AdaptiveBlocker, DuckDBBlocker, extract_blocking_keys
from src.features import extract_pairwise_features
from src.inference import load_model_bundle, normalize_rows, save_model_bundle
from src.pipeline import compute_corpus_token_idf, stream_tsv_chunks
from src.training import (
    GroundTruthStore,
    TargetRecordStore,
    build_training_dataset,
    entity_aware_split,
    iter_s1_batches,
    split_masks,
    train_lgbm_classifier,
    tune_threshold_f05,
)

TRAIN = Path("student_resource/dataset/train")
TEST = Path("student_resource/dataset/test")
CHECKPOINTS = Path("artifacts/checkpoints")
REPORT_PATH = CHECKPOINTS / "train_scale_benchmark.json"
BLOCK_INDEX = CHECKPOINTS / "bench_blocking.duckdb"
TARGET_STORE = CHECKPOINTS / "bench_targets.duckdb"
GT_STORE = CHECKPOINTS / "bench_gt.duckdb"

PROC = psutil.Process()
#: test-split sizes, measured with `--stage counts` (used for extrapolation notes)
TEST_TARGETS = 4_887_273 + 5_082_316


def rss_mb() -> float:
    return round(PROC.memory_info().rss / 1e6, 1)


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as handle:
        while True:
            buf = handle.read(1 << 22)
            if not buf:
                break
            n += buf.count(b"\n")
    return max(0, n - 1)


def load_report() -> dict:
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_report(report: dict) -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[report] {REPORT_PATH}")


def _save_scale_progress(report: dict, results: list) -> None:
    """Persist completed scale-checkpoint rows immediately.

    Keeps a partially completed ``--stage scale`` run from losing its measured
    rows if a later checkpoint fails (e.g. on an OOM), and lets a ``--reuse``
    rerun resume from them.
    """
    report["scale"] = {"checkpoints": list(results), "in_progress": True}
    save_report(report)


def get_idf(sample_size: int = 20_000) -> dict:
    return compute_corpus_token_idf(str(TRAIN / "train_source1.tsv"), sample_size=sample_size)


# ---------------------------------------------------------------------------
# Stage: counts
# ---------------------------------------------------------------------------
def stage_counts(report: dict) -> dict:
    print("=== stage: counts ===", flush=True)
    files = {}
    for folder in (TRAIN, TEST):
        for path in sorted(folder.glob("*.tsv")):
            rows = count_lines(path)
            files[f"{folder.name}/{path.name}"] = rows
            print(f"  {folder.name}/{path.name:28s} {rows:>12,}", flush=True)

    gt = pl.read_csv(TRAIN / "train_ground_truth.tsv", separator="\t")
    matched = gt["matched_entity_ids"].fill_null("")
    exploded = (
        gt.select(pl.col("source1_entity_id"),
                  pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
    )
    prefixes = exploded["matched_entity_ids"].str.slice(0, 3).value_counts()
    card = gt.select(
        (matched.str.count_matches(",") + (matched != "").cast(pl.Int32)).alias("card")
    )["card"]
    stats = {
        "S1_train_entities": int(len(gt)),
        "S2_train_records": files["train/train_source2.tsv"],
        "S3_train_records": files["train/train_source3.tsv"],
        "targets_train_total": files["train/train_source2.tsv"] + files["train/train_source3.tsv"],
        "S1_test_entities": files["test/test_source1.tsv"],
        "targets_test_total": files["test/test_source2.tsv"] + files["test/test_source3.tsv"],
        "gt_pairs": int(len(exploded)),
        "gt_singletons": int((matched == "").sum()),
        "gt_non_singleton": int((matched != "").sum()),
        "gt_s2_pairs": int(prefixes.filter(pl.col("matched_entity_ids") == "S2-")["count"][0]),
        "gt_s3_pairs": int(prefixes.filter(pl.col("matched_entity_ids") == "S3-")["count"][0]),
        "gt_max_cardinality": int(card.max()),
        "gt_mean_cardinality": round(float(card.mean()), 3),
        "gt_p95_cardinality": float(card.quantile(0.95)),
    }
    for key, value in stats.items():
        print(f"  {key:24s} {value:,}" if isinstance(value, int) else f"  {key:24s} {value}")
    report["counts"] = stats
    return stats


# ---------------------------------------------------------------------------
# Stage: equivalence (DuckDBBlocker vs AdaptiveBlocker)
# ---------------------------------------------------------------------------
def stage_equivalence(report: dict, pool: int = 200_000, n_s1: int = 300) -> dict:
    print(f"=== stage: equivalence (pool={pool:,} targets, {n_s1} S1) ===", flush=True)
    idf = get_idf()
    memory = AdaptiveBlocker(max_block_size=500, max_candidates_per_entity=100)
    db_path = CHECKPOINTS / "equiv_blocking.duckdb"
    if db_path.exists():
        os.remove(db_path)
    disk = DuckDBBlocker(db_path, memory_limit="512MB", temp_directory=str(CHECKPOINTS))

    for chunk in stream_tsv_chunks(str(TRAIN / "train_source2.tsv"), chunk_size=50_000, n_rows=pool):
        records = normalize_rows(chunk)
        memory.fit(records, idf_dict=idf)
        disk.fit(records, idf_dict=idf)
        del records, chunk
    disk.finalize()

    s1_sample = next(iter(iter_s1_batches(TRAIN, s1_chunk_size=n_s1, max_s1=n_s1, s1_offset=1_000_000)))
    t0 = time.time()
    mem_hits = {rec["entity_id"]: memory.query(rec, idf_dict=idf) for rec in s1_sample}
    t_mem = time.time() - t0
    t0 = time.time()
    dsk_hits = disk.query_batch(s1_sample, idf_dict=idf)
    t_dsk = time.time() - t0
    disk.close()

    same_set = same_meta = 0
    meta_mismatch = []
    for rec in s1_sample:
        s1_id = rec["entity_id"]
        a = {c["candidate_id"]: (c["channel_count"], c["min_rank"]) for c in mem_hits[s1_id]}
        b = {c["candidate_id"]: (c["channel_count"], c["min_rank"]) for c in dsk_hits[s1_id]}
        if set(a) == set(b):
            same_set += 1
        if a == b:
            same_meta += 1
        elif len(meta_mismatch) < 3:
            diff = {k: (a.get(k), b.get(k)) for k in (set(a) | set(b)) if a.get(k) != b.get(k)}
            meta_mismatch.append({"s1_id": s1_id, "n_memory": len(a), "n_duckdb": len(b),
                                  "n_diff": len(diff), "sample": dict(list(diff.items())[:3])})

    result = {
        "pool": pool,
        "n_s1": len(s1_sample),
        "identical_candidate_sets": same_set,
        "identical_sets_and_metadata": same_meta,
        "mismatch_examples": meta_mismatch,
        "memory_seconds": round(t_mem, 2),
        "duckdb_seconds": round(t_dsk, 2),
        "rss_mb": rss_mb(),
    }
    print(f"  identical candidate sets  : {same_set}/{len(s1_sample)}", flush=True)
    print(f"  identical sets + metadata : {same_meta}/{len(s1_sample)}", flush=True)
    print(f"  query time memory={t_mem:.2f}s duckdb={t_dsk:.2f}s", flush=True)
    report["equivalence"] = result
    return result


# ---------------------------------------------------------------------------
# Stage: ram (in-memory inverted-index cost per target)
# ---------------------------------------------------------------------------
def stage_ram(report: dict, pools=(100_000, 300_000, 600_000)) -> dict:
    print("=== stage: ram (in-memory inverted index) ===", flush=True)
    idf = get_idf()
    measurements = []
    for pool in pools:
        gc.collect()
        blocker = AdaptiveBlocker(max_block_size=500, max_candidates_per_entity=100)
        base_rss = PROC.memory_info().rss
        tracemalloc.start()
        t0 = time.time()
        n = 0
        for chunk in stream_tsv_chunks(str(TRAIN / "train_source2.tsv"),
                                       chunk_size=50_000, n_rows=pool):
            records = normalize_rows(chunk)
            blocker.fit(records, idf_dict=idf)
            n += len(records)
            del records, chunk
        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_delta = PROC.memory_info().rss - base_rss
        row = {
            "targets": n,
            "keys": len(blocker.index),
            "index_mb_traced": round(peak / 1e6, 1),
            "bytes_per_target": round(peak / max(1, n), 1),
            "rss_delta_mb": round(rss_delta / 1e6, 1),
            "seconds": round(elapsed, 1),
            "records_per_second": round(n / elapsed),
        }
        measurements.append(row)
        print(f"  pool={n:>9,}  traced={row['index_mb_traced']:>8.1f} MB "
              f"({row['bytes_per_target']:.0f} B/target)  rss_delta={row['rss_delta_mb']:.0f} MB  "
              f"{row['records_per_second']:,} rec/s", flush=True)
        del blocker
        gc.collect()

    per_target = float(np.mean([m["bytes_per_target"] for m in measurements]))
    rec_per_second = float(np.mean([m["records_per_second"] for m in measurements]))
    counts = report.get("counts", {})
    train_targets = counts.get("targets_train_total", 10_320_219)
    test_targets = counts.get("targets_test_total", 9_969_589)
    result = {
        "measurements": measurements,
        "mean_bytes_per_target": per_target,
        "mean_index_records_per_second": rec_per_second,
        "projected_train_index_gb": round(per_target * train_targets / 1e9, 2),
        "projected_test_index_gb": round(per_target * test_targets / 1e9, 2),
        "projected_train_index_minutes": round(train_targets / rec_per_second / 60, 1),
        "projected_test_index_minutes": round(test_targets / rec_per_second / 60, 1),
    }
    print(f"  => {per_target:.0f} B/target | train index {result['projected_train_index_gb']} GB "
          f"| test index {result['projected_test_index_gb']} GB", flush=True)
    report["ram"] = result
    return result


# ---------------------------------------------------------------------------
# Stage: scale (full on-disk index, same Source-1 sample while the pool grows)
# ---------------------------------------------------------------------------
def _measure_sample(blocker, s1_sample, gt_map, n_indexed, idf, label) -> dict:
    """Candidate/positive statistics for the Source-1 sample against the current index."""
    t0 = time.time()
    hits = blocker.query_batch(s1_sample, idf_dict=idf)
    query_seconds = time.time() - t0

    candidates = 0
    positives_found = 0
    s1_with_candidates = 0
    s1_with_positive = 0
    max_cands = 0
    for rec in s1_sample:
        truth = gt_map.get(rec["entity_id"], set())
        cands = hits[rec["entity_id"]]
        candidates += len(cands)
        max_cands = max(max_cands, len(cands))
        if cands:
            s1_with_candidates += 1
        found = sum(1 for c in cands if c["candidate_id"] in truth)
        positives_found += found
        if found:
            s1_with_positive += 1

    gt_pairs = sum(len(v) for v in gt_map.values())
    n_s1 = len(s1_sample)
    row = {
        "label": label,
        "targets_indexed": int(n_indexed),
        "s1_entities": n_s1,
        "gt_pairs_in_sample": gt_pairs,
        "candidates": candidates,
        "candidates_per_s1": round(candidates / max(1, n_s1), 2),
        "max_candidates_seen": max_cands,
        "s1_with_candidates_pct": round(100 * s1_with_candidates / max(1, n_s1), 2),
        "positives_found": positives_found,
        "positive_recall": round(positives_found / gt_pairs, 4) if gt_pairs else 0.0,
        "s1_with_positive_pct": round(100 * s1_with_positive / max(1, n_s1), 2),
        "query_seconds": round(query_seconds, 1),
        "query_ms_per_s1": round(1000 * query_seconds / max(1, n_s1), 2),
        "rss_mb": rss_mb(),
    }
    print(f"  [{label}] targets={n_indexed:,} | cands/S1={row['candidates_per_s1']} | "
          f"recall={row['positive_recall']:.3f} | query={query_seconds:.0f}s | "
          f"rss={row['rss_mb']:.0f}MB", flush=True)
    return row


def stage_scale(
    report: dict,
    n_s1: int = 20_000,
    s1_offset: int = 1_000_000,
    checkpoints=(250_000, 1_000_000, 2_500_000, 5_000_000),
    reuse: bool = False,
    feature_probe_s1: int = 200,
) -> dict:
    """Index the *complete* training target pool and query a fixed Source-1 sample."""
    print(f"=== stage: scale (S1 sample={n_s1:,} @ offset {s1_offset:,}) ===", flush=True)
    idf = get_idf()

    gt_store = GroundTruthStore(GT_STORE)
    loaded = gt_store.load(TRAIN / "train_ground_truth.tsv")
    if loaded:
        print(f"  ground truth loaded: {loaded:,} rows", flush=True)

    target_store = TargetRecordStore(TARGET_STORE, reset=not reuse)
    blocker = DuckDBBlocker(
        BLOCK_INDEX, max_block_size=500, max_candidates_per_entity=100,
        memory_limit="1024MB", threads=4, temp_directory=str(CHECKPOINTS),
    )
    n_existing = 0
    if reuse and BLOCK_INDEX.exists():
        n_existing, n_keys = blocker.load_existing()
        if n_existing:
            print(f"  reusing blocking index: {n_existing:,} targets / {n_keys:,} key rows", flush=True)

    s1_sample = next(iter(iter_s1_batches(TRAIN, s1_chunk_size=n_s1, max_s1=n_s1,
                                          s1_offset=s1_offset)))
    sample_ids = [rec["entity_id"] for rec in s1_sample]
    gt_map = gt_store.fetch(sample_ids)

    # Resume-aware bookkeeping.  On --reuse the already-indexed records must be
    # skipped while streaming (re-fitting them would duplicate key rows),
    # checkpoints at or below the resumed index size are not re-measured, and
    # checkpoint rows saved by a previous (possibly crashed) run are kept.
    skip_records = n_existing
    results = list(report.get("scale", {}).get("checkpoints", [])) if reuse else []
    indexed = n_existing
    next_cp = bisect.bisect_right(checkpoints, indexed)
    started = time.time()
    for source_num in (2, 3):
        path = TRAIN / f"train_source{source_num}.tsv"
        source_started = time.time()
        for chunk in stream_tsv_chunks(str(path), chunk_size=50_000):
            if skip_records:
                if skip_records >= len(chunk):
                    skip_records -= len(chunk)
                    continue
                chunk = chunk.slice(skip_records, len(chunk) - skip_records)
                skip_records = 0
            records = normalize_rows(chunk)
            del chunk
            blocker.fit(records, idf_dict=idf)
            target_store.add(records)
            indexed += len(records)
            del records
            gc.collect()
            while next_cp < len(checkpoints) and indexed >= checkpoints[next_cp]:
                blocker.finalize()
                results.append(_measure_sample(blocker, s1_sample, gt_map, indexed, idf,
                                               f"{checkpoints[next_cp]:,}"))
                next_cp += 1
                _save_scale_progress(report, results)
        print(f"  train_source{source_num}.tsv indexed: {indexed:,} cumulative "
              f"({time.time() - source_started:.0f}s for this source)", flush=True)

    blocker.finalize()
    # Always take a final full-scale row unless the last recorded row already
    # covers this index size.  (The old `next_cp < len(checkpoints)` guard
    # silently skipped the ~10.3M-target row once every checkpoint had fired.)
    final_label = f"{indexed:,}"
    if not results or results[-1].get("label") != final_label:
        results.append(_measure_sample(blocker, s1_sample, gt_map, indexed, idf, final_label))
        _save_scale_progress(report, results)

    # feature-extraction throughput at full scale (real target records)
    probe = s1_sample[:feature_probe_s1]
    hits = blocker.query_batch(probe, idf_dict=idf)
    needed = {c["candidate_id"] for cands in hits.values() for c in cands}
    lookup = target_store.fetch(needed)
    pairs = 0
    t_feat = time.time()
    for rec in probe:
        for cand in hits[rec["entity_id"]]:
            cand_rec = lookup.get(cand["candidate_id"])
            if cand_rec is None:
                continue
            extract_pairwise_features(
                rec, cand_rec,
                meta={"channel_count": cand["channel_count"], "min_rank": cand["min_rank"]},
                idf_dict=idf,
            )
            pairs += 1
    feature_seconds = time.time() - t_feat
    feature_rate = pairs / max(1e-9, feature_seconds)

    index_bytes = BLOCK_INDEX.stat().st_size if BLOCK_INDEX.exists() else 0
    store_bytes = TARGET_STORE.stat().st_size if TARGET_STORE.exists() else 0
    result = {
        "n_s1_sample": len(s1_sample),
        "s1_offset": s1_offset,
        "reused_targets": int(n_existing),
        "checkpoints": results,
        "index_seconds_total": round(time.time() - started, 1),
        "targets_per_second": round(indexed / max(1e-9, time.time() - started), 1),
        "feature_probe_pairs": pairs,
        "feature_pairs_per_second": round(feature_rate, 1),
        "feature_us_per_pair": round(1e6 / max(1e-9, feature_rate), 2),
        "blocking_index_size_mb": round(index_bytes / 1e6, 1),
        "target_store_size_mb": round(store_bytes / 1e6, 1),
        "rss_mb_peak": rss_mb(),
    }
    print(f"  index built: {indexed:,} targets in {result['index_seconds_total']:.0f}s "
          f"({result['targets_per_second']:,} rec/s) | block index "
          f"{result['blocking_index_size_mb']} MB | features {feature_rate:,.0f} pairs/s",
          flush=True)
    report["scale"] = result
    blocker.close()
    gt_store.close()
    target_store.close()
    return result


# ---------------------------------------------------------------------------
# Stage: e2e (Phase 12 -> 14 on a Source-1 slice, with bundle compat check)
# ---------------------------------------------------------------------------
def stage_e2e(
    report: dict,
    max_s1: int = 3_000,
    s1_offset: int = 1_000_000,
    neg_ratio: int = 2,
    target_limit: int = 250_000,
) -> dict:
    print(f"=== stage: e2e (max_s1={max_s1:,}, neg_ratio={neg_ratio}) ===", flush=True)
    idf = get_idf()
    gt_store = GroundTruthStore(GT_STORE)
    gt_store.load(TRAIN / "train_ground_truth.tsv")

    reuse = BLOCK_INDEX.exists() and TARGET_STORE.exists()
    if reuse:
        blocker = DuckDBBlocker(BLOCK_INDEX, max_block_size=500, max_candidates_per_entity=100,
                                memory_limit="1024MB", threads=4, temp_directory=str(CHECKPOINTS))
        n_existing, _ = blocker.load_existing()
        reuse = n_existing > 0
    if reuse:
        target_store = TargetRecordStore(TARGET_STORE, reset=False)
        print(f"  reusing {n_existing:,} indexed targets from the scale stage", flush=True)
    else:
        blocker = DuckDBBlocker(BLOCK_INDEX, max_block_size=500, max_candidates_per_entity=100,
                                memory_limit="1024MB", threads=4, temp_directory=str(CHECKPOINTS))
        target_store = TargetRecordStore(TARGET_STORE, reset=True)
        print(f"  no reusable index - building a {target_limit:,}-target demo pool", flush=True)
        from src.training import index_targets
        index_targets(TRAIN, idf, target_store, blocker, target_limit=target_limit)

    frame, y, stats = build_training_dataset(
        TRAIN, idf, blocker, target_store, gt_store,
        max_s1=max_s1, s1_offset=s1_offset, neg_ratio=neg_ratio, singleton_neg_ratio=1,
        out_parquet=str(CHECKPOINTS / "bench_e2e_dataset.parquet"), return_frame=True,
        verbose=True,
    )
    processed = stats.pop("processed_s1_ids")
    feature_cols = stats["feature_columns"]

    train_ids, val_ids = entity_aware_split(processed, test_size=0.25, random_state=42)
    train_mask, val_mask = split_masks(frame, train_ids, val_ids)
    split_report = {
        "train_entities": int(len(train_ids)),
        "val_entities": int(len(val_ids)),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positives": int(y[train_mask].sum()),
        "val_positives": int(y[val_mask].sum()),
        "ids_shared_between_sides": len(set(train_ids.tolist()) & set(val_ids.tolist())),
    }
    print(f"  dataset rows={len(frame):,} (pos={int(y.sum()):,}) | "
          f"train rows={split_report['train_rows']:,} | val rows={split_report['val_rows']:,} | "
          f"shared S1 ids={split_report['ids_shared_between_sides']}", flush=True)

    model = train_lgbm_classifier(
        frame[train_mask], y[train_mask], feature_cols,
        frame[val_mask], y[val_mask], verbose=False,
    )
    val_frame = frame[val_mask]
    gt_val = gt_store.fetch(list(val_ids))
    tune = tune_threshold_f05(model, val_frame, val_frame["s1_id"].to_numpy(), gt_val,
                              feature_cols, verbose=True)

    bundle_dir = save_model_bundle(model, feature_cols, idf, tune["best_threshold"],
                                   str(CHECKPOINTS / "bench_model_bundle"))
    loaded_model, loaded_features, loaded_idf, loaded_threshold = load_model_bundle(str(bundle_dir))

    s1_probe = next(iter(iter_s1_batches(TRAIN, 1, 1, 0)))[0]
    cand_probe_id = str(frame["cand_id"].to_numpy()[0])
    cand_probe = target_store.fetch({cand_probe_id})[cand_probe_id]
    probe_features = extract_pairwise_features(
        s1_probe, cand_probe, meta={"channel_count": 1, "min_rank": 0}, idf_dict=idf
    )
    probability = float(loaded_model.predict_proba(
        pd.DataFrame([probe_features])[loaded_features]
    )[0, 1])
    compat = {
        "bundle_dir": str(bundle_dir),
        "n_features": len(loaded_features),
        "feature_cols_match_training": loaded_features == list(feature_cols),
        "feature_schema_matches_extractor": set(loaded_features) == set(probe_features),
        "n_idf_tokens": len(loaded_idf),
        "threshold": loaded_threshold,
        "model_class": type(loaded_model).__name__,
        "has_predict_proba": hasattr(loaded_model, "predict_proba"),
        "probe_probability": round(probability, 6),
        "inference_command": (
            f"python -m src.inference --model-dir {bundle_dir} --output-dir output "
            "--validator student_resource/utils/validate_submission.py"
        ),
    }
    print(f"  bundle: {compat['n_features']} features | idf={compat['n_idf_tokens']:,} tokens | "
          f"threshold={compat['threshold']} | schema ok="
          f"{compat['feature_schema_matches_extractor']}", flush=True)

    result = {
        "dataset": stats,
        "split": split_report,
        "threshold": {k: v for k, v in tune.items() if k != "sweep"},
        "threshold_sweep": tune["sweep"],
        "compat": compat,
        "feature_importance_top10": sorted(
            zip(feature_cols, [int(v) for v in model.feature_importances_]),
            key=lambda kv: -kv[1],
        )[:10],
    }
    report["e2e"] = result
    blocker.close()
    gt_store.close()
    target_store.close()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Real-scale audit benchmark for training")
    parser.add_argument("--stage", default="all",
                        choices=["counts", "equivalence", "ram", "scale", "e2e", "all"])
    parser.add_argument("--reuse", action="store_true",
                        help="Reuse an existing bench blocking index / target store")
    parser.add_argument("--n-s1", type=int, default=20_000,
                        help="Source-1 entities sampled for the scale stage")
    parser.add_argument("--s1-offset", type=int, default=1_000_000,
                        help="Offset into train_source1.tsv for all samples")
    parser.add_argument("--max-s1", type=int, default=3_000,
                        help="Source-1 entities used by the e2e stage")
    parser.add_argument("--neg-ratio", type=int, default=2)
    parser.add_argument("--target-limit", type=int, default=250_000,
                        help="Target pool size if no full index is available for e2e")
    args = parser.parse_args(argv)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    report = load_report()
    report["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["rss_mb_start"] = rss_mb()

    def run(stage_name, fn):
        t0 = time.time()
        fn()
        save_report(report)
        print(f"[stage {stage_name}] done in {time.time() - t0:.0f}s "
              f"(rss {rss_mb():.0f} MB)", flush=True)

    if args.stage in ("counts", "all"):
        run("counts", lambda: stage_counts(report))
    if args.stage in ("equivalence", "all"):
        run("equivalence", lambda: stage_equivalence(report))
    if args.stage in ("ram", "all"):
        run("ram", lambda: stage_ram(report))
    if args.stage in ("scale", "all"):
        run("scale", lambda: stage_scale(report, n_s1=args.n_s1, s1_offset=args.s1_offset,
                                        reuse=args.reuse))
    if args.stage in ("e2e", "all"):
        run("e2e", lambda: stage_e2e(report, max_s1=args.max_s1, s1_offset=args.s1_offset,
                                     neg_ratio=args.neg_ratio, target_limit=args.target_limit))

    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["rss_mb_end"] = rss_mb()
    save_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
