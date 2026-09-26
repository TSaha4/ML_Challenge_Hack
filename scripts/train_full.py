"""
Real-scale training driver: build the competition training sample, fit the model,
tune the F_0.5 threshold and persist the inference bundle.

    # full run (all 2.2M Source-1 entities, all 10.3M targets; needs ~8 GB free RAM)
    python scripts/train_full.py --model-dir artifacts/models/lgb_f05

    # recommended first pass on this machine (~1.8 GB free RAM): entity-level
    # subsample that keeps every labelled pair of the sampled entities
    python scripts/train_full.py --max-s1 500000 --model-dir artifacts/models/lgb_f05

Then regenerate the submission (identical schema to the notebook pipeline):

    python -m src.inference --model-dir artifacts/models/lgb_f05 --output-dir output \
        --validator student_resource/utils/validate_submission.py

Everything is streamed: the inverted index and the normalised target records live in
DuckDB on disk, ground truth is joined per batch and only the labelled sample
(all recoverable positives + a bounded number of negatives per entity) is kept in RAM.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference import load_model_bundle, save_model_bundle
from src.pipeline import compute_corpus_token_idf
from src.training import (
    GroundTruthStore,
    TargetRecordStore,
    build_training_dataset,
    entity_aware_split,
    index_targets,
    mine_hard_negatives,
    save_json,
    split_masks,
    train_lgbm_classifier,
    tune_threshold_f05,
)
from src.blocking import DuckDBBlocker

TRAIN_DIR = Path("student_resource/dataset/train")
CHECKPOINTS = Path("artifacts/checkpoints")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Full-scale training for the ML Challenge 2026")
    parser.add_argument("--train-dir", default=str(TRAIN_DIR))
    parser.add_argument("--model-dir", default="artifacts/models/lgb_f05",
                        help="Bundle directory written by save_model_bundle")
    parser.add_argument("--dataset-out", default=str(CHECKPOINTS / "train_dataset.parquet"),
                        help="Feature-matrix checkpoint (written incrementally)")
    parser.add_argument("--metrics-out", default=str(CHECKPOINTS / "train_metrics.json"))
    parser.add_argument("--index-db", default=str(CHECKPOINTS / "train_blocking.duckdb"))
    parser.add_argument("--target-db", default=str(CHECKPOINTS / "train_targets.duckdb"))
    parser.add_argument("--gt-db", default=str(CHECKPOINTS / "train_gt.duckdb"))
    parser.add_argument("--max-s1", type=int, default=None,
                        help="Source-1 entities used (default: all 2,206,821)")
    parser.add_argument("--s1-offset", type=int, default=0)
    parser.add_argument("--neg-ratio", type=int, default=2,
                        help="Negatives per recovered positive (per entity)")
    parser.add_argument("--singleton-neg-ratio", type=int, default=1)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--idf-sample", type=int, default=50_000)
    parser.add_argument("--target-limit", type=int, default=None,
                        help="Audit only: cap records taken from each target source")
    parser.add_argument("--reuse-index", action="store_true",
                        help="Reuse an existing blocking index / target store")
    parser.add_argument("--mine", action="store_true",
                        help="Phase 15: mine hard negatives with the baseline model and refit")
    parser.add_argument("--mine-s1", type=int, default=50_000,
                        help="Source-1 entities scored during hard-negative mining")
    parser.add_argument("--mine-offset", type=int, default=0)
    parser.add_argument("--mine-threshold", type=float, default=0.55)
    parser.add_argument("--s1-chunk-size", type=int, default=20_000)
    parser.add_argument("--target-chunk-size", type=int, default=100_000)
    parser.add_argument("--blocker-memory-limit", default="1500MB")
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet
    train_dir = Path(args.train_dir)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metrics = {"args": vars(args), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("=" * 80)
    print("FULL-SCALE TRAINING")
    print("=" * 80)
    print(f" train dir   : {train_dir}")
    print(f" model dir   : {args.model_dir}")
    print(f" dataset ckpt: {args.dataset_out}")
    print(f" max S1      : {args.max_s1 if args.max_s1 else 'ALL (2,206,821)'}")
    print(f" neg ratio   : {args.neg_ratio} (+{args.singleton_neg_ratio} for singletons)")

    # ---- 1. IDF -----------------------------------------------------------------
    print("\nStep 1/6: token IDF from the Source-1 training corpus (leak-free)")
    idf = compute_corpus_token_idf(str(train_dir / "train_source1.tsv"), sample_size=args.idf_sample)
    print(f"  {len(idf):,} IDF tokens computed from "
          f"{args.idf_sample:,} Source-1 training names")
    metrics["idf_tokens"] = len(idf)

    # ---- 2. ground truth --------------------------------------------------------
    print("\nStep 2/6: loading ground truth into DuckDB")
    gt_store = GroundTruthStore(args.gt_db)
    loaded = gt_store.load(train_dir / "train_ground_truth.tsv")
    print(f"  {loaded:,} ground-truth rows loaded")

    # ---- 3. target index --------------------------------------------------------
    print("\nStep 3/6: indexing Source 2/3 targets (normalise once, write to disk)")
    target_store = TargetRecordStore(args.target_db, reset=not args.reuse_index)
    blocker = DuckDBBlocker(
        args.index_db,
        max_block_size=500,
        max_candidates_per_entity=100,
        memory_limit=args.blocker_memory_limit,
        threads=4,
        temp_directory=str(CHECKPOINTS),
    )
    n_existing = blocker.load_existing()[0] if args.reuse_index else 0
    if n_existing:
        print(f"  reusing blocking index with {n_existing:,} targets")
    else:
        counts = index_targets(train_dir, idf, target_store, blocker,
                               chunk_size=args.target_chunk_size,
                               target_limit=args.target_limit, verbose=verbose)
        metrics["targets_indexed"] = counts
        print(f"  {counts['total']:,} targets indexed")

    # ---- 4. dataset -------------------------------------------------------------
    print("\nStep 4/6: building the labelled training sample (Phase 12)")
    frame, y, stats = build_training_dataset(
        train_dir, idf, blocker, target_store, gt_store,
        max_s1=args.max_s1, s1_offset=args.s1_offset,
        neg_ratio=args.neg_ratio, singleton_neg_ratio=args.singleton_neg_ratio,
        s1_chunk_size=args.s1_chunk_size, out_parquet=args.dataset_out,
        return_frame=bool(args.mine), verbose=verbose,
    )
    processed = stats.pop("processed_s1_ids")
    metrics["dataset"] = stats
    if frame is None:
        frame = pd.read_parquet(args.dataset_out)
        y = frame.pop("label").to_numpy(dtype=np.int8)
        for col in frame.columns:
            if frame[col].dtype == np.float64:
                frame[col] = frame[col].astype(np.float32)
        stats["rows"] = int(len(frame))
    feature_cols = [c for c in frame.columns if c not in ("s1_id", "cand_id")]
    metrics["feature_columns"] = feature_cols
    positive_rows = int(y.sum())
    print(f"  rows={len(frame):,} positives={positive_rows:,} "
          f"negatives={len(frame) - positive_rows:,} | blocking positive recall="
          f"{stats['positive_recall']:.4f} ({stats['positive_hits']:,}/{stats['gt_pairs']:,})")
    print(f"  feature matrix: {len(frame):,} x {len(feature_cols)} "
          f"({len(frame) * len(feature_cols) * 4 / 1e9:.2f} GB float32)")
    metrics["dataset"]["rows"] = int(len(frame))
    metrics["dataset"]["positives_kept"] = positive_rows
    metrics["dataset"]["negatives_kept"] = int(len(frame) - positive_rows)

    # ---- 5. split + fit + threshold --------------------------------------------
    print("\nStep 5/6: entity-aware split, LightGBM fit, F_0.5 threshold sweep (Phase 13/14)")
    train_ids, val_ids = entity_aware_split(processed, test_size=args.val_size, random_state=42)
    train_mask, val_mask = split_masks(frame, train_ids, val_ids)
    print(f"  train entities={len(train_ids):,} rows={int(train_mask.sum()):,} "
          f"| val entities={len(val_ids):,} rows={int(val_mask.sum()):,}")
    metrics["split"] = {
        "train_entities": int(len(train_ids)),
        "val_entities": int(len(val_ids)),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positives": int(y[train_mask].sum()),
        "val_positives": int(y[val_mask].sum()),
        "ids_shared_between_sides": len(set(train_ids.tolist()) & set(val_ids.tolist())),
    }

    model = train_lgbm_classifier(
        frame[train_mask], y[train_mask], feature_cols,
        frame[val_mask], y[val_mask],
        early_stopping_rounds=args.early_stopping_rounds, verbose=verbose,
    )
    del train_mask
    gc.collect()

    val_frame = frame[val_mask]
    gt_val = gt_store.fetch(list(val_ids))
    tune = tune_threshold_f05(model, val_frame, val_frame["s1_id"].to_numpy(),
                              gt_val, feature_cols, verbose=True)
    metrics["threshold"] = {k: v for k, v in tune.items() if k != "sweep"}
    metrics["threshold_sweep"] = tune["sweep"]
    del val_frame
    gc.collect()

    # ---- 6. optional hard-negative mining + refit -------------------------------
    if args.mine:
        print("\nStep 6/6: Phase 15 hard-negative mining and refit")
        hn_frame, hn_y, hn_stats = mine_hard_negatives(
            train_dir, idf, blocker, target_store, gt_store, model, feature_cols,
            max_s1=args.mine_s1, s1_offset=args.mine_offset,
            score_threshold=args.mine_threshold, s1_chunk_size=args.s1_chunk_size,
            verbose=verbose,
        )
        metrics["mining"] = hn_stats
        print(f"  mined rows: {len(hn_frame):,} ({hn_stats['hard_negatives_kept']:,} hard "
              f"negatives + {hn_stats['positives_kept']:,} positives from "
              f"{hn_stats['pairs_scored']:,} scored pairs)")
        if len(hn_frame):
            frame = pd.concat([frame, hn_frame], ignore_index=True)
            y = np.concatenate([y, hn_y])
            del hn_frame, hn_y
            gc.collect()
            train_mask, val_mask = split_masks(frame, train_ids, val_ids)
            model = train_lgbm_classifier(
                frame[train_mask], y[train_mask], feature_cols,
                frame[val_mask], y[val_mask],
                early_stopping_rounds=args.early_stopping_rounds, verbose=verbose,
            )
            val_frame = frame[val_mask]
            tune = tune_threshold_f05(model, val_frame, val_frame["s1_id"].to_numpy(),
                                      gt_val, feature_cols, verbose=True)
            metrics["threshold_after_mining"] = {k: v for k, v in tune.items() if k != "sweep"}
            metrics["threshold_sweep_after_mining"] = tune["sweep"]
            del val_frame, train_mask, val_mask
            gc.collect()
        frame.to_parquet(args.dataset_out, index=False)
    else:
        print("\nStep 6/6: skipping hard-negative mining (pass --mine to enable)")

    # ---- 7. persist bundle + metrics -------------------------------------------
    bundle = save_model_bundle(model, feature_cols, idf, tune["best_threshold"], args.model_dir)
    _, loaded_features, loaded_idf, loaded_threshold = load_model_bundle(str(bundle))
    metrics["bundle"] = {
        "path": str(bundle),
        "n_features": len(loaded_features),
        "feature_cols_match": loaded_features == list(feature_cols),
        "n_idf_tokens": len(loaded_idf),
        "threshold": loaded_threshold,
        "model_class": type(model).__name__,
        "inference_command": (
            f"python -m src.inference --model-dir {bundle} --output-dir output "
            "--validator student_resource/utils/validate_submission.py"
        ),
    }
    metrics["feature_importance"] = sorted(
        zip(feature_cols, [int(v) for v in model.feature_importances_]), key=lambda kv: -kv[1]
    )[:15]
    metrics["elapsed_seconds"] = round(time.time() - started, 1)
    metrics["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(args.metrics_out, metrics)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE in {time.time() - started:.0f}s")
    print(f"  bundle      : {bundle}")
    print(f"  metrics     : {args.metrics_out}")
    print(f"  features    : {len(loaded_features)} | idf tokens: {len(loaded_idf):,} "
          f"| threshold: {loaded_threshold}")
    print(f"  validation Macro F_0.5 (val entities, singletons included): "
          f"{tune['best_macro_f05']:.4f}")
    print("  Next: " + metrics["bundle"]["inference_command"])
    print("=" * 80)

    blocker.close()
    gt_store.close()
    target_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


