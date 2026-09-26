"""
End-to-end audit harness for the production test-inference engine (src/inference.py).

Steps
-----
1. Slice the real test sources (first N rows) into ``artifacts/checkpoints/mini_test/``.
2. Train a throwaway LightGBM model on real train ground-truth pairs (mini scale) and
   persist it with ``save_model_bundle(...)`` - the same bundle format notebook Phase 14
   writes for the full-scale run.
3. Execute ``run_test_inference(...)`` on the slice through the exact production code
   path (DuckDB target store, bounded Source-1 batches, streamed candidate export).
4. Run the official validator on the produced files and assert the structural
   submission rules (coverage, ordering, prefixes, no duplicates, and that every
   match is also present in the candidate list).

Usage:
    python scripts/validate_inference_mini.py [--s1 2000] [--targets 20000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_pairwise_features
from src.inference import load_model_bundle, normalize_rows, run_test_inference, save_model_bundle
from src.metrics import evaluate_macro_f_beta
from src.pipeline import compute_corpus_token_idf, stream_tsv_chunks

DATA_DIR = PROJECT_ROOT / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VALIDATOR = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"


def build_test_slice(n_s1: int, n_targets: int) -> Path:
    """Copy the first N rows of each real test source into a slice directory."""
    slice_dir = CHECKPOINT_DIR / "mini_test"
    slice_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        ("test_source1.tsv", n_s1),
        ("test_source2.tsv", n_targets),
        ("test_source3.tsv", n_targets),
    )
    for name, n_rows in specs:
        frame = next(stream_tsv_chunks(str(TEST_DIR / name), chunk_size=n_rows, n_rows=n_rows))
        frame.write_csv(str(slice_dir / name), separator="\t")
        print(f"  slice {name}: {frame.shape[0]:,} rows")
        del frame
    return slice_dir


def load_training_corpus(gt_rows: int, max_entities: int):
    """Load a small labelled corpus from the training set (positives + candidate pool)."""
    ground_truth = pl.read_csv(
        str(TRAIN_DIR / "train_ground_truth.tsv"), separator="\t", n_rows=gt_rows
    )
    gt_map = {}
    for row in ground_truth.iter_rows(named=True):
        ids = [i for i in str(row["matched_entity_ids"] or "").split(",") if i]
        if ids:
            gt_map[row["source1_entity_id"]] = ids
        if len(gt_map) >= max_entities:
            break

    s1_ids = list(gt_map.keys())
    target_ids = sorted({tid for ids in gt_map.values() for tid in ids})

    s1_frame = (
        pl.scan_csv(str(TRAIN_DIR / "train_source1.tsv"), separator="\t")
        .filter(pl.col("entity_id").is_in(s1_ids))
        .collect()
    )
    s1_records = {rec["entity_id"]: rec for rec in normalize_rows(s1_frame)}

    target_records = {}
    for source_num in (2, 3):
        frame = (
            pl.scan_csv(str(TRAIN_DIR / f"train_source{source_num}.tsv"), separator="\t")
            .filter(pl.col("entity_id").is_in(target_ids))
            .collect()
        )
        for rec in normalize_rows(frame):
            target_records[rec["entity_id"]] = rec
    return s1_records, target_records, gt_map


def build_feature_matrix(
    s1_records, target_records, gt_map, idf_dict: dict, max_neg_ratio: int = 3, seed: int = 42
):
    """Build (X, y) from true pairs plus random negatives."""
    rng = np.random.default_rng(seed)
    pool = list(target_records.keys())
    rows, labels = [], []

    def make_row(s1, cand, channel_count, min_rank):
        values = extract_pairwise_features(
            s1, cand, meta={"channel_count": channel_count, "min_rank": min_rank},
            idf_dict=idf_dict,
        )
        values["s1_id"] = s1["entity_id"]
        values["cand_id"] = cand["entity_id"]
        return values

    for s1_id, true_ids in gt_map.items():
        s1 = s1_records.get(s1_id)
        if s1 is None or not pool:
            continue
        positives = [tid for tid in true_ids if tid in target_records]
        for cand_id in positives:
            rows.append(make_row(s1, target_records[cand_id], min(3 + len(positives), 4), 0))
            labels.append(1)

        n_neg = 0
        while n_neg < max_neg_ratio * max(len(positives), 1):
            cand_id = pool[int(rng.integers(len(pool)))]
            if cand_id in true_ids:
                continue
            rows.append(make_row(s1, target_records[cand_id], 1, 5))
            labels.append(0)
            n_neg += 1

    return pd.DataFrame(rows), np.asarray(labels)


def train_and_tune(X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    """Train a LightGBM model and tune the Macro F_0.5 decision threshold."""
    import lightgbm as lgb

    feature_cols = [col for col in X.columns if col not in ("s1_id", "cand_id")]
    split = int(len(X) * 0.75)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X.iloc[:split][feature_cols], y[:split])

    s1_array = X["s1_id"].to_numpy()[split:]
    cand_array = X["cand_id"].to_numpy()[split:]
    y_val = y[split:]
    probabilities = model.predict_proba(X.iloc[split:][feature_cols])[:, 1]

    truth = {}
    for i in range(len(y_val)):
        if y_val[i] == 1:
            truth.setdefault(s1_array[i], set()).add(cand_array[i])

    best_threshold, best_f05 = 0.5, -1.0
    for threshold in np.arange(0.30, 0.96, 0.05):
        predictions = {}
        for i in np.where(probabilities >= threshold)[0]:
            predictions.setdefault(s1_array[i], set()).add(cand_array[i])
        f05 = evaluate_macro_f_beta(truth, predictions)["macro_f_beta"]
        if f05 > best_f05:
            best_threshold, best_f05 = round(float(threshold), 2), f05
    return model, feature_cols, best_threshold, best_f05


def check_output_files(matching_path: Path, candidate_path: Path, slice_dir: Path) -> None:
    """Assert every structural rule from the challenge submission specification."""
    rows = matching_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "source1_entity_id\tmatched_entity_ids", f"bad matching header: {rows[0]!r}"

    required = pl.read_csv(str(slice_dir / "test_source1.tsv"), separator="\t")["entity_id"].to_list()
    seen = [line.split("\t")[0] for line in rows[1:]]
    assert seen == required, "rows must cover test_source1.tsv exactly (same order, no extras)"
    assert len(set(seen)) == len(seen), "duplicate source1_entity_id rows"

    candidate_rows = candidate_path.read_text(encoding="utf-8").splitlines()
    assert candidate_rows[0] == "source1_entity_id\tcandidate_entity_ids", \
        f"bad candidate header: {candidate_rows[0]!r}"
    candidates = {}
    for line in candidate_rows[1:]:
        s1_id, _, ids = line.partition("\t")
        candidates[s1_id] = set(ids.split(",")) if ids else set()

    for line in rows[1:]:
        s1_id, _, ids = line.partition("\t")
        matched = ids.split(",") if ids else []
        assert len(matched) == len(set(matched)), f"repeated IDs in list for {s1_id}"
        assert all(mid.startswith(("S2-", "S3-")) for mid in matched), f"bad ID prefix for {s1_id}"
        assert set(matched) <= candidates.get(s1_id, set()), \
            f"matches are not a subset of candidates for {s1_id}"

    n_candidates = sum(len(ids) for ids in candidates.values())
    print(f"  structural checks passed: {len(seen):,} S1 rows, {n_candidates:,} candidate pairs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=int, default=2000, help="Source-1 rows in the slice")
    parser.add_argument("--targets", type=int, default=20000, help="S2/S3 rows in the slice")
    parser.add_argument("--gt-rows", type=int, default=8000, help="Ground-truth rows scanned")
    parser.add_argument("--entities", type=int, default=400, help="Labelled S1 entities used")
    parser.add_argument("--idf-sample", type=int, default=20000, help="IDF corpus sample size")
    args = parser.parse_args()

    started = time.time()
    print("=" * 80)
    print("MINI END-TO-END INFERENCE AUDIT (production engine on a real test-data slice)")
    print("=" * 80)

    print("\n[1/6] Building the test slice...")
    slice_dir = build_test_slice(args.s1, args.targets)

    print("\n[2/6] Computing corpus IDF weights (Source-1 training corpus, leak-free)...")
    idf_dict = compute_corpus_token_idf(
        str(TRAIN_DIR / "train_source1.tsv"), sample_size=args.idf_sample
    )
    print(f"  IDF tokens: {len(idf_dict):,}")

    print("\n[3/6] Loading the labelled training corpus...")
    s1_records, target_records, gt_map = load_training_corpus(args.gt_rows, args.entities)
    print(f"  S1 entities: {len(s1_records):,} | target records: {len(target_records):,}")

    print("\n[4/6] Extracting features and training the audit model...")
    features, labels = build_feature_matrix(s1_records, target_records, gt_map, idf_dict)
    print(f"  feature matrix: {features.shape} | positives: {int(labels.sum()):,}")
    model, feature_cols, threshold, f05 = train_and_tune(features, labels)
    print(f"  tuned threshold: {threshold} | holdout Macro F_0.5: {f05:.4f}")

    bundle_dir = save_model_bundle(
        model, feature_cols, idf_dict, threshold,
        PROJECT_ROOT / "artifacts" / "models" / "mini_audit",
    )
    reloaded_model, reloaded_cols, reloaded_idf, reloaded_threshold = load_model_bundle(bundle_dir)
    assert reloaded_cols == feature_cols, "feature order changed across the bundle round-trip"
    assert reloaded_threshold == threshold, "threshold changed across the bundle round-trip"
    print(f"  model bundle round-trip OK: {bundle_dir}")

    print("\n[5/6] Running the production inference engine on the slice...")
    output_dir = CHECKPOINT_DIR / "mini_output"
    summary = run_test_inference(
        test_dir=slice_dir,
        output_dir=output_dir,
        model=reloaded_model,
        feature_cols=reloaded_cols,
        threshold=reloaded_threshold,
        idf_dict=reloaded_idf,
        chunk_size=5000,
        s1_chunk_size=500,
        max_cands_per_s1=50,
        stream_output=True,
        keep_target_store=False,
        validator_script=str(VALIDATOR) if VALIDATOR.exists() else None,
        check_ids=True,
    )

    print("\n[6/6] Structural submission checks...")
    check_output_files(
        output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv", slice_dir
    )

    passed = summary.get("validator_exit_code", 0) == 0
    print("\n" + "=" * 80)
    print(f"MINI AUDIT RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"  summary: {summary}")
    print(f"  total wall time: {time.time() - started:.1f}s")
    print("=" * 80)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
