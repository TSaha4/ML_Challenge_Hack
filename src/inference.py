"""
Memory-Efficient Test-Time Inference Engine for Entity Resolution.

This module owns the heavy end-to-end *inference* path so the final submission
artifacts can be regenerated from the command line as well as from the notebook:

    python -m src.inference \\
        --test-dir student_resource/dataset/test \\
        --model-dir artifacts/models/lgb_f05 \\
        --output-dir output

Design goals
------------
* Stream both the target (Source 2/3) and the reference (Source 1) sources with
  ``stream_tsv_chunks`` (batched Polars reader), so no full multi-GB file is ever
  materialised in RAM.
* Keep the normalized target records on disk in DuckDB instead of a Python dict
  of dicts (~10 GB of RAM at the ~10M-record test scale); only the candidates
  needed by the current Source-1 batch are pulled back into memory.
* Write ``matching_results.tsv`` / ``candidate_pairs.tsv`` incrementally, so the
  export never holds ~50M candidate ID strings in memory at once.

Heavy third-party imports (``duckdb``, ``joblib``) are performed lazily inside the
functions so importing this module stays cheap.
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.blocking import AdaptiveBlocker, DuckDBBlocker
from src.features import extract_pairwise_features
from src.normalization import normalize_address, normalize_text
from src.pipeline import export_submission, stream_tsv_chunks

#: Column layout of the on-disk normalized target store (order matters for INSERT).
TARGET_COLUMNS: List[str] = [
    "entity_id",
    "country",
    "name_norm",
    "name_core",
    "name_translit",
    "name_initials",
    "addr_postal",
    "addr_norm",
]

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"

#: Entity-ID prefixes that may legitimately appear in a match list (Source 2/3 only).
MATCH_ID_PREFIXES: Tuple[str, ...] = ("S2-", "S3-")


def normalize_rows(chunk) -> List[Dict[str, Any]]:
    """
    Normalize one Polars chunk of Source-1/2/3 rows into index-ready dictionaries.

    Used for both the target sources (S2/S3, which are indexed) and the reference
    source (S1, which is queried) because both share the same raw schema.
    """
    records: List[Dict[str, Any]] = []
    for row in chunk.iter_rows(named=True):
        name_rep = normalize_text(row["business_name"])
        addr_rep = normalize_address(row["business_address"])
        records.append({
            "entity_id": row["entity_id"],
            "country": row["country"],
            "name_norm": name_rep["unicode_normalized"],
            "name_core": name_rep["core_name"],
            "name_translit": name_rep["transliterated"],
            "name_initials": name_rep["initials"],
            "addr_postal": addr_rep["postal_code"],
            "addr_norm": addr_rep["normalized"],
        })
    return records


def save_model_bundle(
    model: Any,
    feature_cols: List[str],
    idf_dict: Optional[Dict[str, float]],
    threshold: float,
    bundle_dir: str,
) -> Path:
    """
    Persist everything the inference engine needs into a single directory:
    the fitted model (joblib), the ordered feature list, the corpus IDF weights,
    and the F_0.5-optimal decision threshold.
    """
    import joblib

    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, bundle / "model.joblib")

    metadata = {
        "feature_cols": list(feature_cols),
        "threshold": float(threshold),
        "idf_dict": {str(k): float(v) for k, v in (idf_dict or {}).items()},
        "n_idf_tokens": len(idf_dict or {}),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(bundle / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return bundle


def load_model_bundle(bundle_dir: str) -> Tuple[Any, List[str], Dict[str, float], float]:
    """Load a bundle written by :func:`save_model_bundle`."""
    import joblib

    bundle = Path(bundle_dir)
    model_path = bundle / "model.joblib"
    meta_path = bundle / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No model bundle found in '{bundle}'. Expected 'model.joblib' and "
            f"'metadata.json' (created by src.inference.save_model_bundle)."
        )

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return (
        model,
        list(metadata["feature_cols"]),
        dict(metadata.get("idf_dict", {})),
        float(metadata.get("threshold", 0.5)),
    )


def run_validator(
    validator_script: str,
    matching_path: str,
    candidate_path: str,
    test_dir: str,
    check_ids: bool = True,
) -> int:
    """Run the official challenge validator and echo its report; return exit code."""
    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
    if check_ids:
        cmd.append("--check-ids")

    print("Running official submission validator...")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    report = (result.stdout or "").strip()
    if report:
        print(report)
    if result.stderr:
        print(result.stderr.strip())
    return int(result.returncode)


def _fetch_targets(con, needed_ids: Set[str], columns_sql: str) -> Dict[str, Dict[str, Any]]:
    """Pull only the target rows needed by the current S1 batch out of DuckDB."""
    if not needed_ids:
        return {}
    con.register("needed_ids", pd.DataFrame({"id": list(needed_ids)}))
    try:
        frame = con.execute(
            f"SELECT {columns_sql} FROM target_records t "
            "JOIN needed_ids n ON t.entity_id = n.id"
        ).fetchdf()
    finally:
        con.unregister("needed_ids")
    lookup = frame.set_index("entity_id").to_dict("index")
    del frame
    return lookup


def run_test_inference(
    test_dir,
    output_dir,
    model: Any,
    feature_cols: List[str],
    threshold: float = 0.65,
    idf_dict: Optional[Dict[str, float]] = None,
    chunk_size: int = 100_000,
    s1_chunk_size: int = 20_000,
    max_block_size: int = 500,
    max_cands_per_s1: int = 100,
    blocker_backend: str = "duckdb",
    blocker_memory_limit: str = "3GB",
    blocker_threads: int = 4,
    target_store_path: Optional[str] = None,
    stream_output: bool = True,
    keep_target_store: bool = False,
    validator_script: Optional[str] = None,
    check_ids: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full, memory-safe test inference: block -> score -> export.

    Parameters
    ----------
    test_dir : directory holding ``test_source1.tsv``, ``test_source2.tsv``, ``test_source3.tsv``.
    output_dir : directory receiving ``matching_results.tsv`` and ``candidate_pairs.tsv``.
    model : fitted classifier exposing ``predict_proba`` (LightGBM/XGBoost/CatBoost sklearn API).
    feature_cols : ordered feature names the model was trained on.
    threshold : probability cut-off tuned for Macro F_0.5.
    idf_dict : corpus token IDF weights used by blocking and feature extraction.
    chunk_size : batch size while indexing Source 2/3.
    s1_chunk_size : batch size while streaming Source 1 (bounds peak RAM).
    blocker_backend : ``"duckdb"`` (default) keeps the inverted index on disk, which
        is required at competition scale (~800 B per indexed target in memory means
        ~8 GB for the ~10M test targets); ``"memory"`` uses the original
        in-process :class:`AdaptiveBlocker` and is only suitable for small audits.
    stream_output : write both TSVs row-by-row instead of accumulating in memory
        (recommended for the full test set; ``False`` is convenient for small audits).
    keep_target_store : keep the DuckDB target store on disk after the run.
    validator_script, check_ids : optionally run the official validator at the end.

    Returns a summary dictionary with counts, timings and (optionally) the validator
    exit code and report.
    """
    import duckdb

    started = time.time()
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    store_path = Path(target_store_path) if target_store_path else output_dir / "_test_targets.duckdb"
    if store_path.exists():
        os.remove(store_path)

    columns_sql = ", ".join(TARGET_COLUMNS)
    block_index_path = store_path.with_name(store_path.stem + "_blocking.duckdb")
    if blocker_backend == "duckdb":
        if block_index_path.exists():
            os.remove(block_index_path)
        blocker = DuckDBBlocker(
            block_index_path,
            max_block_size=max_block_size,
            max_candidates_per_entity=max_cands_per_s1,
            memory_limit=blocker_memory_limit,
            threads=blocker_threads,
            temp_directory=str(output_dir),
        )
    elif blocker_backend == "memory":
        blocker = AdaptiveBlocker(
            max_block_size=max_block_size,
            max_candidates_per_entity=max_cands_per_s1,
        )
    else:
        raise ValueError(f"unknown blocker_backend '{blocker_backend}' (use 'duckdb' or 'memory')")

    if verbose:
        print("=" * 80)
        print("TEST INFERENCE PIPELINE (memory-safe chunked mode)")
        print("=" * 80)
        print(f" test dir        : {test_dir}")
        print(f" output dir      : {output_dir}")
        print(f" threshold       : {threshold}")
        print(f" target store    : {store_path}")
        print(f" blocker backend : {blocker_backend}")
        print(f" features        : {len(feature_cols)}")

    con = duckdb.connect(str(store_path))
    # Bound the buffer pool: the default (80% of RAM) let RSS balloon during
    # the ~10M-row bulk insert on a 14 GB machine.
    con.execute("PRAGMA memory_limit='1024MB'")
    con.execute("PRAGMA threads=4")
    con.execute(
        """CREATE TABLE target_records (
            entity_id VARCHAR PRIMARY KEY,
            country VARCHAR,
            name_norm VARCHAR,
            name_core VARCHAR,
            name_translit VARCHAR,
            name_initials VARCHAR,
            addr_postal VARCHAR,
            addr_norm VARCHAR
        )"""
    )

    n_targets = 0
    try:
        if verbose:
            print("\nStep 1: indexing test Source 2 & 3 into the DuckDB target store...")
        for source_num in (2, 3):
            path = test_dir / f"test_source{source_num}.tsv"
            source_count = 0
            for chunk in stream_tsv_chunks(str(path), chunk_size=chunk_size):
                records = normalize_rows(chunk)
                if not records:
                    continue
                blocker.fit(records, idf_dict=idf_dict)
                frame = pd.DataFrame(records, columns=TARGET_COLUMNS)
                con.register("chunk_frame", frame)
                try:
                    con.execute(
                        f"INSERT OR REPLACE INTO target_records ({columns_sql}) "
                        f"SELECT {columns_sql} FROM chunk_frame"
                    )
                finally:
                    con.unregister("chunk_frame")
                source_count += len(records)
                n_targets += len(records)
                del records, frame
                gc.collect()
            if verbose:
                print(f"  - {path.name}: {source_count:,} records indexed")
        if hasattr(blocker, "finalize"):
            if verbose:
                print("  building the blocking-key lookup index (one-off cost)...")
            blocker.finalize()
        if verbose:
            print(f"  total target records indexed: {n_targets:,}")

        # ------------------------------------------------------------------
        # Step 2 - stream Source 1, block, score and export incrementally.
        # ------------------------------------------------------------------
        if verbose:
            print("\nStep 2: streaming Source 1, scoring candidates and exporting...")

        match_handle = open(matching_path, "w", encoding="utf-8", newline="") if stream_output else None
        cand_handle = open(candidate_path, "w", encoding="utf-8", newline="") if stream_output else None
        if stream_output:
            match_handle.write(MATCHING_HEADER)
            cand_handle.write(CANDIDATE_HEADER)

        predictions: Dict[str, Set[str]] = {}
        candidates_export: Dict[str, Set[str]] = {}
        s1_ids: List[str] = []

        n_s1 = n_with_matches = n_matches = n_candidates = 0
        s1_path = test_dir / "test_source1.tsv"

        try:
            for chunk in stream_tsv_chunks(str(s1_path), chunk_size=s1_chunk_size):
                s1_records = normalize_rows(chunk)
                if not s1_records:
                    continue

                blocking_hits: Dict[str, List[Dict[str, Any]]] = blocker.query_batch(
                    s1_records, idf_dict=idf_dict
                )
                needed_ids: Set[str] = set()
                for hits in blocking_hits.values():
                    needed_ids.update(hit["candidate_id"] for hit in hits)

                target_lookup = _fetch_targets(con, needed_ids, columns_sql)

                feature_rows: List[Dict[str, float]] = []
                scored_pairs: List[Tuple[str, str]] = []
                for s1_rec in s1_records:
                    s1_id = s1_rec["entity_id"]
                    for hit in blocking_hits.get(s1_id, ()):
                        cand_rec = target_lookup.get(hit["candidate_id"])
                        if cand_rec is None:
                            continue
                        meta = {
                            "channel_count": hit["channel_count"],
                            "min_rank": hit["min_rank"],
                        }
                        feature_rows.append(
                            extract_pairwise_features(s1_rec, cand_rec, meta=meta, idf_dict=idf_dict)
                        )
                        scored_pairs.append((s1_id, hit["candidate_id"]))

                probabilities = []
                if feature_rows:
                    feature_frame = pd.DataFrame(feature_rows)
                    missing = [col for col in feature_cols if col not in feature_frame.columns]
                    if missing:
                        raise ValueError(
                            "The model expects features the extractor did not produce: "
                            f"{missing}"
                        )
                    probabilities = model.predict_proba(feature_frame[feature_cols])[:, 1]
                    del feature_frame

                preds: Dict[str, Set[str]] = {rec["entity_id"]: set() for rec in s1_records}
                cands: Dict[str, Set[str]] = {rec["entity_id"]: set() for rec in s1_records}
                for (s1_id, cand_id), prob in zip(scored_pairs, probabilities):
                    cands[s1_id].add(cand_id)
                    if prob >= threshold:
                        preds[s1_id].add(cand_id)

                for s1_rec in s1_records:
                    s1_id = s1_rec["entity_id"]
                    matched_ids = sorted(preds[s1_id])
                    candidate_ids = sorted(cands[s1_id])
                    if stream_output:
                        match_handle.write(f"{s1_id}\t{','.join(matched_ids)}\n")
                        cand_handle.write(f"{s1_id}\t{','.join(candidate_ids)}\n")
                    else:
                        s1_ids.append(s1_id)
                        predictions[s1_id] = set(matched_ids)
                        candidates_export[s1_id] = set(candidate_ids)

                    n_s1 += 1
                    n_matches += len(matched_ids)
                    n_candidates += len(candidate_ids)
                    if matched_ids:
                        n_with_matches += 1

                del s1_records, blocking_hits, needed_ids, target_lookup
                del feature_rows, scored_pairs, preds, cands
                gc.collect()

                if verbose and n_s1 % (s1_chunk_size * 5) < s1_chunk_size:
                    print(
                        f"  processed {n_s1:,} S1 entities | {n_matches:,} matches | "
                        f"{time.time() - started:.0f}s"
                    )
        finally:
            if stream_output:
                match_handle.close()
                cand_handle.close()

        if not stream_output:
            # Reuse the shared exporter (identical code path to the notebook demo).
            export_submission(
                predictions=predictions,
                test_s1_ids=s1_ids,
                output_matching_path=str(matching_path),
                candidates=candidates_export,
                output_candidate_path=str(candidate_path),
            )
    finally:
        con.close()
        if blocker_backend == "duckdb":
            blocker.close()
        if not keep_target_store:
            for cleanup_path in (store_path, block_index_path):
                if cleanup_path.exists():
                    os.remove(cleanup_path)
        elif blocker_backend == "duckdb" and block_index_path.exists():
            print(f"[info] blocking index kept at {block_index_path}")

    elapsed = time.time() - started
    summary: Dict[str, Any] = {
        "target_records": n_targets,
        "s1_entities": n_s1,
        "s1_with_matches": n_with_matches,
        "predicted_matches": n_matches,
        "candidate_pairs": n_candidates,
        "threshold": float(threshold),
        "elapsed_seconds": round(elapsed, 1),
        "matching_path": str(matching_path),
        "candidate_path": str(candidate_path),
    }

    if verbose:
        print("\n" + "=" * 80)
        print(f"Inference complete in {elapsed / 60:.1f} min")
        print(f"  S1 entities scored : {n_s1:,}")
        print(f"  S1 with >=1 match  : {n_with_matches:,}")
        print(f"  predicted matches  : {n_matches:,}")
        print(f"  candidate pairs    : {n_candidates:,}")
        print(f"  matching_results   : {matching_path}")
        print(f"  candidate_pairs    : {candidate_path}")
        print("=" * 80)

    if validator_script:
        summary["validator_exit_code"] = run_validator(
            validator_script, matching_path, candidate_path, test_dir, check_ids=check_ids
        )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point: regenerate both submission TSVs from a model bundle."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate output/matching_results.tsv and output/candidate_pairs.tsv "
                    "for the ML Challenge 2026 business entity resolution test set."
    )
    parser.add_argument("--test-dir", default="student_resource/dataset/test",
                        help="Folder with test_source1/2/3.tsv")
    parser.add_argument("--output-dir", default="output",
                        help="Folder receiving matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--model-dir", required=True,
                        help="Bundle directory written by save_model_bundle(...)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the F_0.5 threshold stored in the bundle")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="Batch size while indexing Source 2/3")
    parser.add_argument("--s1-chunk-size", type=int, default=20_000,
                        help="Batch size while streaming Source 1 (bounds peak RAM)")
    parser.add_argument("--max-cands", type=int, default=100,
                        help="Maximum blocking candidates per Source 1 entity")
    parser.add_argument("--blocker-backend", choices=["duckdb", "memory"], default="duckdb",
                        help="Where the inverted index lives: 'duckdb' (default) keeps it on "
                             "disk and is required at competition scale; 'memory' is for small audits")
    parser.add_argument("--blocker-memory-limit", default="3GB",
                        help="DuckDB RAM budget for the disk-backed blocking index")
    parser.add_argument("--blocker-threads", type=int, default=4,
                        help="DuckDB threads for the disk-backed blocking index")
    parser.add_argument("--validator", default="student_resource/utils/validate_submission.py")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip running the official validator at the end")
    parser.add_argument("--no-check-ids", action="store_true",
                        help="Run the validator without its (memory-hungry) ID existence check")
    args = parser.parse_args(argv)

    model, feature_cols, idf_dict, threshold = load_model_bundle(args.model_dir)
    if args.threshold is not None:
        threshold = args.threshold
    print(f"Loaded bundle '{args.model_dir}': {len(feature_cols)} features, "
          f"threshold={threshold}, {len(idf_dict):,} IDF tokens")

    validator = None if args.no_validate else args.validator
    if validator and not Path(validator).exists():
        print(f"[warn] validator not found at '{validator}' — skipping format validation.")
        validator = None

    summary = run_test_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model=model,
        feature_cols=feature_cols,
        threshold=threshold,
        idf_dict=idf_dict,
        chunk_size=args.chunk_size,
        s1_chunk_size=args.s1_chunk_size,
        max_cands_per_s1=args.max_cands,
        blocker_backend=args.blocker_backend,
        blocker_memory_limit=args.blocker_memory_limit,
        blocker_threads=args.blocker_threads,
        validator_script=validator,
        check_ids=not args.no_check_ids,
    )
    exit_code = int(summary.get("validator_exit_code", 0))
    if exit_code != 0:
        print("\nVALIDATION FAILED — fix the reported issues before submitting.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
