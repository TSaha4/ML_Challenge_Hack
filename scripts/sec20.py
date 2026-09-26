import nbformat as nbf


def get_p20_cells():
    c = []

    c.append(nbf.v4.new_markdown_cell(
"""# Phase 20 — End-to-End Test Inference Pipeline
### ⚠️ DO NOT RUN AUTOMATICALLY — RUN MANUALLY FOR FINAL SUBMISSION

Generates the two scored artifacts over the **full** test set (~1.7M Source-1 entities
against ~10M Source-2/3 targets) using the memory-safe engine in `src/inference.py`:

1. Normalize Test Source 2 & 3 and index them into a **disk-backed DuckDB** store
   (holding ~10M normalized records in a Python dict of dicts would need >10 GB RAM).
2. Fit the multi-channel `AdaptiveBlocker` inverted index incrementally, chunk by chunk.
3. Stream Test Source 1 in bounded batches, generate blocking candidates, compute the
   full pairwise feature matrix and score it with the trained LightGBM model.
4. Apply the $F_{0.5}$-optimal decision threshold tuned in Phase 14.
5. Stream `output/matching_results.tsv` and `output/candidate_pairs.tsv` row by row
   (so ~50M candidate IDs are never materialised in RAM), then run the official
   `validate_submission.py` against the produced files.

Equivalent command line (recommended for the final run):

```bash
python -m src.inference \\
    --model-dir artifacts/models/lgb_f05 \\
    --output-dir output \\
    --validator student_resource/utils/validate_submission.py
```
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Phase 20 — End-to-End Test Inference (manual run, memory-safe streaming engine)
# Decision threshold tuned for Macro F_0.5 in Phase 14 (falls back to 0.65).
INFERENCE_THRESHOLD = float(best_thresh) if 'best_thresh' in globals() else 0.65
MODEL_BUNDLE_DIR = MODEL_DIR / "lgb_f05"


def run_full_test_inference(
    test_dir: Path = TEST_DIR,
    output_dir: Path = OUTPUT_DIR,
    model: Any = model_lgb,
    feature_cols: List[str] = feature_cols,
    idf_weights: Optional[Dict[str, float]] = idf_dict,
    threshold: float = INFERENCE_THRESHOLD,
    chunk_size: int = CHUNK_SIZE,
    s1_chunk_size: int = 20_000,
    max_cands_per_s1: int = MAX_CANDIDATES
) -> Dict[str, Any]:
    '''Run the complete blocking -> scoring -> export pipeline over the test set.

    Normalized target records live on disk (DuckDB) and both output files are written
    incrementally, so peak RAM stays bounded on the ~10M-record test set.
    '''
    validator = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"
    return run_test_inference(
        test_dir=test_dir, output_dir=output_dir, model=model, feature_cols=feature_cols,
        threshold=threshold, idf_dict=idf_weights, chunk_size=chunk_size,
        s1_chunk_size=s1_chunk_size, max_cands_per_s1=max_cands_per_s1,
        validator_script=str(validator) if validator.exists() else None
    )


print("=" * 80)
print("TEST INFERENCE PIPELINE READY — NOT EXECUTED AUTOMATICALLY")
print("=" * 80)
print(f" model bundle      : {MODEL_BUNDLE_DIR} (exists: {MODEL_BUNDLE_DIR.exists()})")
print(f" feature columns   : {len(feature_cols)}")
print(f" decision threshold: {INFERENCE_THRESHOLD}")
print(f" test dir          : {TEST_DIR}")
print(f" output dir        : {OUTPUT_DIR}")
print()
print("Run the full test inference manually with:")
print("  summary = run_full_test_inference()")
print()
print("or from the command line (recommended for the final submission):")
print(f"  python -m src.inference --model-dir {MODEL_BUNDLE_DIR} --output-dir {OUTPUT_DIR}")
print()
print("Validate the engine end-to-end on a small slice first:")
print("  python scripts/validate_inference_mini.py")
"""
    ))

    return c
