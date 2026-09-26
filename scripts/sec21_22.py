import nbformat as nbf

def get_p21_p22_cells():
    c = []
    # Phase 21
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 21 — Submission Export & Integrity Verification

The **real** submission files are produced by Phase 20 (`run_full_test_inference()` or
`python -m src.inference`) and must land in `output/`:

1. `output/matching_results.tsv` (scored leaderboard file)
2. `output/candidate_pairs.tsv` (blocking audit file)

This phase demonstrates the exporter on a 5-entity slice written to `artifacts/demo/`
(never to `output/`, so a demo can never overwrite a real submission) and then runs the
official `validate_submission.py` — against the real files when they exist, otherwise
against the demo slice.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Submission export demonstration (writes to artifacts/demo/, never to output/)
DEMO_DIR = ARTIFACT_DIR / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
demo_matching_file = DEMO_DIR / "demo_matching_results.tsv"
demo_candidate_file = DEMO_DIR / "demo_candidate_pairs.tsv"
demo_slice_dir = DEMO_DIR / "mini_test"
demo_slice_dir.mkdir(parents=True, exist_ok=True)

# 5 real test S1 entities + real S2/S3 IDs, so the demo itself is validator-clean
demo_s1_rows = pl.read_csv(str(TEST_DIR / "test_source1.tsv"), separator="\\t", n_rows=5)
demo_s1_rows.write_csv(str(demo_slice_dir / "test_source1.tsv"), separator="\\t")
sample_test_s1 = demo_s1_rows['entity_id'].to_list()
demo_s2_ids = pl.scan_csv(str(TEST_DIR / "test_source2.tsv"), separator="\\t").select("entity_id").head(2).collect()['entity_id'].to_list()
demo_s3_ids = pl.scan_csv(str(TEST_DIR / "test_source3.tsv"), separator="\\t").select("entity_id").head(1).collect()['entity_id'].to_list()

mock_preds = {sample_test_s1[0]: {demo_s2_ids[0], demo_s3_ids[0]}, sample_test_s1[1]: set()}
mock_cands = {sample_test_s1[0]: {demo_s2_ids[0], demo_s2_ids[1], demo_s3_ids[0]}, sample_test_s1[1]: set()}
for sid in sample_test_s1[2:]:
    mock_preds[sid] = set()
    mock_cands[sid] = set()

export_submission(
    predictions=mock_preds,
    test_s1_ids=sample_test_s1,
    output_matching_path=str(demo_matching_file),
    candidates=mock_cands,
    output_candidate_path=str(demo_candidate_file)
)

print("Exported demo files to:")
print(f" - {demo_matching_file}")
print(f" - {demo_candidate_file}")

with open(demo_matching_file, 'r', encoding='utf-8') as f:
    print(f"\\nMatching Header: {repr(f.readline())}")
    print(f"Sample line 1:   {repr(f.readline())}")

with open(demo_candidate_file, 'r', encoding='utf-8') as f:
    print(f"Candidate Header: {repr(f.readline())}")
    print(f"Sample line 1:    {repr(f.readline())}")

real_matching = OUTPUT_DIR / "matching_results.tsv"
real_candidate = OUTPUT_DIR / "candidate_pairs.tsv"
if real_matching.exists():
    print(f"\\nReal submission file found: {real_matching} ({real_matching.stat().st_size/1024:.1f} KB)")
else:
    print(f"\\nNo real submission yet at {real_matching}")
    print("Run Phase 20 (run_full_test_inference()) or `python -m src.inference` before submitting.")
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Validate the submission format with the official validator
import subprocess

validator_script = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"
real_matching = OUTPUT_DIR / "matching_results.tsv"
real_candidate = OUTPUT_DIR / "candidate_pairs.tsv"

if real_matching.exists():
    matching_target, candidate_target, test_dir_target = real_matching, real_candidate, TEST_DIR
    print("Validating the REAL submission files produced by Phase 20.")
else:
    matching_target, candidate_target, test_dir_target = demo_matching_file, demo_candidate_file, demo_slice_dir
    print("No real submission yet — validating the 5-entity demo slice for format compliance.")
    print("Produce the real files with Phase 20 (or `python -m src.inference`), then re-run this cell.")

print(f"Official validator: {validator_script} (exists: {validator_script.exists()})")
if validator_script.exists():
    cmd = [sys.executable, str(validator_script), "--matching", str(matching_target),
           "--candidate", str(candidate_target), "--test-dir", str(test_dir_target)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    print(f"Validator exit code: {result.returncode} (0 = PASS, safe to submit)")
    print("Add --check-ids for the memory-heavy match-ID existence check on the real files.")
else:
    print("Run manually:")
    print(f"  python {validator_script} --matching {matching_target} --candidate {candidate_target} --test-dir {test_dir_target}")
"""
    ))

    # Phase 22
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 22 — Ablation Benchmark & Experiment Tracker
Consolidates the iterative experimental configurations explored throughout this notebook.

> ⚠️ **The figures in the table below are hand-entered illustrative planning targets, not measured
> validation scores.** Re-run the corresponding phases on the entity-aware validation split
> (Phase 13) and replace them with measured values before citing any number in the methodology
> document or the final submission package.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""ablation_records = [
    {
        "Experiment": "Baseline (Exact Name Matching Only)",
        "Candidate Recall (%)": "34.2%",
        "Validation Macro F_0.5": "0.412",
        "Top-1 Precision": "0.965",
        "Avg Cands/S1": "1.1",
        "Runtime": "Fast (~2m)"
    },
    {
        "Experiment": "+ Unicode & Legal Suffix Normalization",
        "Candidate Recall (%)": "68.5%",
        "Validation Macro F_0.5": "0.684",
        "Top-1 Precision": "0.932",
        "Avg Cands/S1": "4.2",
        "Runtime": "Fast (~4m)"
    },
    {
        "Experiment": "+ Multi-Channel Adaptive Blocking",
        "Candidate Recall (%)": "92.8%",
        "Validation Macro F_0.5": "0.781",
        "Top-1 Precision": "0.895",
        "Avg Cands/S1": "28.4",
        "Runtime": "Medium (~15m)"
    },
    {
        "Experiment": "+ Rare-Token IDF Weighting & Pairwise Features",
        "Candidate Recall (%)": "92.8%",
        "Validation Macro F_0.5": "0.842",
        "Top-1 Precision": "0.924",
        "Avg Cands/S1": "28.4",
        "Runtime": "Medium (~25m)"
    },
    {
        "Experiment": "+ LightGBM Classifier & F_0.5 Threshold Tuning",
        "Candidate Recall (%)": "92.8%",
        "Validation Macro F_0.5": "0.887",
        "Top-1 Precision": "0.941",
        "Avg Cands/S1": "28.4",
        "Runtime": "Model Train (~10m)"
    },
    {
        "Experiment": "+ Iterative Hard Negative Mining",
        "Candidate Recall (%)": "92.8%",
        "Validation Macro F_0.5": "0.908",
        "Top-1 Precision": "0.958",
        "Avg Cands/S1": "28.4",
        "Runtime": "Model Retrain (~15m)"
    }
]

df_ablation = pd.DataFrame(ablation_records)
df_ablation = df_ablation.rename(columns={"Validation Macro F_0.5": "Illustrative Macro F_0.5"})
print("=" * 80)
print("CHALLENGE EXPERIMENT TRACKER & ABLATION STUDY")
print("=" * 80)
print("NOTE: these figures are illustrative planning targets, NOT measured validation scores.")
display(df_ablation)
"""
    ))
    return c
