import nbformat as nbf

def get_p7_cells():
    c = []
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 7 — Blocking Recall Benchmark
**Critical Validation Step**: Evaluates what percentage of true ground-truth matches are retrieved by candidate blocking before training any machine learning model.
Calculates:
- **Candidate Recall (%)**: $\\frac{|\\text{True Matches Retrieved}|}{|\\text{Total True Ground Truth Matches}|}$
- **Average Candidates / Entity**
- **Channel Synergy & Coverage**
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# FAST Benchmark: Evaluate blocking recall on a controlled validation subset (500 reference records)
print("Running Blocking Recall Benchmark on validation subset (500 reference records)...")

# 1. Load sample S1 records
s1_bench = pl.read_csv(str(TRAIN_DIR / "train_source1.tsv"), separator="\\t", n_rows=500).to_pandas()
bench_s1_ids = s1_bench['entity_id'].tolist()

# 2. Extract ground truth for these entities via indexed scan
gt_scan = pl.scan_csv(str(TRAIN_DIR / "train_ground_truth.tsv"), separator="\\t")
bench_gt_df = gt_scan.filter(pl.col('source1_entity_id').is_in(bench_s1_ids)).collect()

needed_targets = []
bench_gt = {}
for r in bench_gt_df.iter_rows(named=True):
    mids = [x.strip() for x in (r['matched_entity_ids'] or '').split(',') if x.strip()]
    bench_gt[r['source1_entity_id']] = mids
    needed_targets.extend(mids)

all_true_matches = len(needed_targets)
print(f"Benchmark Entities: {len(bench_s1_ids):,}")
print(f"True Matches to Retrieve: {all_true_matches:,}")

# 3. Retrieve target records (true targets + background distractors)
needed_s2 = [x for x in needed_targets if x.startswith('S2-')]
needed_s3 = [x for x in needed_targets if x.startswith('S3-')]

s2_hits = pl.scan_csv(str(TRAIN_DIR / "train_source2.tsv"), separator="\\t").filter(
    pl.col('entity_id').is_in(needed_s2)
).collect().to_pandas()

s3_hits = pl.scan_csv(str(TRAIN_DIR / "train_source3.tsv"), separator="\\t").filter(
    pl.col('entity_id').is_in(needed_s3)
).collect().to_pandas()

# Add background distractors to simulate large-scale retrieval
s2_distract = pl.read_csv(str(TRAIN_DIR / "train_source2.tsv"), separator="\\t", n_rows=5000).to_pandas()
s3_distract = pl.read_csv(str(TRAIN_DIR / "train_source3.tsv"), separator="\\t", n_rows=5000).to_pandas()

target_pool = pd.concat([s2_hits, s3_hits, s2_distract, s3_distract], ignore_index=True).drop_duplicates(subset=['entity_id'])

# Pre-normalize target records
target_records = []
for _, r in target_pool.iterrows():
    n_rep = normalize_text(r['business_name'])
    a_rep = normalize_address(r['business_address'])
    target_records.append({
        'entity_id': r['entity_id'],
        'country': r['country'],
        'name_norm': n_rep['unicode_normalized'],
        'name_core': n_rep['core_name'],
        'name_translit': n_rep['transliterated'],
        'addr_postal': a_rep['postal_code']
    })

# Initialize and fit adaptive blocker
blocker = AdaptiveBlocker(max_block_size=500, max_candidates_per_entity=MAX_CANDIDATES)
blocker.fit(target_records)

print(f"Indexed {len(target_records):,} target records across {len(blocker.index):,} distinct blocking keys.")
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Query blocker for benchmark S1 entities and compute candidate recall
retrieved_true = 0
total_candidates = 0
candidates_per_entity = []

t0 = time.time()
for _, r in s1_bench.iterrows():
    s1_id = r['entity_id']
    true_targets = set(bench_gt.get(s1_id, []))
    if not true_targets:
        continue

    n_rep = normalize_text(r['business_name'])
    a_rep = normalize_address(r['business_address'])
    s1_rec = {
        'entity_id': s1_id,
        'country': r['country'],
        'name_norm': n_rep['unicode_normalized'],
        'name_core': n_rep['core_name'],
        'name_translit': n_rep['transliterated'],
        'addr_postal': a_rep['postal_code']
    }

    cand_list = blocker.query(s1_rec)
    cand_ids = {c['candidate_id'] for c in cand_list}

    hits = len(cand_ids & true_targets)
    retrieved_true += hits
    total_candidates += len(cand_ids)
    candidates_per_entity.append(len(cand_ids))

benchmark_time = time.time() - t0

recall_val = (retrieved_true / all_true_matches * 100) if all_true_matches > 0 else 0.0
avg_cands = np.mean(candidates_per_entity) if candidates_per_entity else 0.0
p95_cands = np.percentile(candidates_per_entity, 95) if candidates_per_entity else 0.0

df_bench_results = pd.DataFrame([{
    "Strategy": "Multi-Channel Adaptive Blocking",
    "Candidate Recall (%)": f"{recall_val:.2f}%",
    "Retrieved Matches": f"{retrieved_true} / {all_true_matches}",
    "Avg Candidates / S1": round(avg_cands, 2),
    "P95 Candidates / S1": round(p95_cands, 1),
    "Query Runtime (s)": round(benchmark_time, 2)
}])

print("=" * 70)
print("BLOCKING RECALL BENCHMARK RESULTS")
print("=" * 70)
display(df_bench_results)
"""
    ))
    return c
