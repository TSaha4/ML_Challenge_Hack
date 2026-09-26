import nbformat as nbf

def get_p4_cells():
    c = []
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 4 — Match Noise Analysis
Pairs ground truth true matches to inspect the real-world corruption patterns in:
1. **Business Names**: Acronyms, legal entity suffixes (*Inc*, *Corp*, *Pvt Ltd*), token reordering, typographical errors, and transliterations.
2. **Business Addresses**: Missing pin codes, road abbreviations, building landmarks, and missing administrative divisions.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# FAST: Construct True Matched Pairs by joining S1 with S2 and S3 samples
def build_sample_matched_pairs(num_samples: int = 10) -> pd.DataFrame:
    # 1. Read a small batch of non-singleton entities from ground truth
    gt_scan = pl.scan_csv(str(TRAIN_DIR / "train_ground_truth.tsv"), separator="\\t")
    non_singles = gt_scan.filter(
        pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')
    ).limit(num_samples * 3).collect()
    
    target_s1_ids = non_singles['source1_entity_id'].to_list()
    s1_to_targets = dict(zip(
        non_singles['source1_entity_id'],
        [m.split(',') for m in non_singles['matched_entity_ids']]
    ))
    
    needed_s2 = [m.strip() for ids in s1_to_targets.values() for m in ids if m.strip().startswith('S2-')]
    needed_s3 = [m.strip() for ids in s1_to_targets.values() for m in ids if m.strip().startswith('S3-')]
    
    # 2. Retrieve corresponding records efficiently via indexed scans
    s1_sub = pl.scan_csv(str(TRAIN_DIR / "train_source1.tsv"), separator="\\t").filter(
        pl.col('entity_id').is_in(target_s1_ids)
    ).collect().to_pandas()
    
    s2_sub = pl.scan_csv(str(TRAIN_DIR / "train_source2.tsv"), separator="\\t").filter(
        pl.col('entity_id').is_in(needed_s2)
    ).collect().to_pandas()
    
    s3_sub = pl.scan_csv(str(TRAIN_DIR / "train_source3.tsv"), separator="\\t").filter(
        pl.col('entity_id').is_in(needed_s3)
    ).collect().to_pandas()
    
    target_lookup = pd.concat([s2_sub, s3_sub], ignore_index=True).set_index('entity_id').to_dict('index')
    
    matched_pairs = []
    for _, s1_row in s1_sub.iterrows():
        s1_id = s1_row['entity_id']
        targets = s1_to_targets.get(s1_id, [])
        for tid in targets:
            tid = tid.strip()
            if tid in target_lookup:
                cand_row = target_lookup[tid]
                matched_pairs.append({
                    "S1 ID": s1_id,
                    "Target ID": tid,
                    "Country": s1_row['country'],
                    "S1 Name": s1_row['business_name'],
                    "Target Name": cand_row['business_name'],
                    "S1 Address": s1_row['business_address'],
                    "Target Address": cand_row['business_address']
                })
            if len(matched_pairs) >= num_samples:
                break
        if len(matched_pairs) >= num_samples:
            break
            
    return pd.DataFrame(matched_pairs)

df_noise_samples = build_sample_matched_pairs(num_samples=10)
print(f"Displaying {len(df_noise_samples)} True Ground-Truth Pairs for Noise Profiling:")
for idx, r in df_noise_samples.iterrows():
    print(f"\\n[{idx+1}] Country: {r['Country']}")
    print(f" S1:     {r['S1 ID']} | Name: '{r['S1 Name']}'")
    print(f" Target: {r['Target ID']} | Name: '{r['Target Name']}'")
    print(f" S1 Addr:     '{r['S1 Address']}'")
    print(f" Target Addr: '{r['Target Address']}'")
    print("-" * 75)
"""
    ))
    return c
