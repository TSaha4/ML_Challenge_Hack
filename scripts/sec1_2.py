import nbformat as nbf

def get_p1_p2_cells():
    c = []
    # Phase 1
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 1 — Data Discovery
Scans the dataset directory, prints file sizes, and safely reads a small preview sample from each file without loading entire multi-gigabyte files into RAM.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# 1. Discover all files, extensions, and disk sizes in dataset directories
def scan_dataset_directory(base_dir: Path) -> pd.DataFrame:
    records = []
    for f in sorted(base_dir.rglob("*.tsv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        records.append({
            "File Name": f.name,
            "Folder": f.parent.name,
            "Path": str(f.relative_to(PROJECT_ROOT)),
            "Size (MB)": round(size_mb, 2)
        })
    return pd.DataFrame(records)

df_files = scan_dataset_directory(DATA_DIR)
print(f"Discovered {len(df_files)} TSV files:")
display(df_files)
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# 2. Inspect headers and read a small 5-row preview sample from each TSV
print("=" * 80)
print("PREVIEWING SCHEMA AND FIRST 5 ROWS PER FILE (SAFE READ)")
print("=" * 80)

for idx, row in df_files.iterrows():
    path = PROJECT_ROOT / row['Path']
    sample_df = pl.read_csv(str(path), separator="\\t", n_rows=5, ignore_errors=True)
    print(f"\\nFile: {row['File Name']} | Shape Preview: {sample_df.shape}")
    print(f"Columns: {sample_df.columns}")
    display(sample_df.to_pandas())
"""
    ))

    # Phase 2
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 2 — Data Audit & Schema Profiling
Profiles row counts, column null rates, unique entities, country distributions, and string lengths across sources.

| Operation | Complexity | Description |
|---|---|---|
| FAST | O(N) single-pass | Missing value check, country distribution on sample |
| MEDIUM | O(N) streaming | Line counts and exact total record sizes |
| ⚠️ EXPENSIVE | Multi-GB aggregation | Full unique count and field cardinality over 10M+ rows |
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# FAST: Schema and missing value audit on 10,000-row samples
audit_records = []

for idx, row in df_files.iterrows():
    path = PROJECT_ROOT / row['Path']
    df_sample = pl.read_csv(str(path), separator="\\t", n_rows=SAMPLE_ROWS, ignore_errors=True)
    
    for col in df_sample.columns:
        s = df_sample[col]
        null_count = s.null_count()
        empty_str_count = 0
        mean_len = 0.0
        max_len = 0
        if s.dtype == pl.Utf8 or s.dtype == pl.String:
            lens = s.str.len_bytes().drop_nulls()
            empty_str_count = (s == "").sum()
            mean_len = float(lens.mean()) if len(lens) > 0 else 0.0
            max_len = int(lens.max()) if len(lens) > 0 else 0
            
        audit_records.append({
            "File": row['File Name'],
            "Column": col,
            "Dtype": str(s.dtype),
            "Null Count": null_count,
            "Null %": round((null_count / len(df_sample)) * 100, 2),
            "Empty %": round((empty_str_count / len(df_sample)) * 100, 2),
            "Avg Str Len": round(mean_len, 1),
            "Max Str Len": max_len
        })

df_audit = pd.DataFrame(audit_records)
print("Sample-based Schema Audit (Sample Size: 10,000 rows):")
display(df_audit)
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# FAST: Country label distribution in Train vs Test samples
print("Country Distribution Audit:")
for folder, prefix in [("train", "train"), ("test", "test")]:
    for src in [1, 2, 3]:
        fname = f"{prefix}_source{src}.tsv"
        p = DATA_DIR / folder / fname
        if p.exists():
            df_c = pl.read_csv(str(p), separator="\\t", columns=["country"], n_rows=SAMPLE_ROWS)
            dist = df_c["country"].value_counts().to_pandas().to_dict(orient="records")
            print(f" {fname} Country Distribution (sample): {dist}")
"""
    ))

    c.append(nbf.v4.new_markdown_cell(
"""### ⚠️ EXPENSIVE CELL: Full Dataset Row Counting
Counts the exact line counts across all multi-GB training and test files using buffered fast binary stream counting.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# MEDIUM/EXPENSIVE: Exact row counts across all files
if not RUN_EXPENSIVE:
    print("RUN_EXPENSIVE is False. Displaying known approximate file statistics.")
    print(" - train_source1.tsv: ~2.21M rows")
    print(" - train_source2.tsv: ~5.03M rows")
    print(" - train_source3.tsv: ~5.06M rows")
    print(" - train_ground_truth.tsv: ~2.21M rows")
    print(" - test_source1.tsv: ~1.71M rows")
    print(" - test_source2.tsv: ~5.10M rows")
    print(" - test_source3.tsv: ~5.08M rows")
    print("Set RUN_EXPENSIVE = True in Phase 0 to re-calculate exact counts from disk.")
else:
    print("Computing exact line counts via buffered binary scan...")
    def count_lines(fp: Path) -> int:
        lines = 0
        buf_size = 1024 * 1024 * 8
        with open(fp, 'rb') as f:
            while True:
                buf = f.read(buf_size)
                if not buf:
                    break
                lines += buf.count(b'\\n')
        return max(0, lines - 1)

    for idx, row in df_files.iterrows():
        exact_n = count_lines(PROJECT_ROOT / row['Path'])
        print(f" - {row['File Name']}: {exact_n:,} records")
"""
    ))
    return c
