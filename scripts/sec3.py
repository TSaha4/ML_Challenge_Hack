import nbformat as nbf

def get_p3_cells():
    c = []
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 3 — Ground Truth & Cardinality Analysis
Inspects `train_ground_truth.tsv` to reveal:
- The exact distribution of matches per Reference Entity (`S1`).
- Proportion of singletons (entities with 0 matches).
- Proportion of Source 2 vs Source 3 matches.
- High-frequency hubs and maximum match cardinality.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Read Ground Truth sample or full depending on RUN_EXPENSIVE
gt_path = TRAIN_DIR / "train_ground_truth.tsv"
n_gt_rows = None if RUN_EXPENSIVE else 50_000

print(f"Reading Ground Truth ({'FULL' if RUN_EXPENSIVE else '50,000 rows'} sample)...")
df_gt = pl.read_csv(str(gt_path), separator="\\t", n_rows=n_gt_rows)

# Parse matched_entity_ids
df_gt_pd = df_gt.to_pandas()
df_gt_pd['matched_entity_ids'] = df_gt_pd['matched_entity_ids'].fillna('')
df_gt_pd['match_list'] = df_gt_pd['matched_entity_ids'].apply(lambda s: [x.strip() for x in s.split(',') if x.strip()])
df_gt_pd['match_count'] = df_gt_pd['match_list'].apply(len)
df_gt_pd['s2_count'] = df_gt_pd['match_list'].apply(lambda l: sum(1 for x in l if x.startswith('S2-')))
df_gt_pd['s3_count'] = df_gt_pd['match_list'].apply(lambda l: sum(1 for x in l if x.startswith('S3-')))
df_gt_pd['is_singleton'] = df_gt_pd['match_count'] == 0

n_total = len(df_gt_pd)
n_singletons = df_gt_pd['is_singleton'].sum()
singleton_pct = (n_singletons / n_total) * 100

print("=" * 60)
print(f"GROUND TRUTH STATISTICAL SUMMARY (N = {n_total:,})")
print("=" * 60)
print(f"Total Source 1 Entities Analyzed: {n_total:,}")
print(f"Singletons (Zero matches):       {n_singletons:,} ({singleton_pct:.2f}%)")
print(f"Entities with >= 1 match:        {n_total - n_singletons:,} ({100 - singleton_pct:.2f}%)")
print(f"Average Matches / S1 Entity:     {df_gt_pd['match_count'].mean():.3f}")
print(f"Median Matches / S1 Entity:      {df_gt_pd['match_count'].median():.1f}")
print(f"95th Percentile Matches:         {df_gt_pd['match_count'].quantile(0.95):.1f}")
print(f"Max Matches for Single S1:       {df_gt_pd['match_count'].max()}")
print(f"Total S2 Matches Linked:         {df_gt_pd['s2_count'].sum():,}")
print(f"Total S3 Matches Linked:         {df_gt_pd['s3_count'].sum():,}")
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Visualizing match cardinality distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Match count distribution
sns.countplot(x='match_count', data=df_gt_pd[df_gt_pd['match_count'] <= 10], ax=axes[0], palette='crest')
axes[0].set_title("Match Cardinality (0 to 10 matches)")
axes[0].set_xlabel("Number of Matched S2/S3 Records")
axes[0].set_ylabel("S1 Entity Count")

# Plot 2: Source 2 vs Source 3 contribution
df_source_contrib = pd.DataFrame({
    "Source": ["Source 2", "Source 3"],
    "Matches": [df_gt_pd['s2_count'].sum(), df_gt_pd['s3_count'].sum()]
})
sns.barplot(x="Source", y="Matches", data=df_source_contrib, ax=axes[1], palette='Blues_r')
axes[1].set_title("Total Ground Truth Matches by Target Source")
axes[1].set_ylabel("Total Matched IDs")

plt.tight_layout()
plt.show()
"""
    ))
    return c
