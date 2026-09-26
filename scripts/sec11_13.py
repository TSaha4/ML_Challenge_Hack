import nbformat as nbf

def get_p11_p13_cells():
    c = []
    # Phase 11
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 11 — Pairwise Feature Engineering
Computes high-signal pairwise features for candidate pairs:
- **String Similarities**: Jaro-Winkler, Levenshtein distance ratio, token sort ratio, token set ratio.
- **Token Overlap**: Token Jaccard, token length difference, character length difference.
- **IDF-Weighted Overlap**: Max matching token IDF, sum of matching IDF, weighted Jaccard.
- **Address Signals**: Normalized address Jaro-Winkler, postal code exact equality.
- **Graph & Blocking Metadata**: Channel counts and candidate retrieval min rank.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Feature extraction function verification (defined in src/features.py)
s1_sample = {
    'name_norm': "orelee s barbershop inc",
    'name_core': "orelee s barbershop",
    'name_translit': "orelee s barbershop inc",
    'name_initials': "osbi",
    'addr_norm': "1795 westchester drive high point nc",
    'addr_postal': "27262",
    'country': "US"
}

cand_sample = {
    'name_norm': "orelees barbershop llc",
    'name_core': "orelees barbershop",
    'name_translit': "orelees barbershop llc",
    'name_initials': "obl",
    'addr_norm': "1795 westchester dr high point nc",
    'addr_postal': "27262",
    'country': "US"
}

meta_sample = {"channel_count": 3, "min_rank": 0}

feats = extract_pairwise_features(s1_sample, cand_sample, meta=meta_sample, idf_dict=idf_dict)
print(f"Extracted {len(feats)} pairwise features:")
for k, v in list(feats.items())[:12]:
    print(f" - {k:<28}: {v:.4f}")
print("   ... (additional features computed)")
"""
    ))

    # Phase 12
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 12 — Training Dataset Construction (Hard Negatives)
Constructs balanced training pairs:
- **Positives**: Ground-truth matched pairs.
- **Hard Negatives**: Blocking candidates that share high token/address similarity or postal code but are **not** true matches according to ground truth. Avoids naive random negatives.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""def construct_training_dataset(
    s1_records: Dict[str, Dict[str, Any]],
    target_records: Dict[str, Dict[str, Any]],
    candidate_pairs: List[Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    max_neg_ratio: int = 4
) -> Tuple[pd.DataFrame, np.ndarray]:
    \"\"\"
    Constructs feature matrix X and label vector y for candidate pairs.
    Includes all positives, and samples up to max_neg_ratio hard negatives per positive.
    \"\"\"
    rows = []
    labels = []
    
    pos_count = 0
    neg_count = 0
    
    for c in candidate_pairs:
        s1_id = c['source1_id']
        cand_id = c['candidate_id']
        
        if s1_id not in s1_records or cand_id not in target_records:
            continue
            
        true_targets = ground_truth.get(s1_id, set())
        is_positive = 1 if cand_id in true_targets else 0
        
        if not is_positive and neg_count >= (pos_count + 1) * max_neg_ratio:
            continue
            
        s1 = s1_records[s1_id]
        cand = target_records[cand_id]
        meta = {'channel_count': c['channel_count'], 'min_rank': c['min_rank']}
        
        f = extract_pairwise_features(s1, cand, meta=meta, idf_dict=idf_dict)
        f['s1_id'] = s1_id
        f['cand_id'] = cand_id
        rows.append(f)
        labels.append(is_positive)
        
        if is_positive:
            pos_count += 1
        else:
            neg_count += 1
            
    df_feat = pd.DataFrame(rows)
    y = np.array(labels)
    return df_feat, y

print("Training dataset construction pipeline ready.")
"""
    ))

    # Phase 13
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 13 — Entity-Aware Train / Validation Split
**Strict Anti-Leakage Guard**: Splits by `source1_entity_id` so that an entire reference entity and all its candidate pairs exist **either** in Train **or** in Validation — never both.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""def create_entity_aware_split(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.25,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    \"\"\"
    Partitions features and labels strictly by S1 entity ID.
    \"\"\"
    unique_entities = df_features['s1_id'].unique()
    train_entities, val_entities = train_test_split(
        unique_entities, test_size=test_size, random_state=random_state
    )
    
    train_mask = df_features['s1_id'].isin(train_entities)
    val_mask = df_features['s1_id'].isin(val_entities)
    
    X_train = df_features[train_mask].copy()
    y_train = labels[train_mask]
    
    X_val = df_features[val_mask].copy()
    y_val = labels[val_mask]
    
    print("=" * 60)
    print("ENTITY-AWARE SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total S1 Entities:     {len(unique_entities):,}")
    print(f"Train S1 Entities:     {len(train_entities):,} ({len(train_entities)/len(unique_entities)*100:.1f}%)")
    print(f"Val S1 Entities:       {len(val_entities):,} ({len(val_entities)/len(unique_entities)*100:.1f}%)")
    print(f"Train Pairs:           {len(X_train):,} (Pos: {y_train.sum():,}, Neg: {len(y_train)-y_train.sum():,})")
    print(f"Val Pairs:             {len(X_val):,} (Pos: {y_val.sum():,}, Neg: {len(y_val)-y_val.sum():,})")
    
    return X_train, X_val, y_train, y_val

print("Entity-aware split function compiled.")
"""
    ))
    return c
