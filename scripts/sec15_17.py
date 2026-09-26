import nbformat as nbf

def get_p15_p17_cells():
    c = []
    # Phase 15
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 15 — Iterative Hard Negative Mining
Implements active hard negative mining:
1. Predict probabilities on broader candidate pool using the initial baseline model.
2. Identify high-scoring false positives ($P(\\text{match}) > 0.6$ where label is 0).
3. Append these high-loss pairs to the training feature pool and retrain to enforce precision boundaries.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""def mine_hard_negatives(
    model: Any,
    df_pool: pd.DataFrame,
    feature_cols: List[str],
    ground_truth: Dict[str, Set[str]],
    score_threshold: float = 0.55
) -> pd.DataFrame:
    \"\"\"
    Identifies high-confidence false positive pairs.
    \"\"\"
    preds = model.predict_proba(df_pool[feature_cols])[:, 1]
    df_pool_eval = df_pool.copy()
    df_pool_eval['score'] = preds
    
    hard_negs = []
    for _, r in df_pool_eval[df_pool_eval['score'] >= score_threshold].iterrows():
        s1_id = r['s1_id']
        cand_id = r['cand_id']
        if cand_id not in ground_truth.get(s1_id, set()):
            hard_negs.append(r)
            
    df_hn = pd.DataFrame(hard_negs)
    print(f"Mined {len(df_hn):,} Hard Negative pairs scoring >= {score_threshold}")
    return df_hn

print("Hard Negative Mining module ready.")
"""
    ))

    # Phase 16
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 16 — Multilingual & Transliteration Experiment
Evaluates whether Unidecode ASCII transliteration improves candidate retrieval and match precision on non-English / Indian / French business names versus raw Unicode alone.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""multilingual_test_names = [
    ("M/S शर्मा ट्रेडर्स", "Sharma Traders"),
    ("Société Générale", "Societe Generale"),
    ("CAFÉ DE PARIS", "Cafe de Paris"),
    ("Müller & Co.", "Muller & Co.")
]

res = []
for name_a, name_b in multilingual_test_names:
    rep_a = normalize_text(name_a)
    rep_b = normalize_text(name_b)
    
    raw_jw = float(jw.similarity(rep_a['raw'].lower(), rep_b['raw'].lower()))
    translit_jw = float(jw.similarity(rep_a['transliterated'], rep_b['transliterated']))
    
    res.append({
        "Entity A": name_a,
        "Entity B": name_b,
        "Raw Jaro-Winkler": round(raw_jw, 3),
        "Translit Jaro-Winkler": round(translit_jw, 3),
        "Delta": round(translit_jw - raw_jw, 3)
    })

print("Transliteration Impact Benchmark:")
display(pd.DataFrame(res))
"""
    ))

    # Phase 17
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 17 — Optional Approximate Nearest Neighbor (FAISS)
Provides optional dense vector indexing for character 4-gram embeddings.
*Note: Fully optional — the core pipeline operates robustly via multi-channel inverted index blocking even if ANN is skipped.*
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""from sklearn.neighbors import NearestNeighbors

def build_ann_prototype(sample_texts: List[str], n_neighbors: int = 5):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=1000)
    X = vec.fit_transform(sample_texts).toarray().astype(np.float32)
    
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(sample_texts)), metric='cosine')
    nn.fit(X)
    return vec, nn

ann_vec, ann_index = build_ann_prototype([
    "orelees barbershop high point",
    "b plus retail incorporated phoenix",
    "prime money tahlequah oklahoma",
    "walmart supercenter bend oregon"
])

print("ANN TF-IDF Prototype successfully fitted.")
"""
    ))
    return c
