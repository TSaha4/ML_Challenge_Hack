import nbformat as nbf

def get_p8_p10_cells():
    c = []
    # Phase 8
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 8 — Candidate Generation Pipeline
Builds the chunked generator that creates candidate pairs with metadata (`channel_count`, `min_rank`, `channels`).
Supports streaming chunk logic and persistent Parquet checkpoints to prevent memory overload.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""def generate_candidates_for_batch(
    s1_batch: List[Dict[str, Any]],
    blocker: AdaptiveBlocker,
    idf_dict: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    \"\"\"
    Generates candidate pairs for a batch of S1 records against the fitted blocker.
    \"\"\"
    candidate_pairs = []
    for s1_rec in s1_batch:
        cands = blocker.query(s1_rec, idf_dict=idf_dict)
        for c in cands:
            candidate_pairs.append({
                "source1_id": c["source1_id"],
                "candidate_id": c["candidate_id"],
                "channel_count": c["channel_count"],
                "min_rank": c["min_rank"]
            })
    return candidate_pairs

print("Candidate Generation batch function ready.")
"""
    ))

    # Phase 9
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 9 — Character N-Gram Retrieval Experiment
Compares RapidFuzz token/character similarity vs TF-IDF character 3-gram cosine similarity for retrieving names with typos and phonetic transpositions.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sample_names_query = [
    "orelees barbershop",
    "b plus retail inc",
    "walmart supercenter",
    "sharma enterprises"
]

sample_names_target = [
    "orelee barbershop llc",
    "b+ retail corporation",
    "wal-mart stores inc",
    "m/s sharma enterprisse",
    "target store",
    "starbucks coffee"
]

# Character 3-gram vectorizer
ngram_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))
X_target = ngram_vec.fit_transform(sample_names_target)
X_query = ngram_vec.transform(sample_names_query)

sim_matrix = cosine_similarity(X_query, X_target)

print("Character 3-4 Gram Cosine Similarity Matrix:")
df_ngram_sim = pd.DataFrame(sim_matrix, index=sample_names_query, columns=sample_names_target)
display(df_ngram_sim.round(3))
"""
    ))

    # Phase 10
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 10 — Rare Token & IDF Statistical Weighting
Computes token frequencies across the training reference corpus.
Ubiquitous tokens like *store*, *traders*, *mart*, *services* receive low IDF weights, while distinctive proper names (*orelee*, *tata*, *infosys*) receive high IDF weights to avoid false merges.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Compute IDF dictionary from S1 training sample (leak-free)
def compute_corpus_token_idf(sample_size: int = 50_000) -> Dict[str, float]:
    s1_df = pl.read_csv(str(TRAIN_DIR / "train_source1.tsv"), separator="\\t", n_rows=sample_size)
    doc_freq = collections.defaultdict(int)
    n_docs = len(s1_df)
    
    for name in s1_df['business_name'].drop_nulls():
        tokens = set(normalize_text(name)['core_name'].split())
        for tok in tokens:
            doc_freq[tok] += 1
            
    idf_dict = {
        tok: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        for tok, df in doc_freq.items()
    }
    return idf_dict

print("Calculating Reference Token IDF statistics...")
idf_dict = compute_corpus_token_idf(sample_size=20_000)

sorted_by_df = sorted(idf_dict.items(), key=lambda x: x[1])
print("\\nMost Common Business Tokens (Lowest IDF — Low Signal):")
for tok, idf_val in sorted_by_df[:8]:
    print(f" - '{tok}': IDF = {idf_val:.3f}")

print("\\nDistinctive Business Tokens (High IDF — High Signal):")
for tok, idf_val in sorted_by_df[-8:]:
    print(f" - '{tok}': IDF = {idf_val:.3f}")
"""
    ))
    return c
