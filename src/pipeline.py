"""
Pipeline helper functions for streaming TSV data, chunked inference, and submission export.
"""

import os
from typing import Dict, List, Set, Iterator, Optional, Any
import polars as pl
import pandas as pd


def stream_tsv_chunks(file_path: str, chunk_size: int = 100_000, n_rows: Optional[int] = None) -> Iterator[pl.DataFrame]:
    """
    Streams a large TSV file in memory-efficient Polars chunks.
    """
    reader = pl.read_csv_batched(
        file_path,
        separator="\t",
        batch_size=chunk_size,
        n_rows=n_rows,
        infer_schema_length=10000,
        ignore_errors=True
    )
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        yield batches[0]


def export_submission(
    predictions: Dict[str, Set[str]],
    test_s1_ids: List[str],
    output_matching_path: str,
    candidates: Optional[Dict[str, Set[str]]] = None,
    output_candidate_path: Optional[str] = None
):
    """
    Exports matching results and candidate pairs in the exact format required by the competition:
    - Tab-separated (.tsv)
    - Headers: source1_entity_id \t matched_entity_ids
    - Candidate headers: source1_entity_id \t candidate_entity_ids
    - Preserves all test S1 entities in exact order.
    """
    os.makedirs(os.path.dirname(output_matching_path), exist_ok=True)
    
    with open(output_matching_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1_ids:
            m_set = predictions.get(s1_id, set())
            m_str = ",".join(sorted(m_set))
            f.write(f"{s1_id}\t{m_str}\n")
            
    if candidates and output_candidate_path:
        os.makedirs(os.path.dirname(output_candidate_path), exist_ok=True)
        with open(output_candidate_path, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in test_s1_ids:
                c_set = candidates.get(s1_id, set())
                c_str = ",".join(sorted(c_set))
                f.write(f"{s1_id}\t{c_str}\n")


def compute_corpus_token_idf(train_source1_path: str, sample_size: int = 20_000) -> Dict[str, float]:
    """
    Compute token IDF weights from the Source-1 reference corpus (leak-free).

    IDF(t) = ln((1 + N) / (1 + df(t))) + 1, where df(t) counts the reference
    business names containing token t. Ubiquitous tokens such as *store* or
    *services* receive low weights while distinctive proper nouns receive high
    weights; the weights drive the rare-token blocking channel and the
    IDF-weighted name features in src/features.py.
    """
    import collections
    import math

    from src.normalization import normalize_text

    frame = pl.read_csv(str(train_source1_path), separator="\t", n_rows=sample_size)
    doc_freq: Dict[str, int] = collections.defaultdict(int)
    n_docs = len(frame)

    for name in frame["business_name"].drop_nulls():
        for token in set(normalize_text(name)["core_name"].split()):
            doc_freq[token] += 1

    return {
        token: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        for token, df in doc_freq.items()
    }
