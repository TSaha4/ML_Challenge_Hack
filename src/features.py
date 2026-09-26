"""
Pairwise Feature Extraction Engine for Entity Matching.
Computes multi-field similarity scores, token overlaps, edit distances, and quality metrics.
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
import rapidfuzz.distance.Levenshtein as lev
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.fuzz as fuzz


def jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def compute_weighted_token_similarity(tokens1: List[str], tokens2: List[str], idf_dict: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Computes IDF-weighted token overlap, max rare token IDF match, and weighted Jaccard.
    """
    set1, set2 = set(tokens1), set(tokens2)
    common = set1 & set2
    
    if not common:
        return 0.0, 0.0, 0.0
    
    sum_common_idf = sum(idf_dict.get(t, 1.0) for t in common)
    sum_all_idf = sum(idf_dict.get(t, 1.0) for t in (set1 | set2))
    max_common_idf = max(idf_dict.get(t, 1.0) for t in common)
    
    weighted_jaccard = sum_common_idf / sum_all_idf if sum_all_idf > 0 else 0.0
    return sum_common_idf, max_common_idf, weighted_jaccard


def extract_pairwise_features(
    s1: Dict[str, Any],
    cand: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    idf_dict: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Extracts high-signal pairwise features between S1 and a Candidate record (S2/S3).
    """
    feats: Dict[str, float] = {}
    
    # 1. Exact match indicators
    feats['exact_name_norm'] = 1.0 if s1.get('name_norm') and s1.get('name_norm') == cand.get('name_norm') else 0.0
    feats['exact_name_core'] = 1.0 if s1.get('name_core') and s1.get('name_core') == cand.get('name_core') else 0.0
    feats['exact_country'] = 1.0 if s1.get('country') and s1.get('country') == cand.get('country') else 0.0
    feats['exact_postal'] = 1.0 if s1.get('addr_postal') and s1.get('addr_postal') == cand.get('addr_postal') and len(s1.get('addr_postal', '')) > 2 else 0.0

    # 2. String similarities on Business Name
    n1 = s1.get('name_norm', '')
    n2 = cand.get('name_norm', '')
    c1 = s1.get('name_core', '')
    c2 = cand.get('name_core', '')

    feats['name_jw'] = float(jw.similarity(n1, n2)) if (n1 and n2) else 0.0
    feats['name_core_jw'] = float(jw.similarity(c1, c2)) if (c1 and c2) else 0.0
    feats['name_fuzz_ratio'] = float(fuzz.ratio(n1, n2)) / 100.0 if (n1 and n2) else 0.0
    feats['name_fuzz_partial'] = float(fuzz.partial_ratio(n1, n2)) / 100.0 if (n1 and n2) else 0.0
    feats['name_fuzz_token_sort'] = float(fuzz.token_sort_ratio(n1, n2)) / 100.0 if (n1 and n2) else 0.0
    feats['name_fuzz_token_set'] = float(fuzz.token_set_ratio(n1, n2)) / 100.0 if (n1 and n2) else 0.0
    
    # Transliteration similarity
    t1 = s1.get('name_translit', '')
    t2 = cand.get('name_translit', '')
    feats['name_translit_jw'] = float(jw.similarity(t1, t2)) if (t1 and t2) else 0.0

    # Token overlap and Jaccard
    toks1 = n1.split()
    toks2 = n2.split()
    stoks1 = set(toks1)
    stoks2 = set(toks2)
    feats['name_token_jaccard'] = jaccard_similarity(stoks1, stoks2)
    feats['name_token_len_diff'] = abs(len(toks1) - len(toks2))
    feats['name_char_len_diff'] = abs(len(n1) - len(n2))

    # Initials match
    init1 = s1.get('name_initials', '')
    init2 = cand.get('name_initials', '')
    feats['name_initials_match'] = 1.0 if init1 and init1 == init2 else 0.0

    # 3. Address similarities
    a1 = s1.get('addr_norm', '')
    a2 = cand.get('addr_norm', '')
    feats['addr_jw'] = float(jw.similarity(a1, a2)) if (a1 and a2) else 0.0
    feats['addr_fuzz_ratio'] = float(fuzz.ratio(a1, a2)) / 100.0 if (a1 and a2) else 0.0
    feats['addr_token_sort'] = float(fuzz.token_sort_ratio(a1, a2)) / 100.0 if (a1 and a2) else 0.0
    
    atok1 = set(a1.split())
    atok2 = set(a2.split())
    feats['addr_token_jaccard'] = jaccard_similarity(atok1, atok2)
    feats['addr_char_len_diff'] = abs(len(a1) - len(a2))

    # 4. Rare Token / IDF features
    if idf_dict:
        sum_idf, max_idf, w_jaccard = compute_weighted_token_similarity(toks1, toks2, idf_dict)
        feats['name_idf_sum'] = sum_idf
        feats['name_idf_max'] = max_idf
        feats['name_idf_weighted_jaccard'] = w_jaccard
    else:
        feats['name_idf_sum'] = 0.0
        feats['name_idf_max'] = 0.0
        feats['name_idf_weighted_jaccard'] = 0.0

    # 5. Retrieval Metadata features
    if meta:
        feats['meta_channel_count'] = float(meta.get('channel_count', 1))
        feats['meta_min_rank'] = float(meta.get('min_rank', 0))
    else:
        feats['meta_channel_count'] = 1.0
        feats['meta_min_rank'] = 0.0

    return feats
