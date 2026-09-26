"""
Evaluation Metrics for Entity Resolution: Macro F_0.5 Calculation.
Matches the official competition evaluation rules exactly.
"""

from typing import Dict, List, Set, Union


def compute_f_beta_entity(y_true: Set[str], y_pred: Set[str], beta: float = 0.5) -> float:
    """
    Computes F_beta score for a single reference entity (Source 1).
    Special rules as specified in competition statement:
    - If true is empty and pred is empty: score is 1.0 (correct singleton).
    - If true is empty and pred is not empty: score is 0.0 (false positive on singleton).
    - If true is not empty and pred is empty: score is 0.0 (missed match).
    - Otherwise standard F_beta with beta=0.5 (precision-heavy: 1.25 * P * R / (0.25 * P + R)).
    """
    if not y_true and not y_pred:
        return 1.0
    if not y_true or not y_pred:
        return 0.0

    tp = len(y_true & y_pred)
    if tp == 0:
        return 0.0

    precision = tp / len(y_pred)
    recall = tp / len(y_true)

    beta_sq = beta ** 2
    f_score = (1.0 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
    return float(f_score)


def evaluate_macro_f_beta(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes overall Macro F_beta across all entities in ground truth.
    Returns:
    - macro_f_beta: average score across all entities
    - singleton_count: count of true singletons
    - non_singleton_count: count of entities with matches
    """
    total_score = 0.0
    n = len(ground_truth)
    if n == 0:
        return {"macro_f_beta": 0.0, "singleton_f_beta": 0.0, "matched_f_beta": 0.0}

    singleton_scores = []
    matched_scores = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_f_beta_entity(true_set, pred_set, beta=beta)
        total_score += score
        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            matched_scores.append(score)

    return {
        "macro_f_beta": total_score / n,
        "singleton_f_beta": sum(singleton_scores) / len(singleton_scores) if singleton_scores else 1.0,
        "matched_f_beta": sum(matched_scores) / len(matched_scores) if matched_scores else 0.0,
        "n_entities": float(n),
        "n_singletons": float(len(singleton_scores)),
        "n_with_matches": float(len(matched_scores))
    }
