import nbformat as nbf

def get_p18_p19_cells():
    c = []
    # Phase 18
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 18 — Graph Consistency & Entity Clustering
In entity resolution between Reference ($S1$) and Secondary Sources ($S2, S3$), each $S2$ or $S3$ record represents an atomic physical entity.
Enforces 1-to-1 matching constraints on $S2/S3$ records to prevent the same $S2$ record from being assigned to multiple different $S1$ entities when scores conflict.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""def resolve_target_conflicts(predictions: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    \"\"\"
    Enforces that each S2/S3 candidate record is mapped at most once to its highest-scoring S1 entity.
    \"\"\"
    sorted_preds = sorted(predictions, key=lambda p: p['score'], reverse=True)
    assigned_targets = set()
    s1_matched = collections.defaultdict(set)
    
    for p in sorted_preds:
        s1 = p['s1_id']
        target = p['cand_id']
        
        if target in assigned_targets:
            continue
            
        assigned_targets.add(target)
        s1_matched[s1].add(target)
        
    return s1_matched

print("Graph 1-to-1 target conflict resolver loaded.")
"""
    ))

    # Phase 19
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 19 — Prediction Confidence & Margin Analysis
Calculates confidence metrics for every Reference Entity:
- `top1_score`: Model probability for the top candidate.
- `top2_score`: Model probability for the runner-up candidate.
- `score_margin`: $P_1 - P_2$ (High margin $\\implies$ high certainty).
- Categorizes predictions into **HIGH**, **MEDIUM**, and **LOW** confidence.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""def compute_entity_confidence(candidate_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    \"\"\"
    Computes confidence metrics for candidates of a single S1 entity.
    \"\"\"
    if not candidate_scores:
        return {'confidence': 'HIGH_SINGLETON', 'top1_score': 0.0, 'margin': 1.0}
        
    scores = sorted([c['score'] for c in candidate_scores], reverse=True)
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2
    
    if top1 >= 0.80 and margin >= 0.25:
        level = "HIGH"
    elif top1 >= 0.60:
        level = "MEDIUM"
    else:
        level = "LOW"
        
    return {
        'confidence': level,
        'top1_score': round(top1, 4),
        'top2_score': round(top2, 4),
        'margin': round(margin, 4)
    }

demo_cands = [{'score': 0.91}, {'score': 0.42}, {'score': 0.15}]
print("Sample Entity Confidence Assessment:", compute_entity_confidence(demo_cands))
"""
    ))
    return c
