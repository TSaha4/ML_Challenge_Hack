import nbformat as nbf

def get_p14_cells():
    c = []
    # Phase 14
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 14 — Model Training (LightGBM, XGBoost, CatBoost) & $F_{0.5}$ Tuning
Trains a precision-optimized GBDT classifier, sweeps classification probability thresholds, and optimizes for official **Macro $F_{0.5}$**.
Provides separate cells for LightGBM baseline, XGBoost benchmark, and CatBoost benchmark so you can run the model of your choice.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Demonstration training run on prototype feature set
# Synthesizing prototype training data for pipeline verification
proto_s1 = {}
proto_targets = {}
proto_pairs = []
proto_gt = {}

for idx, r in df_noise_samples.iterrows():
    s1_id = r['S1 ID']
    t_id = r['Target ID']
    
    s1_norm = normalize_text(r['S1 Name'])
    s1_addr = normalize_address(r['S1 Address'])
    cand_norm = normalize_text(r['Target Name'])
    cand_addr = normalize_address(r['Target Address'])
    
    proto_s1[s1_id] = {
        'name_norm': s1_norm['unicode_normalized'],
        'name_core': s1_norm['core_name'],
        'name_translit': s1_norm['transliterated'],
        'name_initials': s1_norm['initials'],
        'addr_norm': s1_addr['normalized'],
        'addr_postal': s1_addr['postal_code'],
        'country': r['Country']
    }
    
    proto_targets[t_id] = {
        'name_norm': cand_norm['unicode_normalized'],
        'name_core': cand_norm['core_name'],
        'name_translit': cand_norm['transliterated'],
        'name_initials': cand_norm['initials'],
        'addr_norm': cand_addr['normalized'],
        'addr_postal': cand_addr['postal_code'],
        'country': r['Country']
    }
    
    proto_gt[s1_id] = {t_id}
    # True pair
    proto_pairs.append({'source1_id': s1_id, 'candidate_id': t_id, 'channel_count': 3, 'min_rank': 0})

# Add hard negative cross-pairs
all_tids = list(proto_targets.keys())
for s1_id in proto_s1:
    for neg_tid in all_tids:
        if neg_tid not in proto_gt[s1_id]:
            proto_pairs.append({'source1_id': s1_id, 'candidate_id': neg_tid, 'channel_count': 1, 'min_rank': 5})

df_feat_demo, y_demo = construct_training_dataset(proto_s1, proto_targets, proto_pairs, proto_gt)
feature_cols = [col for col in df_feat_demo.columns if col not in ['s1_id', 'cand_id']]

print(f"Synthesized Prototype Feature Matrix: {df_feat_demo.shape}")
print(f"Features: {feature_cols}")
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Train Baseline LightGBM Model
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 150,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

model_lgb = lgb.LGBMClassifier(**lgb_params)
model_lgb.fit(df_feat_demo[feature_cols], y_demo)

# Feature Importance
df_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model_lgb.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=df_imp.head(15), palette='viridis')
plt.title("LightGBM Feature Importance (Top 15)")
plt.xlabel("Split Gain Importance")
plt.tight_layout()
plt.show()
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# [Optional Benchmark] Train XGBoost Model
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 150,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'tree_method': 'hist'
}

model_xgb = xgb.XGBClassifier(**xgb_params)
model_xgb.fit(df_feat_demo[feature_cols], y_demo)
print("XGBoost training finished successfully.")
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# [Optional Benchmark] Train CatBoost Model
cb_params = {
    'loss_function': 'Logloss',
    'iterations': 150,
    'learning_rate': 0.05,
    'depth': 6,
    'random_seed': RANDOM_SEED,
    'verbose': 0
}

model_cb = cb.CatBoostClassifier(**cb_params)
model_cb.fit(df_feat_demo[feature_cols], y_demo)
print("CatBoost training finished successfully.")
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# F_0.5 Threshold Tuning Sweep
scores = model_lgb.predict_proba(df_feat_demo[feature_cols])[:, 1]
df_eval = df_feat_demo[['s1_id', 'cand_id']].copy()
df_eval['score'] = scores

best_thresh = 0.5
best_macro_f05 = -1.0
thresholds = np.arange(0.3, 0.95, 0.05)
thresh_results = []

for th in thresholds:
    pred_dict = collections.defaultdict(set)
    for _, r in df_eval[df_eval['score'] >= th].iterrows():
        pred_dict[r['s1_id']].add(r['cand_id'])
        
    metrics = evaluate_macro_f_beta(proto_gt, pred_dict, beta=0.5)
    f05 = metrics['macro_f_beta']
    thresh_results.append({'Threshold': round(th, 2), 'Macro F_0.5': round(f05, 4)})
    if f05 > best_macro_f05:
        best_macro_f05 = f05
        best_thresh = round(th, 2)

print(f"Optimal F_0.5 Threshold: {best_thresh} (Best Macro F_0.5 = {best_macro_f05:.4f})")
display(pd.DataFrame(thresh_results).T)

# Persist the tuned model bundle (model + feature order + IDF weights + threshold) so the
# full-scale test inference can be reproduced from the command line as well as here.
model_bundle_dir = save_model_bundle(
    model=model_lgb,
    feature_cols=feature_cols,
    idf_dict=idf_dict,
    threshold=best_thresh,
    bundle_dir=MODEL_DIR / "lgb_f05"
)
print(f"Saved model bundle for full-test inference: {model_bundle_dir}")
print("Reproduce the submission outputs with:")
print(f"  python -m src.inference --model-dir {model_bundle_dir} --output-dir {OUTPUT_DIR}")
"""
    ))
    return c
