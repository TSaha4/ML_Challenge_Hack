# ER v2 correction plan

Target: fix measured correctness defects and make GPU accuracy experiments reproducible.
No new leaderboard score is claimed without retraining and portal evaluation.

1. Regression-test and correct character multiset overlap, empty inputs, missing neural weights.
2. Bind cached scores/features to data, code, model and neural-weight fingerprints.
3. Preserve exact submission IDs and make failed/missing validation fail explicitly.
4. Separate threshold calibration from an untouched entity holdout; exclude both candidate
   neighborhoods and true targets from every supervised stage, including the neural matcher.
5. Tune assignment thresholds at score boundaries and report blocker ceiling, pair metrics,
   country slices and worst entity errors on the independent holdout.
6. Defer new retrieval channels until candidate-ceiling/error diagnostics on the real dataset justify an ablation.
7. Test on CPU with tiny fixtures and document a clean CUDA retraining recipe.
