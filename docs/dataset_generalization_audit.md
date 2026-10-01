# Dataset and generalization audit

Audited on 27 September 2026 using the supplied train/test TSVs, root
matching_results.tsv, training_logs.txt and training_logs_v4_0968.txt. No external
business data or test labels were used.

The strongest evidence is a model-version comparison mismatch, a measurable
train/test distribution shift, and a retrieval ceiling below the requested score.
The logs show some fitting gap, but do not establish severe parameter overfitting
or general underfitting as the sole cause of the leaderboard result.

## Which scores belong together

| Run | Local macro F0.5 | Selected threshold | Prediction threshold | Portal score |
|---|---:|---:|---:|---:|
| v2 | 0.9775 | 0.65 | See historical run log | 0.970, recorded in log summary |
| v4 | 0.9844 | 0.80 | 0.75 | 0.968 |
| v5 | 0.9849 | 0.625 | 0.70 | Not supplied |

The uploaded root TSV has 1,732,544 rows, 99,102 empty predictions and 5,875,951
matches. These exactly match the v4 prediction log (training_logs_v4_0968.txt,
lines 88–94). V5 instead logs 98,264 empty predictions and 5,891,926 matches
(training_logs.txt, lines 861 onward). The dedicated v4 log explicitly identifies
v4 as the submission that scored 0.968. Thus 0.9849 versus 0.968 compares different
runs; the corresponding v4 local-to-portal gap is 0.0164.

V4's curve is nearly flat from thresholds 0.75 to 0.80 (both round to 0.9844).
The threshold override should be corrected for reproducibility, but those logs do
not support attributing the whole portal gap to a 0.05 threshold difference.
Every test partition is explicitly scored in the v4 log; although the old cache
implementation was unsafe, these logs do not demonstrate stale scores caused this
specific submission's drop.

## Submission correctness

The official validator reports PASS. Its memory-heavy ID option was left off on
the 8 GB Mac; a separate disk-backed DuckDB audit checked membership against all
9,969,589 test targets, not a sample.

- Missing/extra/duplicate Source-1 IDs: 0.
- Unknown target IDs and duplicate (Source-1, target) pairs: 0.
- Targets assigned to multiple Source-1 entities: 0.
- Cross-country matches: 0.
- Train/test Source-1 ID overlap: 0.

candidate_pairs.tsv was not supplied, so candidate-subset consistency cannot be
independently checked here. The historical v4 log records successful validation
of that file. Format validity does not establish match correctness.

## Full-data distribution differences

| Country | Training Source-1 | Training share | Test Source-1 | Test share |
|---|---:|---:|---:|---:|
| US | 1,323,633 | 59.98% | 663,106 | 38.27% |
| India | 883,188 | 40.02% | 809,986 | 46.75% |
| France | 0 | 0% | 259,452 | 14.98% |

Training has 2,206,821 references, 10,320,219 targets and 7,638,365 true matching
pairs. There are 123,247 true singletons (5.585%). Test has 1,732,544 references
and 9,969,589 targets. Targets per reference rise from 4.6765 to 5.7543 (+23.0%).
The rise is also present within countries: US 4.6742→5.7563 and India 4.6800→5.8243.
This is observable population shift, but without test labels it does not prove
that every additional target is a distractor.

The model uses raw candidate counts, same-name frequencies and sibling counts.
Those features may change meaning as the reference/target population changes.
The strongest logged v4 gains are nn_p, score_gap and reranking rank rq. Their
high gain does not establish their reliability for unseen France.

Predicted matches per test entity are US 3.4515, India 3.3585, France 3.3415.
Predicted singleton rates are approximately US 5.649%, India 5.821%, France 5.588%.
Similar match rates across countries are NOT evidence of equal accuracy: false
matches and missed matches can cancel in aggregate counts.

A random US/India holdout cannot directly measure France performance. The portal
scores a hidden subset of the provided test records; a lower score does not itself
prove that the portal introduces an unrelated, more advanced dataset.

## Overfitting and underfitting evidence from the actual logs

V4 XGBoost at iteration 100: training logloss 0.00290, early-stop logloss 0.00305.
At iteration 400: training 0.00193, early-stop 0.00277. At the last logged iteration
488: training 0.00177, early-stop 0.00277. The selected iteration was 428 (early
stopping monitored AUCPR). This is fitting pressure after validation improvements
flatten, not evidence that unlimited extra trees will help. The gap alone cannot
explain a country-shifted macro F0.5 drop: these are pairwise losses on a different
validation protocol.

V4 neural training ended with train loss 0.0062 and validation logloss 0.0066.
V5 neural validation loss continued improving across four epochs
(0.0086→0.0070→0.0058→0.0055), despite a growing final train/validation gap
(0.0041 versus 0.0055). Training loss is averaged during each changing epoch,
whereas validation is measured at its end, so those are not an exact simultaneous
generalization-gap estimate. There is no clear severe-underfitting diagnosis.

V2→v4 improved repeatedly reused local validation from 0.9775 to 0.9844 while the
reported portal result decreased from 0.970 to 0.968. This is a reason to test
country transfer, population shift and validation-set model-selection overfitting.
It does not identify which added feature/model caused the decline. The new
independent entity holdout reduces future selection bias; it cannot retroactively
make historical repeated tuning unbiased.

## Real-data CPU screening experiment

A reproducible bounded experiment selected 5,460 real Source-1 entities, all their
19,049 labelled target pairs and additional random target distractors: 29,503
targets and 369,144 candidate pairs. Production normalization, corrected
reranking/features and sibling features were used. The neural matcher and learned
transliteration were omitted. Candidate neighborhoods and true target owners for
calibration/holdout were excluded from fitting. There were 296 calibration and
263 holdout entities; the fitting matrix contained 73,486 pairs.

**These scores are not leaderboard estimates.** The reference universe is far
smaller, most unmatched targets are absent, and blocking recall is artificially
easy compared with the full task: 99.921% pair recall and a 1.0 holdout macro ceiling.

| Experiment | Train loss | Early-stop loss | Calibration F0.5 | Holdout F0.5 |
|---|---:|---:|---:|---:|
| Depth 4, stronger regularization, quarter of fit targets | 0.00518 | 0.00624 | 0.98770 | 0.98922 |
| Depth 4, stronger regularization, full fit sample | 0.00238 | 0.00278 | 0.99831 | 0.99008 |
| Depth 9, existing capacity, full fit sample | 0.00097 | 0.00217 | 0.99717 | 0.99213 |

The deeper model's holdout gain over the regularized model is 0.00206, with a
paired entity-bootstrap 95% interval of [-0.00022, 0.00514]. It is not a clear
statistical win. Conversely, there is no evidence here that shrinking the model
is the solution. More fitting data helped the regularized model's point estimate,
but this small experiment cannot set production hyperparameters.

Unseen-country stress test, trained/calibrated on US only: US holdout 0.99452,
India holdout 0.98080 (India recall 0.92876). With both countries in fitting, the
same-depth experiment had India holdout 0.98787 (recall 0.98153). India-only fitting
produced US holdout 0.99218. These are small-sample stress tests of transfer;
they do not measure or predict France's true score.

## Retrieval is a necessary improvement for 0.999

The actual v4 and v5 logs report a candidate-set oracle macro F0.5 ceiling of
**0.9942** on the 22,224-entity validation set. With those candidates fixed, no
classifier or threshold can reach 0.999 on that set. This is distinct from pair
recall and is not a measured ceiling for the hidden test labels.

Earlier logged retrieval probes show a particularly weak missing-address slice:
US R@10 was about 0.641–0.674 and India about 0.674–0.684 before later reranking
improvements. Those are historical probe numbers, not current v4 slice scores.
They identify a slice to remeasure, not a reason to blindly merge generic names.

## Measured retrieval sweep on the small fixture

The fixture contains 5,460 references, 29,503 targets and 19,049 true pairs.
It is much easier than the full reference universe and omits learned
transliteration. These are retrieval counts, not classifier scores.

| Setting changed from sample baseline | Candidate pairs | True pairs retrieved | Recovered / lost versus baseline |
|---|---:|---:|---:|
| Baseline cap=100, k=10, k_wide=80, k_char=6 | 369,144 | 19,034 | 0 / 0 |
| cap=500 | 378,789 | 19,037 | 3 / 0 |
| k_wide=200 | 372,680 | 19,035 | 1 / 0 |
| k=20 and k_char=12 | 685,424 | 19,035 | 1 / 0 |

Raising the cap recovered three of fifteen misses with 2.6% more candidates;
doubling retained candidates recovered one with 85.7% more candidates. Cap=500
is already the production default, so this is evidence about the failure mode,
not a new production improvement. All three alternatives still miss true pairs.
Observed misses include heavily altered transliterations, generic names and
records with missing addresses. Targeted retrieval changes should be evaluated
on those slices before a broad fan-out increase. Full-data false-positive impact
and memory/runtime remain unmeasured. No production retrieval defaults were
changed on this evidence. Results: screening/retrieval_report.json.

## Recommended next GPU experiment

Technical research supports testing retrieval and model capacity separately.
[Papadakis et al., 2022](https://arxiv.org/abs/2202.12521) compare blocking,
similarity joins and nearest-neighbor retrieval on ten entity-resolution datasets;
configuration materially affects both recall and precision. This motivates a
measured retrieval sweep, not an assumption that adding a larger neural model
will recover pairs removed before classification.
[XGBoost's official parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html)
describes deeper trees as more prone to overfitting and larger min_child_weight
and regularization as more conservative. Our small held-out experiments do not
establish a reliable benefit from reducing depth, so production defaults remain
unchanged.

1. Preserve v4/v5 model bundles and outputs. Use the corrected branch with a new
   artifact directory and rebuild both train and test. The latest supplied v6 log
   shows training blocking at cap=1000; test settings for v6 were not supplied.
   Train/test settings must agree. The code now checks this.
2. Establish a corrected baseline with the existing depth 9. Save full learning
   curves, selected iteration, calibrated threshold, country metrics, blocking
   ceiling and error slices on complete candidate populations. Do not select
   variants on the final holdout or tune repeatedly on the portal.
3. Compare no-neural, neural, and sibling-feature ablations using the same split
   and candidate files. Add truly held-out-country stress runs, ensuring learned
   transliteration also excludes that country's labels. Compare population/count
   feature ablations or normalized counts to assess density sensitivity.
4. Improve missing-address/generic-name retrieval where the measured oracle
   ceiling is limiting. Compare cap/k/reranking changes one at a time, with matched
   train/test settings, then retrain. More candidates alone can add false positives.
5. Compare modest depth/regularization changes on calibration only, retain the
   independently evaluated winner, and upload the exact TSV named by its manifest.

No production capacity defaults were changed merely because the portal score is
lower. Training now exposes explicit capacity and early-stopping controls, writes
learning_curves.json and fit diagnostics, and prediction records the model/output
identities, threshold and validation status in submission_manifest.json.

## Reproduction and evidence files

Run from the repository root using the project environment:

```bash
python -m scripts.audit_dataset
python -m scripts.audit_submission --matching matching_results.tsv
# CPU screening; CUDA is deliberately disabled. Requires torch and xgboost.
CUDA_VISIBLE_DEVICES="" ER_MIN_FREE_GB=0 POLARS_MAX_THREADS=2 python -m scripts.audit_generalization
CUDA_VISIBLE_DEVICES="" POLARS_MAX_THREADS=2 python -m scripts.audit_country_transfer
ER_MIN_FREE_GB=0 POLARS_MAX_THREADS=2 python -m scripts.audit_retrieval
```

On this Mac, XGBoost used the OpenMP library bundled with the locally installed
Torch wheel via DYLD_LIBRARY_PATH. Local CPU package versions differ from the
pinned CUDA environment; full-scale CUDA behavior has not been validated here.

Evidence is under artifacts/data_audit/: profiles.json, ground_truth_profile.json,
submission_audit.json, official_validator.log, screening/report.json and
screening/transfer_report.json. Audit outputs and temporary DuckDB storage are
ignored by Git. The source dataset and uploaded TSV were not modified.
