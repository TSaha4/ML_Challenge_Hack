# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We treat the task as **target-centric assignment**: every Source-2/3 record belongs to at most
one Source-1 entity, so each record is matched to its most likely Source-1 entity and kept only
when a GPU-trained XGBoost model is confident enough (threshold tuned for macro F0.5). Candidate
generation is a rare-key inverted index executed on the GPU and re-ranked by character-n-gram
similarity, which keeps **98.1 %** of true matches with ~13 candidates per record. On a held-out
set of 22,224 Source-1 entities the pipeline scores **macro F0.5 = 0.9844** (pair precision 99.7 %,
recall 96.3 %). Two further ideas drive the final model: **collective evidence** from the other records
competing for the same Source-1 entity, and a **character-level neural matcher** (trained from
scratch) stacked into XGBoost.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.2M Source-1 / 10.3M Source-2+3 training records:

* **Structure.** 7,638,365 ground-truth pairs cover 7,638,365 *distinct* targets → each
  Source-2/3 record matches **at most one** Source-1 entity. 26 % of targets match nothing
  (distorted copies of businesses absent from Source-1); 5.6 % of Source-1 entities are
  singletons; a matched entity has 3.5 records on average (1–11).
* **Name noise.** case / punctuation / accents, character typos, digit-for-letter typos
  (`C0mpany`, `Hea1th`), junk prefixes (`--`, `>>`, `#`), honorifics (Dr, Shri, Smt, M/s),
  appended filler words (Center, Partners, Services), token reordering and duplication,
  legal-suffix changes (Pvt/Private, Ltd/Limited, LLC), website forms
  (`reliableanchorone.com`, `#primarycare`), aliases where a random trade name precedes the
  real one (`Ciraflux DBA: Via Heartland Security Group`, `… f/k/a …`, `… née …`), and
  **Indic-script names** (about 23 % of Indian target names: Devanagari, Gujarati, Tamil,
  Telugu, Kannada, Bengali …). Some true matches carry an entirely random name and are linked
  only by the address.
* **Address noise.** abbreviations (Rd/Road, St/Street), component reordering, state written
  as full name / code / native script, perturbed or zero-padded house numbers (`001128`,
  `4`→`7`), unit markers (`Door No`, `H.No`, `Flat No`, `N°`, `(21)`), city typos, missing
  components and ~3 % missing addresses (`None`, `null`).
* **Generic names collide.** Many unrelated businesses share names like "Pediatric Medicine
  LLC"; 32 % of Source-1 singletons have an exact-name twin in Source-2. Address agreement is
  therefore what protects precision.
* **France** (15 % of test Source-1) is absent from training, so every feature is
  language-agnostic and country is used only as a string namespace for blocking keys.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier + per-record assignment (GPU accelerated)
**Core Innovation:** target-centric assignment exploiting the "each record has ≤ 1 owner"
structure; CUDA inverted-index blocking with character-n-gram re-ranking; exact char-n-gram and
hashed-token similarity features computed on the GPU; an Indic→Latin transliteration dictionary
learned from the training pairs.

```
normalise (polars) → GPU rare-key blocking → GPU char-gram re-rank → GPU pair features
→ XGBoost (CUDA) → per-record argmax + F0.5 threshold → matching_results / candidate_pairs
```

---

## 3. Candidate Generation (Blocking)

- **Normalisation** (vectorised polars): transliteration (learned Indic dictionary, then
  Unidecode), alias splitting (DBA / f/k/a / née / aka — the real name follows the marker),
  website-name extraction, honorific and junk removal, digit-for-letter repair inside words,
  legal-form canonicalisation (`private`→`pvt`, `limited`→`ltd`, …), token de-duplication; for
  addresses: abbreviation canonicalisation (US / India / France street types and directions),
  state names → codes, unit-marker removal, leading-zero stripping.
- **Blocking keys used** (each prefixed by the record's country string):
  core-name tokens, the space-less core name (so `reliableanchorone.com` meets "Reliable
  Anchor One"), address tokens and adjacent address-token bigrams (`15277 spencer`).
  Keys hashed to 64-bit integers.
- **Scoring and top-k (GPU).** Source-1 keys with document frequency ≤ 500 form a sorted
  inverted index in VRAM. For each record the matched postings are expanded with
  `searchsorted`/`repeat_interleave`, and each (record, Source-1) pair is scored by the summed
  IDF (`ln N/df`) of shared name keys and shared address keys.
- **Character re-rank (GPU).** The IDF top-200 per record are re-scored with character 3-gram
  Jaccard of core name and address; a pair is kept if it is in the IDF top-10 **or** the
  character top-6. This rescues near-duplicates whose tokens differ slightly (`296` vs `2296`,
  typos) and was the single largest improvement.
- **Candidate pairs generated:** 133.5M test pairs for 9,969,589 test records (13.4 per record);
  136.3M on train. Only 9 of 1,732,544 test Source-1 entities have no candidate (1,859 before the re-rank).
- **How true matches were not lost:** multi-channel keys (name, squashed name, address tokens
  and bigrams) so a random name or a missing address still leaves another route; the
  df ≤ 500 cap drops only near-useless keys; the character re-rank covers token-level typos.
  **Measured recall of the candidate set on all 7.64M training pairs: 98.08 %**
  (95.74 % before the character re-rank). Blocking runs in ~15 minutes per split on an RTX 3060.

---

## 4. Matching Model

**Features used** (84, all computed on the GPU except three rapidfuzz scores):
- **Name features:** exact char 2-gram and 3-gram multiset Jaccard and both containments for the
  normalised name, core name and space-less name; hashed-token Jaccard and IDF-weighted token
  overlap/coverage of the core name; the alias-prefix overlap; rapidfuzz `ratio` (name) and
  `token_set_ratio` (core name); exact core / space-less equality; **name rarity** — how many
  Source-1 entities in the country carry exactly this core name (both sides of the pair).
- **Address features:** char 2/3-gram overlaps, hashed-token Jaccard and IDF-weighted coverage,
  numeric-token Jaccard, first house-number equality, missing-number flag, rapidfuzz
  `token_set_ratio`, token counts and lengths, missing-address flags.
- **Other / competition context:** blocking IDF scores (name, address), shared-key count, IDF rank
  and character rank of this Source-1 among the record's candidates, character score, margin
  over the best competing Source-1, score relative to the record's best, number of candidates,
  how many records rank this Source-1 first, record flags (source 2 vs 3, alias, website-name,
  Indic script).

- **Collective (sibling) features:** for pair (record t, Source-1 s), the other records whose
  top candidate is also s: how many share t's house number / full address / all numbers / core
  name / source, how many agree with s's own house number, and the fractions. Rationale found in
  EDA: a source perturbs a business's address once and all its copies share it, so on train a
  record whose number disagrees with s is a true match 13 % of the time alone but 64 % / 87 % when
  one / two siblings share that number. Also numeric house-number distance (absolute and log).
- **Neural feature `nn_p`:** probability from a character-level decomposable-attention matcher
  (byte embedding → 2 conv layers per field; name and address of both sides soft-aligned to each
  other; [x, aligned, x−aligned, x·aligned] → pooled → MLP; ~0.3M parameters, trained from
  scratch with AMP on 5.3M pairs, 2 epochs, held-out logloss 0.0066 from text alone). It is trained
  on a reserved record fold (`tid % 10 == 7`) that XGBoost never trains on, so the stacked feature
  is out-of-sample.

**Model type:** XGBoost gradient-boosted trees (`device=cuda`, depth 9, eta 0.08, early stopping on
held-out records → 428 rounds) over the 83 engineered features + `nn_p`, trained on 10.5M labelled
pairs from 800k records; features streamed from disk into a GPU `QuantileDMatrix`. Both models are
our own (XGBoost Apache-2.0, PyTorch BSD) and far below the 8B-parameter limit.

**Decision rule / threshold selection:** each record is assigned to its highest-probability
Source-1 candidate if `p ≥ threshold`. The threshold is chosen by grid search of the **official
macro F0.5** (per-entity F0.5 averaged over all Source-1 entities, singletons included) on the
validation entities. The curve is flat between 0.60 and 0.80 (0.9842–0.9844); we submit 0.75.
A per-entity expected-F0.5 decision rule was also evaluated and gave no gain (+0.0002).

**Validation protocol:** Source-1 ids with `id % 100 == 0` (22,224 entities, ~1.28M records that
have one of them as a candidate) are excluded from training, from early stopping and from the
transliteration dictionary; every record that could be assigned to them is scored, so false
merges into validation entities are counted.

---

## 5. Results & Error Analysis

| Version | Candidate recall | Macro F0.5 | Singleton F0.5 | Matched F0.5 | Pair P | Pair R |
|---|---|---|---|---|---|---|
| v1: IDF blocking, 500k training records | 95.7 % | 0.9678 | 0.973 | 0.968 | 0.994 | 0.928 |
| v2: + char re-rank, typo repair, name rarity, 800k records | 98.1 % | 0.9775 | 0.970 | 0.978 | 0.993 | 0.952 |
| v3: + collective sibling features, house-number distance | 98.1 % | 0.9804 | 0.972 | 0.981 | 0.995 | 0.960 |
| **v4: + stacked neural matcher (`nn_p`)** | **98.1 %** | **0.9844** | 0.978 | 0.985 | 0.997 | 0.963 |

- **F_0.5 Score (macro):** **0.9844** on held-out validation entities (thr 0.75).
- **Where the remaining score is lost (v1 analysis):** ~85 % of the loss is **missed** records, not
  wrong merges — precision is already ~99.3 %.
- **Common false positives (wrong merges):** mostly records of businesses absent from Source-1
  that share a generic name *and* a nearby address with a Source-1 entity (same street, close house
  number); a smaller share are records attached to the wrong one of several same-name entities.
- **Common false negatives (missed matches):** records with a missing address and a generic name
  (many Source-1 entities share it — genuinely ambiguous); records whose only link is a perturbed
  address with a random name; same-street / different-house-number cases where the model is
  (rightly) unsure; a few records never retrieved by blocking.
- **Test output:** 1,732,544 rows, 5.80M matched records; France, India and US receive very
  similar match rates (3.37 / 3.33 / 3.36 matches per entity; 5.4 / 5.9 / 5.8 % predicted
  singletons), suggesting the language-agnostic features transfer to the unseen country.

---

## 6. Conclusion

Exploiting the one-owner-per-record structure turns entity resolution into a per-record ranking
problem that a GPU pipeline solves end to end on a laptop (6 GB GPU, ≤ 5 GB RAM) in about an hour.
The biggest lesson: blocking quality dominated — recovering near-duplicate candidates with a
character-level re-rank was worth more than any model change, while the F0.5 metric rewarded a
precision-leaning, per-record decision rule.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its `README.md` for exact commands.

| Path | Role |
|---|---|
| `src/er/steps/prepare.py` | learn transliteration dictionary; normalise a split → parquet |
| `src/er/steps/block.py` | GPU candidate generation → candidate parts |
| `src/er/steps/train.py` | GPU features, XGBoost training, validation, threshold tuning |
| `src/er/steps/predict.py` | score test candidates, write `output/*.tsv`, run the validator |
| `src/er/text.py`, `translit.py` | normalisation, learned transliteration |
| `src/er/blocking.py`, `gpu_blocking.py` | blocking keys, CUDA index / top-k / re-rank |
| `src/er/features.py`, `gpu_features.py` | candidate context, CUDA pair features |
| `src/er/model.py`, `io.py`, `resources.py` | model/assignment/metric, id encoding, RAM guard |

Entry points: `python -m src.er.steps.prepare --split train|test`, `python -m src.er.steps.block
--split train|test`, `python -m src.er.steps.train`, `python -m src.er.steps.predict`.

### B. Additional Results

- Blocking recall on train at different settings (30k-record sample): IDF top-10 only 96.1 %;
  + char re-rank of IDF top-40: 97.2 %; top-100: 97.9 %; top-200: 98.3 %.
- Most important features (XGBoost gain): character rank of the candidate, margin over the best
  competitor, first house-number equality, address `token_set_ratio`, name 2-gram containment,
  IDF rank.
- A "sibling support" rule (accept lower-probability records when the entity already has a very
  confident record) was evaluated and gave no gain, so it is not used.
- No external data, APIs or services are used. Built-in knowledge is limited to normalisation
  tables (US/Indian state names → codes, street-type abbreviations, legal forms, honorifics).
