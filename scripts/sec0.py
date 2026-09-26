import nbformat as nbf

def get_p0_cells():
    c = []
    c.append(nbf.v4.new_markdown_cell(
"""# ML Challenge 2026: Competition-Grade Business Entity Resolution Solution
---
**Objective**: Build a competition-winning entity resolution pipeline that matches reference business records from **Source 1** with candidate duplicates in **Source 2** and **Source 3**.

**Evaluation Metric**: Official **Macro $F_{0.5}$** across all test Source 1 entities (Precision is weighted $2\\times$ over Recall, with strict singleton evaluation).

### High-Level Architectural Flow:
1. **Multi-Representation Normalization**: Unicode NFKC, transliteration (Unidecode), legal suffix detachment, address token parsing & postal code extraction.
2. **Multi-Channel Adaptive Blocking**: Inverted indexes with country, postal code, core name, rare token, and transliterated keys. Dynamic sub-blocking prevents combinatorial explosion without sacrificing recall.
3. **Statistical Token Weighting**: Offline reference corpus IDF computation to down-weight ubiquitous tokens (e.g., *mart*, *traders*, *store*) and elevate distinct proper nouns.
4. **Rich Pairwise Feature Engineering**: Exact identifiers, token Jaccard, RapidFuzz string similarities (Levenshtein, Jaro-Winkler, token sort/set), length diffs, and channel metadata.
5. **Entity-Aware Validation Split**: S1 entity partitioning preventing target leak.
6. **Cost-Sensitive GBDT Matching**: LightGBM pairwise ranking/classification tuned specifically to maximize $F_{0.5}$ macro score.
7. **Post-Processing & Leaderboard Validation**: Confidence margin thresholding, singleton guardrails, and compliance checks with `validate_submission.py`.
"""
    ))

    c.append(nbf.v4.new_markdown_cell(
"""# Phase 0 — Configuration & Environment Setup
Initializes all core imports, random seeds, project directory paths, dataset pointers, artifact storage, and operational control switches (`RUN_EXPENSIVE`, `SAMPLE_ROWS`, `CHUNK_SIZE`).
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""import os
import sys
import gc
import re
import math
import time
import random
import unicodedata
import collections
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator

# Core Data Science & High Performance Libraries
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Similarity & Text Normalization
import rapidfuzz
from rapidfuzz import fuzz, distance
from rapidfuzz.distance import Levenshtein as lev, JaroWinkler as jw
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# GBDT Frameworks
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Utilities & Visualization
from tqdm.auto import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local project source code is in Python path
PROJECT_ROOT = Path(os.getcwd()).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import reusable modules created in src/
from src.normalization import normalize_text, normalize_address
from src.blocking import extract_blocking_keys, AdaptiveBlocker
from src.features import extract_pairwise_features, jaccard_similarity
from src.metrics import compute_f_beta_entity, evaluate_macro_f_beta
from src.pipeline import stream_tsv_chunks, export_submission, compute_corpus_token_idf
from src.inference import run_test_inference, save_model_bundle, load_model_bundle

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline

print("Environment successfully initialized.")
print(f"Project Root: {PROJECT_ROOT}")
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Global Configuration Variables & Control Flags
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Control execution footprint during notebook exploration
RUN_EXPENSIVE = False         # Toggle to True when you want to run expensive full-dataset operations
SAMPLE_ROWS = 10_000          # Row limit for interactive fast audits and prototyping
CHUNK_SIZE = 100_000          # Batch size for streaming large TSV files
MAX_CANDIDATES = 100          # Maximum candidates per entity from blocking

# Environment-aware dataset directory discovery (supports local Windows, Linux, and Google Colab)
colab_candidate = Path("/content/drive/MyDrive/ML_challenge/student_resource/dataset")
local_candidate = PROJECT_ROOT / "student_resource" / "dataset"

if colab_candidate.exists():
    DATA_DIR = colab_candidate
elif local_candidate.exists():
    DATA_DIR = local_candidate
else:
    # Fallback to local candidate as default
    DATA_DIR = local_candidate

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

for d in [ARTIFACT_DIR, MODEL_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("Directories verified:")
print(f" - Data Dir:  {DATA_DIR} (Exists: {DATA_DIR.exists()})")
print(f" - Train Dir: {TRAIN_DIR} (Exists: {TRAIN_DIR.exists()})")
print(f" - Test Dir:  {TEST_DIR} (Exists: {TEST_DIR.exists()})")
print(f" - Artifacts: {ARTIFACT_DIR}")
print(f" - Outputs:   {OUTPUT_DIR}")
print(f"Control Flags: RUN_EXPENSIVE={RUN_EXPENSIVE}, SAMPLE_ROWS={SAMPLE_ROWS}")
"""
    ))
    return c
