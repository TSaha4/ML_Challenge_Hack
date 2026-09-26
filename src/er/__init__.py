"""
Vectorised, target-centric entity-resolution pipeline (v2).

Every Source-2/3 record belongs to at most one Source-1 entity, so the pipeline
retrieves the best Source-1 candidates *per target record*, scores each
(target, Source-1) pair with LightGBM and assigns the target to its best
Source-1 entity when the score clears an F_0.5-tuned threshold.

Modules
-------
io        : TSV loading and compact integer entity-id encoding
text      : polars-vectorised name / address normalisation
translit  : Indic-script token transliteration dictionary learned from training pairs
blocking  : rare-key inverted-index candidate generation (chunked polars joins)
features  : pairwise features (rapidfuzz cpdist, multi-threaded C++)
model     : LightGBM training, assignment and threshold tuning
"""
