"""Content identities and atomic writes for reproducible, resumable experiments."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

# Increment whenever feature meanings or preprocessing semantics change.
FEATURE_VERSION = 2
ART = Path(os.environ.get('ER_ARTIFACT_DIR', 'artifacts/er'))


def fingerprint(paths, config=None) -> str:
    """Hash ordered file contents and settings; identities survive directory moves."""
    h = hashlib.sha256(json.dumps(config or {}, sort_keys=True).encode())
    for path in sorted(map(Path, paths), key=str):
        file_hash = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                file_hash.update(chunk)
        h.update(file_hash.digest())
    return h.hexdigest()


def pipeline_files():
    return sorted(Path(__file__).parent.rglob('*.py'))


def runtime_versions():
    return {name: importlib.metadata.version(name)
            for name in ('numpy', 'polars', 'pyarrow', 'torch', 'xgboost', 'RapidFuzz')}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def write_manifest(path, identity, **details):
    write_json(path, {'identity': identity, **details})


def require_manifest(path, identity):
    path = Path(path)
    if not path.exists() or json.loads(path.read_text()).get('identity') != identity:
        raise ValueError(f'Stale or incomplete cache: {path}. Rebuild without --reuse-features.')


def atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.parquet.tmp')
    frame.write_parquet(tmp)
    tmp.replace(path)


def require_current_blocking(root, split):
    """Old candidates were reranked with different similarity semantics."""
    path = Path(root) / f'{split}_cands' / 'manifest.json'
    if not path.exists() or json.loads(path.read_text()).get('feature_version') != FEATURE_VERSION:
        raise ValueError(f'Rebuild {split} preparation and blocking with the current code before training/prediction.')

    manifest = json.loads(path.read_text())
    current = fingerprint([Path(root) / f'{split}_s1.parquet', Path(root) / f'{split}_tg.parquet'])
    if manifest['identity'] != current:
        raise ValueError(f'{split} normalized inputs changed. Rebuild blocking and its sibling tables.')
