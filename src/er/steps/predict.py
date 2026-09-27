"""Step 4: score the test candidates and write the two submission files.

    python -m src.er.steps.predict [--threshold 0.6]

Streams the test candidate parts (1M targets each): GPU featurisation -> GPU
XGBoost scoring -> per-target argmax assignment above the tuned threshold.
Writes ``output/matching_results.tsv`` and ``output/candidate_pairs.tsv`` (the exact
pair set the model scored), one row per test Source-1 entity in file order, then
runs the official validator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from src.er.io import DATA_DIR
from src.er.resources import lower_priority, rss_gb, wait_for_ram

from src.er.artifacts import ART


def write_lists(s1_order: pl.DataFrame, pairs: pl.LazyFrame, header: str, path: Path,
                chunk: int = 200_000, *, target_ids: pl.LazyFrame):
    """Preserve original ID spelling, including leading zeroes, and all singletons."""
    lookup = target_ids.select(pl.col('id').alias('tid'), pl.col('entity_id').alias('_target_id'))
    joined = pairs.unique(['s1', 'tid']).join(lookup, on='tid', how='left')
    if joined.filter(pl.col('_target_id').is_null()).select(pl.len()).collect().item():
        raise ValueError('Scored candidate references an unknown target ID')
    grouped = (joined.group_by('s1').agg(pl.col('_target_id').unique().sort().str.join(',').alias('ids'))
               .collect(engine='streaming'))
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as fh:
        fh.write(header)
        for i in range(0, s1_order.height, chunk):
            sub = s1_order.slice(i, chunk).join(grouped, left_on='id', right_on='s1', how='left',
                                               maintain_order='left')
            lines = sub.select((pl.col('entity_id') + pl.lit('\t') + pl.col('ids').fill_null('')).alias('l'))
            fh.write('\n'.join(lines['l'].to_list()) + '\n')
    tmp.replace(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ART / "model"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--chunk-targets", type=int, default=100_000)
    ap.add_argument("--validator", default="student_resource/utils/validate_submission.py")
    ap.add_argument("--nn", default=None, help="neural matcher weights (if the model uses nn_p)")
    ap.add_argument("--skip-validation", action="store_true", help="Explicitly skip the official validator")
    args = ap.parse_args(argv)
    lower_priority()
    from src.er import model as M
    from src.er.gpu_features import GpuFeaturizer

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s rss {rss_gb():.1f}GB] {m}", flush=True)
    from src.er.artifacts import (fingerprint, pipeline_files, runtime_versions, atomic_parquet,
                                  write_manifest, require_current_blocking, write_json)
    if not args.skip_validation and not Path(args.validator).is_file():
        raise FileNotFoundError(f"Validator missing: {args.validator}; use --skip-validation only for experiments")
    require_current_blocking(ART, 'test')
    booster, meta = M.load(args.model_dir)
    block_settings = json.loads((ART / 'test_cands' / 'manifest.json').read_text()).get('settings', {})
    if block_settings != meta.get('blocking_settings', {}):
        raise ValueError('Train/test blocking settings differ. Rebuild both splits with the same settings and retrain.')
    needs_nn = 'nn_p' in booster.feature_names
    if needs_nn and not args.nn:
        raise ValueError('This model requires --nn with the neural checkpoint used during training')
    if args.nn and fingerprint([args.nn]) != meta.get('nn_identity'):
        raise ValueError('Neural checkpoint does not match the checkpoint used to train this model')
    thr = args.threshold if args.threshold is not None else meta["threshold"]
    log(f"model best_iteration={meta['best_iteration']}  threshold={thr}")

    s1 = pl.read_parquet(ART / "test_s1.parquet")
    fz = GpuFeaturizer(s1, pl.read_parquet(ART / "test_s1stats.parquet"), sib_dir=ART / "test_sibs", nn_path=args.nn)
    s1_order = s1.select("id", "entity_id")
    tg_scan = pl.scan_parquet(ART / "test_tg.parquet")
    parts = sorted((ART / "test_cands").glob("part-*.parquet"))

    if not parts:
        raise ValueError('No candidate partitions; run the test blocking step first')
    identity_files = ([Path(args.model_dir) / 'xgb.json', Path(args.model_dir) / 'meta.json',
                       ART / 'test_s1.parquet', ART / 'test_tg.parquet', ART / 'test_s1stats.parquet']
                      + parts + sorted((ART / 'test_sibs').glob('*.parquet')) + pipeline_files())
    if args.nn:
        identity_files.append(Path(args.nn))
    identity = fingerprint(identity_files, runtime_versions())
    scored_dir = ART / 'test_scored' / identity
    scored_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(scored_dir / 'manifest.json', identity, model=str(args.model_dir))
    log(f'score cache: {scored_dir}')
    for i, part_file in enumerate(parts):
        out = scored_dir / f"part-{i:03d}.parquet"
        if out.exists():
            continue  # resumable
        cands = pl.read_parquet(part_file)
        tids = cands["tid"].unique(maintain_order=True).to_frame()
        res = []
        for j in range(0, tids.height, args.chunk_targets):
            wait_for_ram()
            sub = tids.slice(j, args.chunk_targets)
            c = cands.join(sub, on="tid", how="semi")
            tg = tg_scan.join(sub.lazy().rename({"tid": "id"}), on="id", how="semi").collect()
            f = fz.featurize(c, tg)
            if f.height:
                res.append(f.select("tid", "s1").with_columns(pl.Series("p", M.predict(booster, f))))
        result = pl.concat(res) if res else pl.DataFrame(schema={'tid': pl.Int64, 's1': pl.Int64, 'p': pl.Float32})
        atomic_parquet(result, out)
        log(f"scored part {i + 1}/{len(parts)}: {cands.height:,} pairs")
        del cands, res

    scored = pl.scan_parquet([scored_dir / f"part-{i:03d}.parquet" for i in range(len(parts))])
    best = (scored.sort(["p", "s1"], descending=[True, False]).group_by("tid").first()
            .filter(pl.col("p") >= thr).select("tid", "s1"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wait_for_ram()
    write_lists(s1_order, best, "source1_entity_id\tmatched_entity_ids\n", out_dir / "matching_results.tsv", target_ids=tg_scan.select("id", "entity_id"))
    log("wrote matching_results.tsv")
    wait_for_ram()
    write_lists(s1_order, scored.select("tid", "s1"), "source1_entity_id\tcandidate_entity_ids\n",
                out_dir / "candidate_pairs.tsv", target_ids=tg_scan.select("id", "entity_id"))
    log("wrote candidate_pairs.tsv")

    m = pl.read_csv(out_dir / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    nonempty = m["matched_entity_ids"].fill_null("") != ""
    log(f"S1 rows {m.height:,}; with matches {nonempty.sum():,} ({nonempty.mean():.1%}); "
        f"matched ids {m['matched_entity_ids'].fill_null('').str.count_matches(',').sum() + nonempty.sum():,}")

    manifest = {
        'model_dir': str(Path(args.model_dir).resolve()),
        'model_identity': fingerprint([Path(args.model_dir) / 'xgb.json', Path(args.model_dir) / 'meta.json']),
        'score_cache_identity': identity, 'threshold': thr, 'calibrated_threshold': meta['threshold'],
        'threshold_overridden': args.threshold is not None, 'blocking_settings': block_settings,
        'matching_identity': fingerprint([out_dir / 'matching_results.tsv']),
        'candidate_identity': fingerprint([out_dir / 'candidate_pairs.tsv']),
        'source1_rows': m.height, 'validator_status': 'skipped' if args.skip_validation else 'pending',
    }
    write_json(out_dir / 'submission_manifest.json', manifest)
    if args.skip_validation:
        log("Official validation explicitly skipped")
        return
    cmd = [sys.executable, args.validator,
           "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "candidate_pairs.tsv"),
           "--test-dir", str(DATA_DIR / "test"), "--check-ids"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    manifest['validator_status'] = 'passed' if r.returncode == 0 else 'failed'
    manifest['validator_exit_code'] = r.returncode
    write_json(out_dir / 'submission_manifest.json', manifest)
    log(f"validator exit code {r.returncode}")
    if r.returncode:
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
