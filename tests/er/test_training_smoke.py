"""Tiny real-feature/XGBoost CPU run; no challenge accuracy is measured here."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl


class TrainingSmoke(unittest.TestCase):
    def test_train_reports_independent_holdout_in_fresh_directory(self):
        import xgboost as xgb
        from src.er import io, model
        from src.er.steps import train
        from src.er.text import normalize_frame
        from src.er.features import s1_stats
        from src.er.siblings import build_sibling_tables
        from src.er.artifacts import fingerprint, write_manifest, FEATURE_VERSION
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root/'data'
            (data/'train').mkdir(parents=True)
            ids = [100, 104] + [i for i in range(201, 225) if i % 100 not in (0, 4)]
            raw = pl.DataFrame({'id':ids, 'entity_id':[f'S1-{i}' for i in ids],
                'business_name':[f'company {i} alpha' for i in ids],
                'business_address':[f'{i} main street' for i in ids], 'country':['US']*len(ids)})
            s1 = normalize_frame(raw)
            tids = [20000000000+i for i in ids]
            tg = normalize_frame(raw.with_columns(pl.Series('id', tids),
                pl.Series('entity_id',[f'S2-{i}' for i in ids]))).with_columns(pl.lit(2,pl.Int8).alias('src'))
            s1.write_parquet(root/'train_s1.parquet')
            tg.write_parquet(root/'train_tg.parquet')
            # Calibration/holdout neighborhoods are separate from fit/early stopping.
            pairs=[]
            for j, (sid,tid) in enumerate(zip(ids,tids)):
                choices = [sid] if j < 2 else [sid, ids[2+(j-1)%(len(ids)-2)]]
                for rank, choice in enumerate(choices):
                    pairs.append({'tid':tid,'s1':choice,'sn':float(2-rank),'sa':1.,'nk':2,'rk':rank,'qs':1.-rank*.1,'rq':rank})
            c = pl.DataFrame(pairs)
            (root/'train_cands').mkdir()
            c.write_parquet(root/'train_cands/part-000.parquet')
            s1_stats(c).write_parquet(root/'train_s1stats.parquet')
            build_sibling_tables(str(root/'train_cands/part-*.parquet'),str(root/'train_tg.parquet'),str(root/'train_s1.parquet'),root/'train_sibs')
            write_manifest(root/'train_cands/manifest.json',fingerprint([root/'train_s1.parquet',root/'train_tg.parquet']),feature_version=FEATURE_VERSION)
            pl.DataFrame({'source1_entity_id':[f'S1-{i}' for i in ids],
                          'matched_entity_ids':[f'S2-{i}' for i in ids]}).write_csv(data/'train/train_ground_truth.tsv',separator='\t')
            original_train = xgb.train
            def small_train(params, dtrain, rounds, **kwargs):
                return original_train({**params,'device':'cpu','nthread':1},dtrain,8,**kwargs)
            with patch.object(train,'ART',root), patch.object(io,'DATA_DIR',data), \
                 patch.object(train,'wait_for_ram'), patch.object(train,'lower_priority'), patch.object(xgb,'train',small_train):
                args=['--n-train','12','--n-es','4','--chunk-targets','4','--model-dir',str(root/'model')]
                train.main(args)
                train.main(args+['--reuse-features'])
            meta=json.loads((root/'model/meta.json').read_text())
            self.assertEqual(meta['holdout']['n_s1'],1)
            self.assertEqual(meta['calibration']['n_s1'],1)
            self.assertEqual(meta['feature_version'],FEATURE_VERSION)
            self.assertTrue((root/'model/threshold_curve.json').exists())
            self.assertTrue((root/'model/learning_curves.json').exists())
            self.assertIn('train_logloss', meta['fit_diagnostics'])
            self.assertEqual(meta['parameters']['max_depth'], 9)
            self.assertTrue((root/'model/holdout_errors.parquet').exists())
            self.assertEqual(meta['features'],model.FEATURES)


if __name__ == '__main__': unittest.main()
