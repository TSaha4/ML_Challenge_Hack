"""Unseen-country stress tests and bootstrap uncertainty on the screening fixture."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from src.er.gpu_features import FEATURES
from src.er import model as M
from src.er.diagnostics import entity_errors


def main():
    root=Path('artifacts/data_audit/screening')
    f=pl.read_parquet(root/'features.parquet');s1=pl.read_parquet(root/'s1.parquet');truth=pl.read_parquet(root/'truth.parquet')
    folds=s1.select(pl.col('id').alias('s1'),(pl.col('id').hash(seed=41)%20).alias('fold'))
    cal=folds.filter(pl.col('fold')==0)['s1'];hold=folds.filter(pl.col('fold')==1)['s1']
    held=pl.concat([cal,hold])
    reserved=pl.concat([f.filter(pl.col('s1').is_in(held.implode())).select('tid'),truth.filter(pl.col('s1').is_in(held.implode())).select('tid')]).unique()['tid']
    fitting=f.filter(~pl.col('tid').is_in(reserved.implode()))
    esids=fitting.select('tid').unique().filter(pl.col('tid').hash(seed=17)%5==0)['tid']
    es=fitting.filter(pl.col('tid').is_in(esids.implode()));fit=fitting.filter(~pl.col('tid').is_in(esids.implode()))
    def mat(frame):return xgb.DMatrix(frame.select(FEATURES).to_numpy().astype(np.float32),label=frame['y'].to_numpy(),feature_names=FEATURES)
    allmat=mat(f);results=[]
    for country in ['us','india']:
        ids=s1.filter(pl.col('cty')==country)['id']
        tr=fit.filter(pl.col('s1').is_in(ids.implode()));ev=es.filter(pl.col('s1').is_in(ids.implode()))
        calibrate=cal.filter(cal.is_in(ids.implode()))
        booster=xgb.train({**M.XGB_PARAMS,'device':'cpu','nthread':2,'seed':42,'eval_metric':'logloss'},mat(tr),500,
            evals=[(mat(ev),'es')],early_stopping_rounds=40,verbose_eval=False)
        scored=f.select('tid','s1').with_columns(pl.Series('p',booster.predict(allmat,iteration_range=(0,booster.best_iteration+1))))
        threshold,_=M.tune_threshold(scored,truth,calibrate)
        pred=M.assign(scored,threshold)
        row={'trained_and_calibrated_on':country,'fit_pairs':tr.height,'threshold':threshold,'holdout_by_country':{}}
        for (c,),group in s1.filter(pl.col('id').is_in(hold.implode())).partition_by('cty',as_dict=True).items():
            row['holdout_by_country'][c]=M.macro_f05(pred,truth,group['id'])
        results.append(row);print(row,flush=True)
    rng=np.random.default_rng(5);scores={};cis={}
    for name in ['regularized_25pct','regularized_full','current_depth_full']:
        a=pl.read_parquet(root/f'{name}_errors.parquet').sort('s1')['f05'].to_numpy()
        scores[name]=a
        boot=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
        cis[name]={'mean':float(a.mean()),'bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist()}
    diff=scores['current_depth_full']-scores['regularized_full']
    boot=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1)
    cis['deep_minus_regularized']={'mean':float(diff.mean()),'paired_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist()}
    report={'country_transfer':results,'uncertainty':cis,'scope':'Screening sample only. France has no labels; transfer to France is not measured.'}
    (root/'transfer_report.json').write_text(json.dumps(report,indent=2));print(cis,flush=True)

if __name__=='__main__':main()
