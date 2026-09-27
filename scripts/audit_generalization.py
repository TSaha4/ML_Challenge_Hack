"""CPU screening experiment on sampled real entities; not a leaderboard estimate.

Includes every labelled target for sampled entities plus random distractors. Uses
production normalization, blocking and features, but no neural matcher or learned
transliteration. All limits and omissions are recorded in the report.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import xgboost as xgb

from src.er.text import normalize_frame
from src.er.io import encode_s1, encode_target
from src.er.gpu_blocking import GpuIndex
from src.er.gpu_features import GpuFeaturizer, FEATURES
from src.er.features import s1_stats
from src.er.siblings import build_sibling_tables
from src.er import model as M
from src.er.diagnostics import entity_errors


def scan(path):
    return pl.scan_csv(path,separator='\t',quote_char=None,infer_schema=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='student_resource/dataset')
    p.add_argument('--out-dir',default='artifacts/data_audit/screening')
    p.add_argument('--sample-mod',type=int,default=400)
    args=p.parse_args()
    root=Path(args.data_dir)/'train';out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    start=time.time()
    def log(msg): print(f'[{time.time()-start:.1f}s] {msg}',flush=True)
    raw=scan(root/'train_source1.tsv').filter(pl.col('entity_id').hash(seed=23)%args.sample_mod==0).collect(engine='streaming')
    raw=raw.with_columns(encode_s1().alias('id'))
    gt=scan(root/'train_ground_truth.tsv').filter(pl.col('source1_entity_id').is_in(raw['entity_id'].implode())).collect(engine='streaming')
    gt=(gt.select(encode_s1('source1_entity_id').alias('s1'),pl.col('matched_entity_ids').str.split(',').alias('tid'))
        .explode('tid').filter(pl.col('tid').is_not_null()&(pl.col('tid')!='')))
    target_ids=gt['tid']
    truth=gt.with_columns(encode_target('tid').alias('tid'))
    parts=[]
    for source in (2,3):
        part=(scan(root/f'train_source{source}.tsv').filter(pl.col('entity_id').is_in(target_ids.implode())|
              (pl.col('entity_id').hash(seed=29)%1000==0)).collect(engine='streaming')
              .with_columns(encode_target().alias('id')))
        parts.append(part)
    targets=pl.concat(parts)
    s1=normalize_frame(raw)
    tg=normalize_frame(targets).with_columns((pl.col('id')//10000000000).cast(pl.Int8).alias('src'))
    s1.write_parquet(out/'s1.parquet');tg.write_parquet(out/'tg.parquet');truth.write_parquet(out/'truth.parquet')
    log(f'sampled S1={s1.height}, targets={tg.height}, true pairs={truth.height}')
    idx=GpuIndex(s1,cap=100,device='cpu')
    cands=idx.candidates(tg,k=10,k_wide=80,k_char=6,slice_targets=1000,max_pairs=1000000)
    del idx
    cands.write_parquet(out/'candidates.parquet')
    build_sibling_tables(str(out/'candidates.parquet'),str(out/'tg.parquet'),str(out/'s1.parquet'),out/'sibs')
    fz=GpuFeaturizer(s1,s1_stats(cands),sib_dir=out/'sibs')
    features=[]
    for start_i in range(0,tg.height,1000):
        sub=tg.slice(start_i,1000)
        c=cands.filter(pl.col('tid').is_in(sub['id'].implode()))
        if c.height: features.append(fz.featurize(c,sub,gpu_chunk=2000))
    f=pl.concat(features).join(truth.with_columns(pl.lit(1).alias('y')),on=['s1','tid'],how='left').with_columns(pl.col('y').fill_null(0))
    f.write_parquet(out/'features.parquet')
    del fz,features
    folds=s1.select(pl.col('id').alias('s1'),(pl.col('id').hash(seed=41)%20).alias('fold'))
    cal=folds.filter(pl.col('fold')==0)['s1'];hold=folds.filter(pl.col('fold')==1)['s1']
    held=pl.concat([cal,hold])
    reserved=pl.concat([cands.filter(pl.col('s1').is_in(held.implode())).select('tid'),
                       truth.filter(pl.col('s1').is_in(held.implode())).select('tid')]).unique()['tid']
    fitting=f.filter(~pl.col('tid').is_in(reserved.implode()))
    es_tids=fitting.select('tid').unique().filter(pl.col('tid').hash(seed=17)%5==0)['tid']
    es=fitting.filter(pl.col('tid').is_in(es_tids.implode()))
    fit=fitting.filter(~pl.col('tid').is_in(es_tids.implode()))
    log(f'pairs={f.height}; fit={fit.height}, early-stop={es.height}; calibration entities={cal.len()}, holdout={hold.len()}')
    report={'scope':'CPU screening, sampled candidate universe; not production/leaderboard accuracy',
            'omissions':['neural matcher','learned transliteration','full reference universe','most unmatched target distractors'],
            'sampled_s1':s1.height,'targets':tg.height,'true_pairs':truth.height,'candidate_pairs':cands.height,
            'calibration_entities':cal.len(),'holdout_entities':hold.len(),'experiments':[]}
    def matrix(frame): return xgb.DMatrix(frame.select(FEATURES).to_numpy().astype(np.float32),label=frame['y'].to_numpy(),feature_names=FEATURES)
    allmat=matrix(f);esmat=matrix(es)
    configs=[('regularized_25pct',.25,4,20,10.),('regularized_full',1.,4,20,10.),('current_depth_full',1.,9,5,1.)]
    for name,fraction,depth,child,reg in configs:
        sub=fit if fraction==1. else fit.filter(pl.col('tid').hash(seed=31)%4==0)
        history={}
        booster=xgb.train({**M.XGB_PARAMS,'device':'cpu','nthread':2,'max_depth':depth,
            'min_child_weight':child,'lambda':reg,'seed':42,'eval_metric':'logloss'},matrix(sub),500,
            evals=[(matrix(sub),'train'),(esmat,'early_stop')],early_stopping_rounds=40,evals_result=history,verbose_eval=False)
        probabilities=booster.predict(allmat,iteration_range=(0,booster.best_iteration+1))
        scored=f.select('tid','s1').with_columns(pl.Series('p',probabilities))
        threshold,curve=M.tune_threshold(scored,truth,cal)
        pred=M.assign(scored,threshold)
        oracle=cands.join(truth,on=['s1','tid'],how='semi').with_columns(pl.lit(1.).alias('p'))
        truth_counts=truth.group_by('s1').len()
        row={'name':name,'fit_rows':sub.height,'best_iteration':booster.best_iteration,'threshold':threshold,
             'train_logloss':history['train']['logloss'][booster.best_iteration],
             'early_stop_logloss':history['early_stop']['logloss'][booster.best_iteration],
             'calibration':M.macro_f05(pred,truth,cal),'holdout':M.macro_f05(pred,truth,hold),
             'holdout_blocking_ceiling':M.macro_f05(M.assign(oracle,.5),truth,hold)}
        row['country_holdout']={}
        for (country,),group in s1.filter(pl.col('id').is_in(hold.implode())).partition_by('cty',as_dict=True).items():
            row['country_holdout'][country]=M.macro_f05(pred,truth,group['id'])
        row['positive_candidate_recall']=oracle.height/truth.height
        report['experiments'].append(row)
        scored.write_parquet(out/f'{name}_scored.parquet')
        booster.save_model(out/f'{name}.json')
        entity_errors(pred,scored,truth,hold).write_parquet(out/f'{name}_errors.parquet')
        (out/'report.json').write_text(json.dumps(report,indent=2))
        log(row)
    log('done')

if __name__=='__main__': main()
