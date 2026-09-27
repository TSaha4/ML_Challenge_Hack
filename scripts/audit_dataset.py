"""Streaming, label-free train/test profiles plus labelled training population checks."""
import argparse
import json
from pathlib import Path

import polars as pl


def scan(path):
    return pl.scan_csv(path, separator='\t', quote_char=None, infer_schema=False)


def run(data, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    profiles=[]
    for split in ('train', 'test'):
        for source in (1,2,3):
            path=data/split/f'{split}_source{source}.tsv'
            df=scan(path)
            n=pl.col('business_name').fill_null('')
            a=pl.col('business_address').fill_null('')
            rows=(df.group_by('country').agg(
                pl.len().alias('rows'),
                (n.str.strip_chars()=='').sum().alias('missing_name'),
                (a.str.strip_chars()=='').sum().alias('missing_address'),
                n.str.contains(r'[^\x00-\x7F]').sum().alias('nonascii_name'),
                n.str.contains(r'[ऀ-෿]').sum().alias('indic_name'),
                a.str.contains(r'[^\x00-\x7F]').sum().alias('nonascii_address'),
                n.str.contains(r'(?i)\b(dba|fka|aka|formerly|trading as)\b').sum().alias('alias_name'),
                n.str.contains(r'(?i)\.(com|in|fr|net|org)\b').sum().alias('domain_name'),
                n.str.len_chars().mean().alias('mean_name_length'),
                a.str.len_chars().mean().alias('mean_address_length'),
                (n.str.len_bytes()>64).sum().alias('name_over_feature_limit'),
                (a.str.len_bytes()>112).sum().alias('address_over_feature_limit'),
                pl.col('entity_id').str.contains(r'^S[123]-0\d').sum().alias('zero_padded_ids'),
            ).collect(engine='streaming').to_dicts())
            for r in rows: r.update(split=split,source=source)
            profiles.extend(rows)
            sample=df.filter(pl.col('entity_id').hash(seed=17)%1000==0).collect(engine='streaming')
            sample.write_parquet(out/f'{split}_source{source}_sample.parquet')
            print(split,source,rows,flush=True)
            (out/'profiles.json').write_text(json.dumps(profiles,indent=2))
    gt=scan(data/'train/train_ground_truth.tsv').with_columns(
        pl.col('matched_entity_ids').fill_null('').alias('m'),
        pl.col('source1_entity_id').str.slice(3).cast(pl.Int64).alias('id'))
    gt=gt.with_columns(pl.when(pl.col('m')=='').then(0).otherwise(pl.col('m').str.count_matches(',')+1).alias('n_matches'))
    summary=gt.select(pl.len().alias('entities'),(pl.col('n_matches')==0).sum().alias('singletons'),
        pl.col('n_matches').sum().alias('true_pairs'),pl.col('n_matches').max().alias('max_matches'),
        pl.col('n_matches').mean().alias('mean_matches')).collect(engine='streaming').to_dicts()[0]
    folds=gt.with_columns((pl.col('id')%100).alias('fold')).group_by('fold').agg(
        pl.len().alias('entities'),(pl.col('n_matches')==0).mean().alias('singleton_rate'),
        pl.col('n_matches').mean().alias('mean_matches')).collect(engine='streaming').sort('fold').to_dicts()
    (out/'ground_truth_profile.json').write_text(json.dumps({'summary':summary,'folds':folds},indent=2))
    print('ground truth',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='student_resource/dataset')
    p.add_argument('--out-dir',default='artifacts/data_audit')
    args=p.parse_args()
    run(args.data_dir,args.out_dir)
