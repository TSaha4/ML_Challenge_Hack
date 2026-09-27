"""Disk-backed submission membership/ownership and country-distribution audit."""
import argparse
import json
from pathlib import Path

import duckdb


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='student_resource/dataset')
    p.add_argument('--matching',default='matching_results.tsv')
    p.add_argument('--out-dir',default='artifacts/data_audit')
    args=p.parse_args();out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    c=duckdb.connect(str(out/'submission.duckdb'))
    c.execute("SET memory_limit='256MB'");c.execute('SET threads=2')
    c.execute("SET preserve_insertion_order=false")
    read="read_csv(?,delim='\\t',header=true,quote='',all_varchar=true)"
    c.execute('CREATE OR REPLACE TABLE submission AS SELECT source1_entity_id,coalesce(matched_entity_ids,\'\') matched_entity_ids FROM '+read,[args.matching])
    c.execute('CREATE OR REPLACE TABLE refs AS SELECT entity_id,country FROM '+read,[str(Path(args.data_dir)/'test/test_source1.tsv')])
    c.execute("CREATE OR REPLACE TABLE pairs AS SELECT source1_entity_id s1,unnest(string_split(matched_entity_ids,',')) tid FROM submission WHERE matched_entity_ids<>''")
    c.execute('CREATE OR REPLACE TABLE target_ids AS SELECT entity_id,country FROM '+read,[str(Path(args.data_dir)/'test/test_source2.tsv')])
    c.execute('INSERT INTO target_ids SELECT entity_id,country FROM '+read,[str(Path(args.data_dir)/'test/test_source3.tsv')])
    def rows(sql):
        cur=c.execute(sql);cols=[d[0] for d in cur.description];return [dict(zip(cols,row)) for row in cur.fetchall()]
    report={}
    for label,sql in {
        'submission_rows':'SELECT count(*) n FROM submission',
        'missing_s1':'SELECT count(*) n FROM refs r ANTI JOIN submission s ON r.entity_id=s.source1_entity_id',
        'unknown_s1':'SELECT count(*) n FROM submission s ANTI JOIN refs r ON r.entity_id=s.source1_entity_id',
        'duplicate_s1':'SELECT count(*) n FROM (SELECT source1_entity_id FROM submission GROUP BY 1 HAVING count(*)>1)',
        'pairs':'SELECT count(*) n FROM pairs',
        'unknown_targets':'SELECT count(*) n FROM pairs p ANTI JOIN target_ids t ON p.tid=t.entity_id',
        'duplicate_pairs':'SELECT count(*) n FROM (SELECT s1,tid FROM pairs GROUP BY 1,2 HAVING count(*)>1)',
        'targets_assigned_to_multiple_entities':'SELECT count(*) n FROM (SELECT tid FROM pairs GROUP BY 1 HAVING count(DISTINCT s1)>1)',
        'cross_country_pairs':'SELECT count(*) n FROM pairs p JOIN refs r ON p.s1=r.entity_id JOIN target_ids t ON p.tid=t.entity_id WHERE r.country<>t.country',
    }.items():
        report[label]=rows(sql)[0]['n'];print(label,report[label],flush=True)
    report['by_country']=rows("""SELECT country,count(*) entities,
      count(*) FILTER(WHERE matched_entity_ids='') predicted_singletons,
      avg(CASE WHEN matched_entity_ids='' THEN 0 ELSE len(string_split(matched_entity_ids,',')) END) mean_matches,
      sum(CASE WHEN matched_entity_ids='' THEN 0 ELSE len(string_split(matched_entity_ids,',')) END) matches
      FROM submission s JOIN refs r ON s.source1_entity_id=r.entity_id GROUP BY country""")
    c.execute('CREATE OR REPLACE TABLE train_refs AS SELECT entity_id,country FROM '+read,[str(Path(args.data_dir)/'train/train_source1.tsv')])
    c.execute('CREATE OR REPLACE TABLE truth AS SELECT * FROM '+read,[str(Path(args.data_dir)/'train/train_ground_truth.tsv')])
    report['training_by_country']=rows("""SELECT country,count(*) entities,
       count(*) FILTER(WHERE coalesce(matched_entity_ids,'')='') true_singletons,
       avg(CASE WHEN coalesce(matched_entity_ids,'')='' THEN 0 ELSE len(string_split(matched_entity_ids,',')) END) mean_matches
       FROM truth t JOIN train_refs r ON t.source1_entity_id=r.entity_id GROUP BY country""")
    report['train_test_reference_id_overlap']=rows('SELECT count(*) n FROM refs JOIN train_refs USING(entity_id)')[0]['n']
    (out/'submission_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    c.close()

if __name__=='__main__':main()
