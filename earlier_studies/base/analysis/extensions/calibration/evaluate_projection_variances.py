"""Evaluate the declared post hoc variance refinement on all saved filter draws."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, chi2

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
ROOT=args.root.resolve()
SOURCE=ROOT/'filter_study'
OUT=args.out.resolve()
OUT.mkdir(exist_ok=False)
cells=json.loads((SOURCE/'cells.json').read_text())
u=(np.array([0,0,1,0,1,0],float),np.array([1,0,1,0,0,0],float))
w=np.array([0,-1,0,1,0,0],float)
alpha,eta=0.05,0.01
rows=[]
all_decisions=[]
for cell in cells:
    raw=SOURCE/cell['raw_file']
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=cell['sha256']:
        raise ValueError('Saved covariance input hash mismatch.')
    n=cell['records_per_condition']
    nu=n-1
    multiplier=nu/chi2.ppf(eta/4,nu)
    t=math.log(2/(alpha-eta))
    decisions=[]
    with np.load(raw) as data:
        for index,name in enumerate(('covariance_A','covariance_B')):
            covariance=data[name]
            s=np.einsum('i,kij,j->k',u[index],covariance,w)
            p=multiplier*np.einsum('i,kij,j->k',u[index],covariance,u[index])
            q=multiplier*np.einsum('i,kij,j->k',w,covariance,w)
            d=multiplier*np.maximum(covariance[:,0,0],covariance[:,1,1])
            b=2*cell['filter_a']*d
            limit=b+np.sqrt(2*(p*q+b*b)*t/nu)+(np.sqrt(p*q)+b)*t/nu
            decisions.append(np.abs(s)>limit)
    reject=decisions[0]&decisions[1]
    all_decisions.append(reject)
    count=int(reject.sum())
    total=len(reject)
    lo=0 if count==0 else float(beta.ppf(.025,count,total-count+1))
    hi=1 if count==total else float(beta.ppf(.975,count+1,total-count))
    rows.append(dict(cell_index=cell['cell_index'],filter_a=cell['filter_a'],
                     records_per_condition=n,model=cell['model'],
                     method='same_sample_projection_variances',rejections=count,
                     replicates=total,fraction=count/total,cp95_lower=lo,cp95_upper=hi,
                     analysis_status='post_hoc_all_saved_cells'))
np.savez_compressed(OUT/'decisions.npz',decisions=np.stack(all_decisions))
with (OUT/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
report=dict(status='PROJECTION_VARIANCE_REANALYSIS_COMPLETE',cells=len(rows),
            decisions=sum(row['replicates'] for row in rows),new_random_draws=0,
            analysis_status='Theory-led post hoc refinement on all saved filter-study draws.',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_cells_sha256=hashlib.sha256((SOURCE/'cells.json').read_bytes()).hexdigest(),
            summary_sha256=hashlib.sha256((OUT/'summary.csv').read_bytes()).hexdigest(),
            decisions_sha256=hashlib.sha256((OUT/'decisions.npz').read_bytes()).hexdigest())
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
for row in rows:
    if row['filter_a']==.15:
        print(row['records_per_condition'],row['model'],row['rejections'])
