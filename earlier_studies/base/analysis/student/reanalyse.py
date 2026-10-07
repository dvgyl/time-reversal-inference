"""Post hoc Student-t calibration on saved draws; never regenerates observations."""
import argparse
import csv
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import math
import scipy
from scipy.stats import t, beta


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pvalue(stat, n, two_sided=False):
    if stat is None or n < 3:
        return 1.0
    a, b, c = (Fraction(stat[k]) for k in ('variance_u', 'variance_v', 'covariance'))
    if a <= 0 or b <= 0:
        return 1.0
    r2 = c*c/(a*b)
    if not 0 <= r2 <= 1:
        raise ValueError('invalid correlation')
    z = math.copysign(math.inf if r2 == 1 else math.sqrt(float((n-2)*r2/(1-r2))), c)
    return float(2*t.sf(abs(z),n-2) if two_sided else t.cdf(z,n-2))


def interval(k,n):
    return [0 if k == 0 else float(beta.ppf(.025,k,n-k+1)),
            1 if k == n else float(beta.ppf(.975,k+1,n-k))]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    groups=defaultdict(list);rows=[];inputs={};closest=1.0
    for study,folder in [('two_time','compute/evaluation'),('three_time','refinement/study_results')]:
        source=args.baseline/folder
        manifest=json.loads((source/'case_hashes.json').read_text())
        inputs[str(source/'case_hashes.json')]=sha(source/'case_hashes.json')
        if set(manifest)!=set(p.name for p in (source/'cases').glob('*.json')):raise ValueError('case inventory mismatch')
        for name,h in sorted(manifest.items()):
            f=source/'cases'/name
            if sha(f)!=h:raise ValueError('case hash mismatch')
            d=json.loads(f.read_text());case=d['case'];n=case['n'];param=case.get('a',case.get('r'))
            methods={}
            if study=='two_time':
                for method,field,N in [('oracle_t','fixed_analytic',n),('learned_t','learned_analytic',n-n//2)]:
                    cert=d[field]['certificates'][1]
                    pv=pvalue(cert['exact'] if cert else None,N)
                    old=bool(cert and Fraction(cert['exact']['covariance'])<0 and Fraction(cert['exact']['r_squared'])>=Fraction(cert['threshold_squared']))
                    methods[method]=(pv,old)
            else:
                ps=[pvalue(c['statistic'],n,True) for c in d['direct_certificates']]
                methods['reflection_t']=(max(ps),d['decisions']['direct_pearson'])
            for method,(pv,old) in methods.items():
                rejected=pv<=.05
                if old and not rejected:raise ValueError('lost conservative rejection')
                closest=min(closest,abs(pv-.05))
                row=dict(study=study,case=name,sha256=h,family=case['family'],parameter=param,n=n,is_null=case['is_null'],method=method,pvalue=pv,rejected=rejected,conservative_rejected=old)
                rows.append(row);groups[study,case['family'],param,n,case['is_null'],method].append(row)
    summary=[]
    for key,rs in sorted(groups.items()):
        study,family,param,n,null,method=key;k=sum(x['rejected'] for x in rs);old=sum(x['conservative_rejected'] for x in rs);lo,hi=interval(k,len(rs))
        summary.append(dict(study=study,family=family,parameter=param,n=n,is_null=null,method=method,rejections=k,sets=len(rs),fraction=k/len(rs),cp95_lower=lo,cp95_upper=hi,conservative_rejections=old,added=k-old))
    write_csv(args.out/'decisions.csv',rows);write_csv(args.out/'summary.csv',summary)
    report=dict(status='STUDENT_T_REANALYSIS_PASS',post_hoc=True,new_draws=0,source_manifests=inputs,python=platform.python_version(),scipy=scipy.__version__,decisions=len(rows),summary_rows=len(summary),minimum_pvalue_distance_from_alpha=closest,source_script_sha256=sha(Path(__file__)),totals={m:dict(rejections=sum(x['rejected'] for x in rows if x['method']==m),null_rejections=sum(x['rejected'] for x in rows if x['method']==m and x['is_null']),sets=sum(x['method']==m for x in rows)) for m in ['oracle_t','learned_t','reflection_t']},limitations='Theoretical finite-sample calibration under ideal independent Gaussian records; SciPy numerical tails on stored finite-precision draws. Not interval-certified probabilities or physical evidence.')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
