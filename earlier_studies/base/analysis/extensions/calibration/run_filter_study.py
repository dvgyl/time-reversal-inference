"""Run the fixed filter study using centered Gaussian scatter matrices."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta, chi2, t, wishart

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'filter_study'
OUT.mkdir(exist_ok=False)
r = 0.4
replicates = 5000
alpha = 0.05
eta = 0.01
methods = ('student_ignores_filter', 'known_variance_filter', 'same_sample_variance_filter')
models = {'null_common_zero': (0,0), 'null_common_one': (1,1), 'different_delays': (0,1)}
u = (np.array([0,0,1,0,1,0], dtype=float), np.array([1,0,1,0,0,0], dtype=float))
w = np.array([0,-1,0,1,0,0], dtype=float)
rows, cells = [], []
cell_index = 0
frequency = np.linspace(-np.pi, np.pi, 20001)
for a in (0.0,0.05,0.15,0.25):
    transfer = 1+a*np.exp(-1j*frequency)
    phase = np.abs(transfer.imag)/np.abs(transfer)
    if phase.max() > a+2e-15:
        raise ArithmeticError('The fixed phase bound fails.')
    for records in (128,512,2048,8192,32768):
        nu = records-1
        variance_quantile = chi2.ppf(eta/2, nu)
        covariance_tolerance = 2*a
        known_t = math.log(2/alpha)
        adaptive_t = math.log(2/(alpha-eta))
        known_limit = covariance_tolerance + math.sqrt(2*(16+covariance_tolerance**2)*known_t/nu)
        known_limit += (4+covariance_tolerance)*known_t/nu
        adaptive_limit = covariance_tolerance + math.sqrt(2*(16+covariance_tolerance**2)*adaptive_t/nu)
        adaptive_limit += (4+covariance_tolerance)*adaptive_t/nu
        student_limit = t.ppf(1-alpha/2, records-2)
        for model, delays in models.items():
            candidate_decisions=[]
            covariance_samples=[]
            population=[]
            for condition, delay in enumerate(delays):
                gamma=np.empty((6,6))
                for i in range(6):
                    for j in range(6):
                        lag=i//2-j//2
                        ci,cj=i%2,j%2
                        if ci==cj==0:
                            gamma[i,j]=float(lag==0)+a/(1+a*a)*float(abs(lag)==1)
                        elif ci==cj==1:
                            gamma[i,j]=float(lag==0)
                        else:
                            k=lag if ci==0 else -lag
                            gamma[i,j]=r/math.sqrt(1+a*a)*(float(k==delay)+a*float(k==delay+1))
                innovation_times=range(-delay-1,3)
                transform=np.zeros((6,2*len(innovation_times)))
                for time in range(3):
                    for lag,tap in ((delay,1/math.sqrt(1+a*a)),(delay+1,a/math.sqrt(1+a*a))):
                        ix=2*(time-lag-innovation_times.start)
                        transform[2*time,ix]+=tap*math.sqrt(1-r*r)
                        transform[2*time,ix+1]+=tap*r
                    ix=2*(time-innovation_times.start)
                    transform[2*time+1,ix+1]=1
                if not np.allclose(gamma,transform@transform.T,atol=2e-15,rtol=0):
                    raise ArithmeticError('The filter and covariance constructions disagree.')
                minimum_eigenvalue=float(np.linalg.eigvalsh(gamma)[0])
                if minimum_eigenvalue<=0:
                    raise ArithmeticError('The record covariance is not positive definite.')
                contrast=u[condition]
                kappa=float(contrast@gamma@w)
                if condition==0:
                    expected=r/math.sqrt(1+a*a)*(1 if delay==0 else -a)
                else:
                    expected=-r/math.sqrt(1+a*a)*(a if delay==0 else 1)
                if abs(kappa-expected)>1e-14:
                    raise ArithmeticError('The analytic reflection contrast does not match.')
                seed=[20261003,7001,cell_index,condition]
                rng=np.random.default_rng(np.random.SeedSequence(seed))
                scatters=wishart.rvs(df=nu,scale=gamma/nu,size=replicates,random_state=rng)
                covariance_samples.append(scatters)
                s=np.einsum('i,kij,j->k',contrast,scatters,w)
                p=np.einsum('i,kij,j->k',contrast,scatters,contrast)
                q=np.einsum('i,kij,j->k',w,scatters,w)
                residual=p*q-s*s
                if (residual<=0).any():
                    raise ArithmeticError('A sampled projected covariance is degenerate.')
                statistic=np.abs(s)*np.sqrt((records-2)/residual)
                ceiling=nu*np.maximum(scatters[:,0,0],scatters[:,1,1])/variance_quantile
                candidate_decisions.append(np.stack((statistic>student_limit,
                                                     np.abs(s)>known_limit,
                                                     np.abs(s)>ceiling*adaptive_limit),axis=1))
                population.append(dict(condition=condition,delay=delay,seed=seed,
                                       covariance=gamma.tolist(),contrast_covariance=kappa,
                                       minimum_eigenvalue=minimum_eigenvalue))
            decisions=candidate_decisions[0]&candidate_decisions[1]
            filename=f'cell_{cell_index:03d}.npz'
            np.savez_compressed(OUT/filename,covariance_A=covariance_samples[0],
                                covariance_B=covariance_samples[1],decisions=decisions)
            cells.append(dict(cell_index=cell_index,filter_a=a,records_per_condition=records,
                              model=model,replicates=replicates,population=population,
                              raw_file=filename,sha256=hashlib.sha256((OUT/filename).read_bytes()).hexdigest()))
            for column,method in enumerate(methods):
                count=int(decisions[:,column].sum())
                lo=0.0 if count==0 else float(beta.ppf(0.025,count,replicates-count+1))
                hi=1.0 if count==replicates else float(beta.ppf(0.975,count+1,replicates-count))
                rows.append(dict(cell_index=cell_index,filter_a=a,records_per_condition=records,
                                 model=model,method=method,rejections=count,replicates=replicates,
                                 fraction=count/replicates,cp95_lower=lo,cp95_upper=hi))
            cell_index+=1
with (OUT/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT/'cells.json').write_text(json.dumps(cells,indent=2)+'\n')
report=dict(status='FILTER_STUDY_COMPLETE',cells=len(cells),summary_rows=len(rows),
            monte_carlo_pairs=sum(c['replicates'] for c in cells),
            covariance_draws=2*sum(c['replicates'] for c in cells),
            methods=list(methods),scipy=scipy.__version__,numpy=np.__version__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            summary_sha256=hashlib.sha256((OUT/'summary.csv').read_bytes()).hexdigest(),
            cells_sha256=hashlib.sha256((OUT/'cells.json').read_bytes()).hexdigest(),
            scope='New synthetic FIR-filter study. No empirical observations. Same-sample scale estimates.',
            limitation='Monte Carlo performance at the fixed grid does not prove uniform validity.')
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
for row in rows:
    if row['records_per_condition']==32768:
        print(row['filter_a'],row['model'],row['method'],row['rejections'],flush=True)
