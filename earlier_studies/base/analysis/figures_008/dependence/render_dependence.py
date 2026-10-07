"""Plot exact false rejection and the stated variance-coverage cost."""
from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
alpha=.05
eta=.01
windows=np.unique((np.geomspace(4,10000,240)//4*4).astype(int))
t=math.log(2/alpha)
iid_radius=(np.sqrt(8*(windows-1)*t)+2*t)/windows
false=np.exp(-iid_radius)
valid_radius=2*math.sqrt(t)+t
length=np.unique(np.geomspace(10,10000,300).astype(int))
rows=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(9,4.8))
ax[0].plot(windows,false,color='#A34337',label='Independent-window calibration')
ax[0].axhline(alpha,color='#454545',ls='--',label='Exact covariance-specific cutoff')
ax[0].axhline(math.exp(-valid_radius),color='#315F8B',label='General trajectory bound')
ax[0].set(xscale='log',xlabel='Windows in one recording',ylabel='Actual false-rejection probability',yscale='log',ylim=(.0001,1.05),title='a   Harmonic reversible null')
ax[0].legend(frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(0,-.23),borderaxespad=0)
for n,p in zip(windows,false):rows.append({'panel':'false_rejection','count':int(n),'rho':'','value':float(p),'status':'finite'})
for rho,color in [(1,'#454545'),(5,'#315F8B'),(20,'#147D72')]:
    denominator=1-rho/length-2*np.sqrt(rho*math.log(2/eta)/length)
    mask=denominator>0
    inflation=np.full(len(length),np.nan)
    inflation[mask]=1/denominator[mask]
    ax[1].plot(length,inflation,color=color,label=f'Relative bound rho = {rho}')
    for n,c,v in zip(length,denominator,inflation):
        rows.append({'panel':'variance_coverage','count':int(n),'rho':rho,'value':float(v) if c>0 else '', 'status':'finite' if c>0 else 'abstain'})
ax[1].set(xscale='log',xlabel='Observations per channel',ylabel='Variance upper-bound multiplier',ylim=(1,8),title='b   Same-record variance calibration')
ax[1].legend(frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(0,-.23),borderaxespad=0)
for a in ax:a.grid(axis='y',color='#DDDDDD',lw=.5)
fig.tight_layout()
fig.savefig(ROOT/'dependent_calibration.pdf',metadata={'CreationDate':None,'ModDate':None})
fig.savefig(ROOT/'dependent_calibration.png',dpi=180)
with (ROOT/'dependent_calibration.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
report={'kind':'analytic evaluations; no random draws','nominal_alpha':alpha,'variance_eta':eta,'channels':2,'harmonic_counts_divisible_by_four':True,'variance_plot_y_limit':8,'rows':len(rows),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'dependent_calibration.csv',ROOT/'dependent_calibration.png',ROOT/'dependent_calibration.pdf']}}
(ROOT/'FIGURE_BINDING.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'new_random_draws':0}))
