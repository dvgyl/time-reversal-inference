"""Reproducible figures from same-draw reanalysis and proved analytic formulas."""
import argparse,csv,json,hashlib
from pathlib import Path
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
def curve(ax,rs,label,color,marker,ls='-'):
    rs=sorted(rs,key=lambda r:int(r['n']));n=[int(r['n']) for r in rs]
    ax.plot(n,[float(r['fraction']) for r in rs],label=label,color=color,marker=marker,ls=ls,ms=3,lw=1.3)
    ax.fill_between(n,[float(r['cp95_lower']) for r in rs],[float(r['cp95_upper']) for r in rs],color=color,alpha=.08)

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);src=Path(__file__).parent/'results/summary.csv';rows=read(src)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None});fig.savefig(a.out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(7.3,2.7),layout='constrained')
    x=np.linspace(0,.7,220);axs[0].plot(x,1-np.sqrt(2)*x,label='Observed 4×4 covariance',color='#296080');axs[0].plot(x,1-2*x,label='Source 2×2 constraint',color='#ac4337');axs[0].axhline(0,color='gray',lw=.7);axs[0].axvline(.5,color='gray',lw=.7,ls=':');axs[0].set(xlabel='Cross-covariance parameter x',ylabel='Smallest eigenvalue',title='a  Positivity boundary');axs[0].legend(fontsize=6.4,loc='lower left')
    methods=[('oracle_t','Oracle direction','#296080','o'),('learned_t','Learned direction','#28794b','s')]
    for method,label,col,m in methods:curve(axs[1],[r for r in rows if r['family']=='positive_parity' and r['parameter']=='3/5' and r['method']==method],label,col,m)
    old=read(a.baseline/'compute/evaluation/summary.csv')
    sdp=[dict(r,n=r['n']) for r in old if r['family']=='positive_parity' and r['a']=='3/5' and r['method']=='general_sdp'];curve(axs[1],sdp,'Semidefinite search','#ac4337','^','--')
    axs[1].set(xlabel='Independent records n',ylabel='Rejection fraction',title='b  Student-t tests, x = 0.6',ylim=(-.02,1.03));axs[1].set_xscale('log',base=2);axs[1].set_xticks([64,256,1024,8192],['64','256','1k','8k']);axs[1].legend(fontsize=6.8,loc='lower right')
    families=['negative_parity','scaled_positive']
    actual=sorted(set(r['family'] for r in rows if r['study']=='two_time'));print('families',actual)
    families=[next(f for f in actual if 'negative' in f),next(f for f in actual if 'scal' in f)]
    for j,(method,label,col,m) in enumerate(methods):
        vals=[next(r for r in rows if r['family']==f and r['parameter']=='3/5' and r['n']=='128' and r['method']==method) for f in families]
        y=[float(r['fraction']) for r in vals];err=np.array([[v-float(r['cp95_lower']) for v,r in zip(y,vals)],[float(r['cp95_upper'])-v for v,r in zip(y,vals)]])
        axs[2].bar(np.arange(2)+(j-.5)*.33,y,.30,color=col,label=label,yerr=err,capsize=2,error_kw={'lw':.8})
    axs[2].set_xticks([0,1],['Sign-reversed','Rescaled']);axs[2].set(title='c  Direction matters, n = 128',ylabel='Rejection fraction',ylim=(0,1.05))
    for ax in axs:style(ax)
    save(fig,'calibrated_inference')
    fig,axs=plt.subplots(1,3,figsize=(7.3,2.6),sharey=True,layout='constrained');old=read(a.baseline/'refinement/study_results/summary.csv')
    for ax,param in zip(axs,['1/10','1/5','2/5']):
        curve(ax,[r for r in rows if r['family']=='AB' and r['parameter']==param], 'Student-t reflection','#28794b','o')
        sr=[dict(r,n=r['n_per_condition']) for r in old if r['family']=='AB' and r['r']==param and r['method']=='general_sdp'];curve(ax,sr,'Semidefinite search','#ac4337','s','--')
        ax.set(xlabel='Records per condition n',title='r = '+str(float(Fraction(param))),ylim=(-.02,1.03));ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024,4096],['64','256','1k','4k']);style(ax)
    axs[0].set_ylabel('Common-model rejection fraction');axs[0].legend(fontsize=7,loc='upper left');save(fig,'connected_inference')
    eta=np.geomspace(.001,1,220);theta=np.arccos(1/(1+eta));low=np.pi/theta-1;upper=2*(np.floor(np.pi/(2*theta))+1);div=8/(eta*(eta+3))
    fig,axs=plt.subplots(1,2,figsize=(7.3,2.9),layout='constrained');axs[0].loglog(eta,div,color='#296080');axs[0].set(xlabel='Regularization η',ylabel='Reversal divergence I₃',title='a  Observed irreversibility')
    axs[1].loglog(eta,low,label='Necessary delay bound',color='#296080');axs[1].loglog(eta,upper,label='Sufficient even delay',color='#ac4337');axs[1].loglog(eta,np.pi/np.sqrt(2*eta),label='Sharp leading term',color='#28794b',ls=':');axs[1].set(xlabel='Regularization η',ylabel='Common delay (samples)',title='b  Sharp asymptotic constant');axs[1].legend(fontsize=7)
    for ax in axs:style(ax)
    save(fig,'delay_resource')
    with (a.out/'delay_resource.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['eta','divergence','necessary_lower','polynomial_upper','leading_term']);w.writerows(zip(eta,div,low,upper,np.pi/np.sqrt(2*eta)))
    k=np.arange(-4,5);fig,axs=plt.subplots(1,3,figsize=(7.3,2.7),sharey=True,layout='constrained')
    for ax,delta,T,title in zip(axs,[0,1,1],[3,2,3],['a  Reversible source','b  One-sample channel delay','c  Longer calibrated window']):
        y=.25*np.exp(-.9*(k-delta)**2);ax.plot(k,y,'o-',color='#296080',ms=3);ax.axvline(delta,color='#ac4337',ls='--',lw=1,label='Symmetry axis');ax.axvspan(-(T-1),T-1,color='#28794b',alpha=.09);ax.set(xlabel='Lag k (samples)',title=title,xticks=[-3,-1,0,1,3]);style(ax)
    axs[0].set_ylabel('Cross-covariance');axs[2].legend(fontsize=7);save(fig,'delay_mechanism')
    (a.out/'manifest.json').write_text(json.dumps({'source_script':sha(Path(__file__)),'same_draw_summary':sha(src),'data_status':'post hoc reanalysis plus analytic illustrations; no primary data','outputs':{p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.pdf','.png','.csv']}},indent=2)+'\n')
    print('REVISION_FIGURES_RENDERED')
if __name__=='__main__':main()
