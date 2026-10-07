"""Draw analytical covariances and saved decision counts. Generate no records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle

COLORS = ['#D55E00', '#0072B2', '#CCB974']
OUTCOMES = ['Rejection', 'Nonrejection', 'Abstention']
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})

def save(fig, output, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(output/(name+'.'+ext), dpi=220, bbox_inches='tight')
    plt.close(fig)

def box(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',fc='white',ec='#444444',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7)

def arrow(ax,a,b,color='#444444'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.1))

def exact_cov(k, temperature=1):
    S=np.array([[1+(1+temperature)/6,-(1+temperature)/3],[-(1+temperature)/3,temperature+(1+temperature)/6]])
    P=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    E=P@np.diag(np.exp(-np.array([1.5,.5])*abs(k)))@P.T
    return E@S if k>=0 else S@E

def equilibrium(output):
    fig, axes=plt.subplots(1,3,figsize=(7,2.8),gridspec_kw={'width_ratios':[1.25,1,1]})
    ax=axes[0];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(0,1.02,'(a) Equilibrium source\nand sensors',fontweight='bold',fontsize=9)
    for width in [.23,.36,.48]:ax.add_patch(Ellipse((.25,.53),width,width*.65,angle=30,fill=False,ec='#777777',lw=.7))
    ax.plot(.25,.53,'o',color='#0072B2',ms=4)
    ax.text(.25,.18,r'$T_1=T_2=1$',ha='center')
    ax.text(.25,.08,'Coupled harmonic potential',ha='center',fontsize=7)
    arrow(ax,(.44,.65),(.62,.75));arrow(ax,(.44,.42),(.62,.35))
    box(ax,.63,.68,.33,.19,r'$Y_1=X_1$')
    box(ax,.53,.23,.44,.25,'$Y_2(t)=X_2(t)$\n$+0.03X_2(t-1)$')
    ax.text(.8,.94,'Immediate',ha='center',fontsize=7)
    ax.text(.8,.14,'Small echo',ha='center',fontsize=7)
    k=np.arange(-8,9);matched=np.array([exact_cov(int(i))[0,1] for i in k]);echo=matched+.03*np.array([exact_cov(int(i+1))[0,1] for i in k])
    scale=np.sqrt(exact_cov(0)[0,0]*exact_cov(0)[1,1])
    ax=axes[1]
    ax.plot(k,matched/scale,'o-',color='#0072B2',ms=3,label='Matched sensors')
    ax.plot(k,echo/scale,'s--',color='#D55E00',ms=3,mfc='white',label='Small echo')
    ax.set(title='(b) Recorded cross-covariance',xlabel='Lag k (samples)',ylabel=r'$C_{12}(k)/\sqrt{\Sigma_{11}\Sigma_{22}}$')
    ax.set_xticks([-8,-4,0,4,8]);ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.0));ax.grid(alpha=.15)
    ax=axes[2]
    odd=(echo-echo[::-1])/scale
    ax.axhline(0,color='#0072B2',lw=1.1,ls=':',label='Matched sensors')
    ax.plot(k,odd,'s-',color='#D55E00',ms=3,label='Small echo')
    ax.set(title='(c) Time-order contrast',xlabel='Lag k (samples)',ylabel=r'$[C_{12}(k)-C_{12}(-k)]/\sqrt{\Sigma_{11}\Sigma_{22}}$')
    ax.set_xticks([-8,-4,0,4,8]);ax.grid(alpha=.15)
    fig.tight_layout(w_pad=1.4);save(fig,output,'equilibrium_sensor_example')
    with (output/'equilibrium_sensor_example.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['lag','matched_C12','echo_C12','echo_time_order_contrast'])
        writer.writerows(zip(k,matched,echo,echo-echo[::-1]))

def mechanism(output):
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),gridspec_kw={'width_ratios':[1,1.1]})
    ax=axes[0];ax.set(xlim=(-1.6,1.6),ylim=(0,1));ax.axis('off')
    ax.text(-1.6,1.02,'(a) Two channels: calibrate',fontsize=9,fontweight='bold')
    box(ax,-1.53,.71,.91,.17,'Known input');arrow(ax,(-.6,.795),(-.3,.795));box(ax,-.28,.71,1.75,.17,'Relative sensor phase\nbound r')
    ax.axhspan(.23,.44,xmin=.36,xmax=.64,color='#0072B2',alpha=.16)
    ax.plot([-1.5,1.5],[.33,.33],color='#444444',lw=1)
    for x,t in [(-.45,r'$-2r\sqrt{v_1v_2}$'),(.45,r'$2r\sqrt{v_1v_2}$')]:
        ax.plot([x,x],[.27,.39],color='#444444',lw=.8);ax.text(x,.16,t,ha='center',fontsize=8)
    ax.text(0,.5,'Detector allowance',ha='center',fontsize=8)
    ax.text(1.5,.36,r'$\kappa$',ha='right');ax.text(0,.02,'Add a sampling margin before rejection',ha='center',fontsize=8)
    ax=axes[1];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(0,1.02,'(b) Three channels: cancel sensor phase',fontweight='bold',fontsize=9)
    points=[(.25,.7),(.8,.7),(.52,.2)]
    for i,(x,y) in enumerate(points):
        ax.add_patch(Circle((x,y),.065,fc='white',ec='#0072B2',lw=1.2));ax.text(x,y,str(i+1),ha='center',va='center');ax.plot([x,x],[y+.067,y+.125],color='#0072B2',lw=.8);ax.plot([x-.025,x,x+.025],[y+.15,y+.125,y+.15],color='#0072B2',lw=.8)
    for i,j in [(0,1),(1,2),(2,0)]:
        a=np.array(points[i]);b=np.array(points[j]);u=(b-a)/np.linalg.norm(b-a);arrow(ax,a+.07*u,b-.07*u,'#0072B2')
    ax.text(.525,.8,r'$\theta_1-\theta_2$',ha='center');ax.text(.81,.38,r'$\theta_2-\theta_3$',ha='center');ax.text(.22,.38,r'$\theta_3-\theta_1$',ha='center')
    ax.text(.52,.55,r'$F_{12}F_{23}F_{31}$',ha='center');ax.text(.52,.03,'Sensor phases sum to zero',ha='center')
    fig.tight_layout(w_pad=2);save(fig,output,'mechanism_compact')

def stack(ax, rows, positions, width=.6):
    bottoms=np.zeros(len(rows))
    for metric,color,label in zip(['reject','nonreject','abstain'],COLORS,OUTCOMES):
        y=np.array([int(r[metric])/int(r['n']) for r in rows])
        ax.bar(positions,y,bottom=bottoms,width=width,color=color,edgecolor='white',linewidth=.3,label=label)
        bottoms+=y
    y=np.array([int(r['reject'])/int(r['n']) for r in rows])
    lo=np.array([float(r['rejection_lower']) for r in rows]);hi=np.array([float(r['rejection_upper']) for r in rows])
    ax.errorbar(positions,y,yerr=np.vstack([y-lo,hi-y]),fmt='none',ecolor='#222222',elinewidth=.8,capsize=1.5)
    ax.set_ylim(0,1.05);ax.set_yticks([0,.5,1]);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.14)

def operating(rows,output):
    lookup={(r['cell'],r['method']):r for r in rows}
    selected=[]
    cells=sorted({r['cell']:int(r['N']) for r in rows if r['cell'].startswith('source_199_200_strong_')}.items(),key=lambda t:t[1])
    fig,axes=plt.subplots(2,2,figsize=(7,4.7),sharey=True)
    for ax,method,title in zip(axes.flat,['lag','two_sided','two_sided_intersection','supplied'],['Lag set','Two-sided set','Intersection','Supplied persistence']):
        data=[lookup[(cell,method)] for cell,N in cells];selected.extend(data)
        x=np.log2([(N-1)*.005 for cell,N in cells]);stack(ax,data,x)
        ticks=[10,40,160,640,2560]
        ax.set_xticks(np.log2(ticks),[f'{value:,}' for value in ticks]);ax.set_title(title);ax.set_xlabel(r'Record length in correlation times, $(N-1)(1-\phi)$')
        if method=='two_sided':ax.plot(x[-1],.90,marker='_',ms=13,color='black')
    axes[0,0].set_ylabel('Outcome fraction');axes[1,0].set_ylabel('Outcome fraction')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.035),frameon=False)
    fig.tight_layout();save(fig,output,'source_operating_compact')
    fig,axes=plt.subplots(2,3,figsize=(7,4.7),sharex=True,sharey=True)
    for i,prefix in enumerate(['cycle_null_','cycle_rotating_']):
        cells=sorted({r['cell']:int(r['N']) for r in rows if r['cell'].startswith(prefix)}.items(),key=lambda t:t[1])
        for j,(method,title) in enumerate(zip(['common','diagonal','combined_frequencies'],['Common bound','Diagonal bounds','Two frequencies'])):
            ax=axes[i,j];data=[lookup[(cell,method)] for cell,N in cells];selected.extend(data)
            x=np.log2([N*np.log(4) for cell,N in cells]);stack(ax,data,x)
            ticks=[14,16,18,20]
            ax.set_xticks(ticks,[rf'$2^{{{power}}}$' for power in ticks])
            if i==0:ax.set_title(title)
            if j==0:ax.set_ylabel(('Equilibrium' if i==0 else 'Rotation')+'\nOutcome fraction')
            if i==1:ax.set_xlabel(r'Record length / $\tau$')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.035),frameon=False)
    fig.tight_layout();save(fig,output,'cycle_operating_compact')
    methods=['calibrated','phase_ignored','baseline/independent_windows','baseline/moving_block','baseline/imaginary_coherency']
    labels=['C','P','I','B','H']
    fig,axes=plt.subplots(1,3,figsize=(7,3.5),sharey=True)
    for ax,T in zip(axes,[1,2,4]):
        cells=sorted({r['cell']:int(r['N']) for r in rows if r['cell'].startswith(f'brownian_T{T}_')}.items(),key=lambda t:t[1])
        data=[lookup[(cell,method)] for cell,N in cells for method in methods];selected.extend(data)
        x=np.array([i*6+j for i in range(3) for j in range(5)]);stack(ax,data,x,width=.78)
        ax.set_xticks(x,labels*3,fontsize=6.5);ax.set_title(r'$T_2/T_1=$'+str(T))
        for i,(cell,N) in enumerate(cells):
            ax.text(i*6+2,-.19,f'{N/2:,.0f}',ha='center',transform=ax.get_xaxis_transform(),fontsize=7)
            if i<2:ax.axvline(i*6+5,color='#bbbbbb',lw=.5)
        ax.set_xlabel(r'Record length / $\tau$',labelpad=21)
    axes[0].set_ylabel('Outcome fraction')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.07),frameon=False)
    fig.tight_layout();save(fig,output,'brownian_operating_compact')
    with (output/'operating_compact_counts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
    return len(selected)

def analytical_references(output):
    """Calculate model references without evaluating any observed record."""
    e=.03
    root=-e/(1+e*e+np.sqrt(1-e*e+e**4))
    phase=e*(1-root*root)/np.sqrt(1+2*e*root+e*e)
    N=524289; n=N-1; q=56.; t=np.log(2/.03)
    norm=np.sqrt((2*n+2-4/n-4/n**2)/(2*n*n))
    radius=2*min(q*norm,np.sqrt(2*q*N)/n)*np.sqrt(t)+2*q*t/n
    correction=1-q/N-2*np.sqrt(q*np.log(200)/N)
    rows=[]
    for temperature in [1,2,4]:
        S=exact_cov(0,temperature)
        source=exact_cov(1,temperature)[0,1]-exact_cov(-1,temperature)[0,1]
        observed=source+e*(exact_cov(2,temperature)[0,1]-S[0,1])
        v1=S[0,0]; v2=(1+e*e)*S[1,1]+2*e*exact_cov(1,temperature)[1,1]
        envelope=np.linalg.eigvalsh(S)[-1]*(1+np.exp(-.5))/(1-np.exp(-.5))*max(1/v1,(1+e)**2/v2)
        rows.append({'temperature_ratio':temperature,'source_normalized_contrast':abs(source)/np.sqrt(S[0,0]*S[1,1]),'observed_normalized_contrast':abs(observed)/np.sqrt(v1*v2),'sufficient_model_envelope':envelope,'sufficient_model_envelope_rounded_up':np.ceil(envelope*100)/100,'entropy_rate':(temperature-1)**2/(8*temperature)})
    with (output/'analytical_model_references.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    data={'echo':e,'phase_supremum':float(phase),'maximizer_cos_omega':float(root),'twice_phase_supremum':float(2*phase),'N':N,'shared_family_envelope':q,'normalized_sampling_radius':float(radius),'variance_correction':float(correction),'population_scale_threshold_reference':float((2*phase+radius)/correction),'scope':'Analytical model references. The threshold uses population variances as a reference. No fitted threshold, record decision, or observed power was recalculated.'}
    (output/'analytical_threshold_reference.json').write_text(json.dumps(data,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=args.package_root/'results/current/operating/complete_counts_with_design.csv'
    with source.open(newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        n=int(row['n']);counts=[int(row[k]) for k in ['reject','nonreject','abstain']]
        if sum(counts)!=n or min(counts)<0:raise ValueError('Saved outcome counts are inconsistent.')
        if abs(counts[0]/n-float(row['rejection_fraction']))>1e-14:raise ValueError('Saved fraction is inconsistent.')
    equilibrium(args.output);mechanism(args.output);count=operating(rows,args.output);analytical_references(args.output)
    metadata={'source':'results/current/operating/complete_counts_with_design.csv','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selected_count_rows':count,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'analytical_model':{'drift':[[1,.5],[.5,1]],'diffusion':[[1,0],[0,1]],'sampling_interval':1,'echo':.03,'lag_convention':'Cij(k)=Cov(Yi(t+k),Yj(t))','matched_responses':[1,1],'echo_response':'H1=1; H2=1+0.03 exp(-i omega)'},'correlation_time_conventions':{'source':'tau approximation = 1/(1-phi); x=(N-1)(1-phi), phi=.995','brownian':'tau = 1/min eigenvalue(A) = 2 samples; x=N/2','cycle':'tau = 1/log(4) samples; x=N log(4)'},'uncertainty':'Saved pointwise 95 percent rejection intervals. All three outcomes use full denominators.'}
    (args.output/'article_figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'figures':5,'count_rows':count,'output':str(args.output)}))
if __name__=='__main__':main()
