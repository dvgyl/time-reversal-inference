"""Render every planned filter setting, including the post hoc refinement."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
ROOT=args.root.resolve()
OUT=args.out.resolve()
OUT.mkdir(exist_ok=True)
rows=[]
for relative in ('filter_study/summary.csv','projection_variance_results/summary.csv'):
    with (ROOT/relative).open() as f:
        rows.extend(csv.DictReader(f))
styles={
    'student_ignores_filter': ('#A34337','--','Ignores filter shape'),
    'known_variance_filter': ('#7B8794',':','Known variance ceiling'),
    'same_sample_variance_filter': ('#315F8B','-','Estimated common ceiling'),
    'same_sample_projection_variances': ('#147D72','-','Estimated projection variances*'),
}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,4,figsize=(12,6),sharex=True,sharey=True)
for column,a in enumerate((0.0,.05,.15,.25)):
    axes[0,column].set_title(f'Filter coefficient a = {a:g}',fontweight='bold')
    for method,(color,linestyle,label) in styles.items():
        for row_index,model in ((0,'null_common_zero'),(0,'null_common_one'),(1,'different_delays')):
            points=sorted((x for x in rows if float(x['filter_a'])==a and x['method']==method
                           and x['model']==model),key=lambda x:int(x['records_per_condition']))
            x=[int(p['records_per_condition']) for p in points]
            y=[float(p['fraction']) for p in points]
            marker='x' if model=='null_common_one' else 'o'
            axis=axes[row_index,column]
            axis.plot(x,y,color=color,linestyle=linestyle,marker=marker,markersize=3,
                      linewidth=1.5,label=label if row_index==1 else None,alpha=.9)
            axis.fill_between(x,[float(p['cp95_lower']) for p in points],
                              [float(p['cp95_upper']) for p in points],color=color,alpha=.09)
    for axis in axes[:,column]:
        axis.set_xscale('log',base=2)
        axis.set_xticks([128,2048,32768],['128','2,048','32,768'])
        axis.set_ylim(-.035,1.045)
        axis.set_yticks([0,.25,.5,.75,1])
        axis.grid(axis='y',alpha=.18,linewidth=.6)
    axes[0,column].axhline(.05,color='black',lw=.8,alpha=.4)
    axes[1,column].set_xlabel('Independent records per condition')
axes[0,0].set_ylabel('False rejection fraction')
axes[1,0].set_ylabel('Alternative rejection fraction')
fig.suptitle('Filter shape changes the validity of a time-reversal test',x=.075,y=.985,
             ha='left',fontsize=15,fontweight='bold')
handles,labels=axes[1,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=4,
           frameon=False,fontsize=8)
fig.text(.075,.015,'5,000 synthetic pairs per cell. Shading shows marginal 95% binomial intervals. '
         'Both common-delay nulls are shown.\n*Theory-led post hoc refinement on all saved draws. '
         'The a = 0.25 setting retains the failure to detect the alternative.',fontsize=8)
fig.tight_layout(rect=(.025,.13,1,.945))
for extension in ('png','pdf'):
    fig.savefig(OUT/f'filter_study.{extension}',dpi=180,bbox_inches='tight')
plt.close(fig)
report=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            inputs={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                    ('filter_study/summary.csv','projection_variance_results/summary.csv')},
            outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (OUT/'filter_study.png',OUT/'filter_study.pdf')},
            coverage='All 60 model cells and all four procedures. Null configurations share each upper panel.')
(OUT/'FIGURE_BINDING.json').write_text(json.dumps(report,indent=2)+'\n')
print('Saved all-cell figure:',OUT/'filter_study.png')
