"""Render the analytic connected-window display."""
from pathlib import Path
from fractions import Fraction
import argparse,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
FIG=args.out.resolve();FIG.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
TEAL,ORANGE,GREY='#007C83','#B75A22','#64717D'
def write_csv(name,rows):
    with (FIG/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(FIG/(name+'.png'),dpi=180)
    plt.close(fig)
# Connected-window witnesses: exact rational covariance values, SI S22.
window_rows = []
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True)
for ax, horizon, delay, title in zip(axes, [2, 3, 3], [1, 1, 2],
        ['a  Two times; common delay 1', 'b  Three times; calibrated delays',
         'c  Three times; delay 2 allowed']):
    for k in range(-2, 6):
        observed_a = Fraction(1, 4) if k == 0 else Fraction(0)
        observed_b = Fraction(1, 4) if k == 1 else Fraction(0)
        witness_a = Fraction(1, 4) if k in (0, 2 * delay) else Fraction(0)
        witness_b = (Fraction(1, 4) if k == 1 else Fraction(0)) if delay == 1 else (Fraction(1, 4) if k in (1, 3) else Fraction(0))
        window_rows.append({'panel':title[0], 'lag':k, 'horizon':horizon, 'delay':delay,
                            'observed_A':str(observed_a), 'observed_B':str(observed_b),
                            'common_witness_A':str(witness_a), 'common_witness_B':str(witness_b),
                            'retained':abs(k)<horizon})
    ax.axvspan(-horizon+.5, horizon-.5, color='#EEF2F3', zorder=0)
    ax.axhline(0, color=GREY, lw=.5)
    ks = list(range(-2, 6))
    values = window_rows[-len(ks):]
    ax.plot(ks, [float(Fraction(r['observed_A'])) for r in values], 'o', color=TEAL, ms=6, label='Observed A')
    ax.plot(ks, [float(Fraction(r['observed_B'])) for r in values], 's', color=ORANGE, ms=4, label='Observed B')
    ax.plot([0, 2*delay], [.25, .25], 'o', mfc='none', mec=TEAL, ms=11, mew=1.4, label='Common-delay A')
    ax.plot([1] if delay == 1 else [1, 3], [.25] if delay == 1 else [.25, .25], 's', mfc='none', mec=ORANGE, ms=9, mew=1.2, label='Common-delay B')
    ax.set(title=title, xlabel='Cross-covariance lag k', xticks=[-2,-1,0,1,2,3,4,5], ylim=(-.025,.37))
    if ax is axes[0]:
        ax.text(.03,.9,'Shading: retained lags', transform=ax.transAxes, fontsize=9, color=GREY)
axes[0].set_ylabel('Cross covariance')
axes[1].annotate('Delay 1 fails in A', xy=(2,.25), xytext=(.4,.32), fontsize=9,
                 arrowprops=dict(arrowstyle='->',color=GREY))
axes[1].text(.03,.45,'Delay 0 fails in B:\nC(1) differs from C(−1)',transform=axes[1].transAxes,fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,-.035), ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=[0,.06,1,1])
write_csv('connected_window.csv', window_rows)
save(fig,'connected_window')

