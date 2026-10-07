"""Build the complete matched and interpolated smoothing-table slices."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--costs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.costs.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r['common_response'] == 'no_dominant_tap'
                and float(r['calibration_noise']) == 0.02 and float(r['delta']) in [0, 0.05]]
    if len(selected) != 30:
        raise ValueError('The complete thirty-row source slice is required.')
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Direct calibration of a shared smoothing response. Both channels use $G(B)=(1+0.5B)^2$. Channel 1 also uses $1-\delta+\delta B$. Each complete response has unit coefficient norm and no dominant tap. The calibration noise standard deviation is at most $0.02$. Each channel has four declared coefficient positions and uses $4M$ calibration observations. Entries give the first sufficient testing length on the declared grid for size at most $0.05$ and power at least $0.95$. Phase bounds are rounded for display. Counts use full-precision values. A dash means that no tested length passes. The separate-channel phase construction gives no sufficient count in any row. The known-response control uses exact phase and filter shape. These are conditional analytic bounds, not measured performance or optimal counts.}',
             r'\label{tab:relative-calibration}',
             r'\begin{tabular}{@{}rrrrrr@{}}', r'\toprule',
             r'$\delta$ & Blocks $M$ & Phase bound $r^+$ & $S=1$ & $S=2$ & $S=4$ \\',
             r'\midrule']
    for delta in [0, 0.05]:
        for blocks in [64, 256, 1024, 4096, 16384]:
            group = [r for r in selected if float(r['delta']) == delta and int(r['impulse_blocks']) == blocks]
            if [int(r['spectral_ratio']) for r in group] != [1, 2, 4]:
                raise ValueError('The source ratios must be complete and ordered.')
            counts = [format(int(r['first_sufficient_grid_length']), ',') if r['first_sufficient_grid_length'] else '---' for r in group]
            lines.append(f'{delta:g} & {blocks:,} & {float(group[0]["phase_upper"]):.5f} & '+ ' & '.join(counts)+r'\\')
        controls = [r for r in selected if float(r['delta']) == delta and int(r['impulse_blocks']) == 64]
        values = [format(int(r['exact_first_grid_length']), ',') if r['exact_first_grid_length'] else '---' for r in controls]
        lines.append(r'\multicolumn{2}{l}{Known response, $\delta='+f'{delta:g}'+r'$} & '+f'{float(controls[0]["exact_phase"]):.5f} & '+' & '.join(values)+r'\\')
        if delta == 0:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    args.out.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
