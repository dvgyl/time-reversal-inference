"""Render the complete calibration-count slice from the saved cost table."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / 'dependent_cost_results/dependent_cost.csv').open()))
selected = [row for row in rows if (int(row['length']), float(row['filter_a']),
            float(row['calibration_noise'])) == (2, 0.05, 0.1)]
assert len(selected) == 15
lines = [r'\begin{table}[t]\centering\small',
         r'\caption{\textbf{Calibration effort changes sufficient recording length.} '
         r'The source spectral ratio is $M_0/m_0$. Each finite-calibration row uses $2M$ calibration '
         r'observations per channel, filter length two, $a=0.05$, and calibration '
         r'noise ceiling $0.1$. Entries are the first sufficient $N$ on the declared '
         r'grid for size at most $0.05$ and power at least $0.95$. A dash means no '
         r'passing grid point through $524{,}288$. It does not prove impossibility. '
         r'The last row uses supplied exact phase information. All values are '
         r'conditional analytic bounds.}\label{tab:calibration}',
         r'\begin{tabular}{rrrrr}\toprule',
         r'Impulse blocks $M$ & Phase bound $r^+$ & Ratio 1 & Ratio 2 & Ratio 4\\\midrule']
for blocks in [16, 64, 256, 1024, 4096]:
    group = [row for row in selected if int(row['impulse_blocks']) == blocks]
    assert [int(row['spectral_ratio']) for row in group] == [1, 2, 4]
    values = [format(int(row['first_sufficient_grid_length']), ',')
              if row['first_sufficient_grid_length'] else r'---' for row in group]
    lines.append(f"{blocks:,} & {float(group[0]['phase_upper']):.5f} & " +
                 ' & '.join(values) + r'\\')
known = [format(int(row['known_phase_first_grid_length']), ',') for row in selected[:3]]
lines += [r'\midrule Exact phase & $0.05$ & ' + ' & '.join(known) + r'\\',
          r'\bottomrule\end{tabular}\end{table}']
(ROOT / 'calibration_table.tex').write_text('\n'.join(lines) + '\n')
