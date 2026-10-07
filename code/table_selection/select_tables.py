"""Select fixed publication table rows from complete validated summaries."""

import argparse
import csv
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostic_core'))
import present_diagnostics as displays


IDENTITY = ('cycle_identity_null', 'cycle_identity_rotating')
NUMERICAL = ('source_97_100_null_65537', 'source_97_100_strong_65537')
FACTORS = ('1', '3/2', '2', '4')
GAINS = ('1/16', '1/4', '1', '4', '16', '-1')
CYCLE_METHODS = ('common', 'diagonal', 'combined_frequencies')
CYCLE_CELLS = ('cycle_null_131072', 'cycle_rotating_131072', *IDENTITY)
EDGES = ('0-1', '1-2', '2-0')
COMPONENTS = ('tail_bias', 'centering', 'stochastic_sqrt', 'stochastic_linear',
              'recording', 'total_display')
SETTING_METHODS = ('two_sided',
                   *(f'numerical/depth_{n}/two_sided' for n in (20, 28, 36)),
                   *(f'numerical/precision_{n}/two_sided' for n in (80, 160, 320)),
                   'fractional_8/two_sided', 'fractional_12/two_sided',
                   'recorder_8/two_sided', 'recorder_12/two_sided')
NEAR_METHODS = ('calibrated', 'phase_ignored', 'baseline/independent_windows',
                'baseline/independent_windows_phase_adjusted', 'baseline/moving_block',
                'baseline/moving_block_phase_adjusted', 'baseline/imaginary_coherency')
NOTE = ('R, NR, and A mean rejection, nonrejection, and abstention. The full denominator '
        'includes all independent records. The nominal target does not give a size guarantee. '
        'A violated source, covariance, response, or error assumption removes that guarantee. '
        'The baseline rules have no finite-record source-null guarantee here. Interval endpoints '
        'are rounded outward to 0.001. Descriptive medians have three significant digits. '
        'A missing value is shown by a dash. Exact values and all finite and missing counts '
        'remain in the associated CSV files.')
TARGET_CODES = {
    'Source reversibility under stated assumptions': 'T0',
    'Zero observed lag contrast': 'T1',
    'Observed lag contrast within plug-in phase tolerance': 'T2',
    'Zero observed imaginary coherency': 'T3',
    'Zero observed two-channel contrast': 'T4',
}
TARGET_NOTE = ('Nominal targets: T0, source reversibility under the stated assumptions; '
               'T1, zero observed lag contrast; T2, observed lag contrast within plug-in phase '
               'tolerance; T3, zero observed imaginary coherency; T4, zero observed two-channel contrast.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specification():
    inflation = [(c, f'inflation_K{k}_B{b}') for c in IDENTITY for k in FACTORS for b in FACTORS]
    gain = [(c, f'gain_{g}/{m}') for c in IDENTITY for g in GAINS
            for m in ('common', 'diagonal', 'inverse_transport')]
    settings = [(c, m) for c in NUMERICAL for m in SETTING_METHODS]
    violations = [(f'source_{v}_{regime}', 'two_sided')
                  for v in ('reference_snr_10', 'reference_snr_100', 'student_t_5')
                  for regime in ('null', 'strong')]
    violations += [(f'cycle_correlated_error_{regime}', 'diagonal')
                   for regime in ('null', 'rotating')]
    violations += [(c, f'underestimated_{bound}_{factor}') for c in IDENTITY
                   for bound in ('K', 'B') for factor in ('2/3', '1/2')]
    near = [(f'phase_boundary_{ratio}', m) for ratio in ('1_2', '9_10', '99_100')
            for m in NEAR_METHODS]
    cycle = [(c, m, e, term) for c in CYCLE_CELLS for m in CYCLE_METHODS
             for e in EDGES for term in COMPONENTS]
    return {
        'inflation_counts': ('complete_counts.csv', ('cell', 'method'), inflation),
        'inflation_pairs': ('declared_paired_controls.csv', ('cell', 'variant', 'reference'),
                            [(c, m, 'diagonal') for c, m in inflation]),
        'gain_counts': ('complete_counts.csv', ('cell', 'method'), gain),
        'gain_pairs': ('declared_paired_controls.csv', ('cell', 'variant', 'reference'),
                      [(c, m, 'common' if m.endswith('/common') else 'diagonal') for c, m in gain]),
        'settings_counts': ('complete_counts.csv', ('cell', 'method'), settings),
        'settings_pairs': ('declared_paired_controls.csv', ('cell', 'variant', 'reference'),
                          [(c, m, 'two_sided') for c, m in settings if m != 'two_sided']),
        'violations_counts': ('complete_counts.csv', ('cell', 'method'), violations),
        'near_boundary_counts': ('complete_counts.csv', ('cell', 'method'), near),
        'cycle_components': ('complete_cycle_radius_components.csv',
                             ('cell', 'method', 'edge', 'component'), cycle),
    }


def select_exact(rows, key_fields, expected):
    indexed = {}
    for line, row in enumerate(rows, 2):
        key = tuple(row[k] for k in key_fields)
        if key in indexed:
            raise ValueError('Duplicate input identity: ' + repr(key))
        indexed[key] = (line, row)
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate selection identity.')
    missing = set(expected) - indexed.keys()
    if missing:
        raise ValueError('A declared table row is missing: ' + repr(sorted(missing)))
    return [indexed[key] for key in expected]


def checked_display(fixture_root, fixture):
    marker_path = fixture_root / 'output/DISPLAY_COMPLETE.json'
    marker = json.loads(marker_path.read_text())
    if marker.get('fixture') is not fixture:
        raise ValueError('Fixture mode differs from the complete display.')
    for section, base in (('inputs', None), ('outputs', fixture_root / 'output')):
        seen = set()
        for entry in marker[section]:
            path = Path(entry['path']) if base is None else base / entry['path']
            if base is not None and Path(entry['path']).name != entry['path']:
                raise ValueError('A display output path is invalid.')
            if path.resolve() in seen or digest(path) != entry['sha256']:
                raise ValueError('A bound display file differs or is duplicated.')
            seen.add(path.resolve())
    registry_path = fixture_root / ('REGISTRY_FIXTURE.json' if fixture else 'REGISTRY_001.json')
    registry, metadata, _, _, _ = displays.checked_inputs(
        registry_path, fixture_root / 'analysis', fixture_root / 'presentation',
        fixture_root / 'diagnostics', fixture)
    if marker['records'] != metadata['records']:
        raise ValueError('Display and analysis record counts differ.')
    return registry, marker, marker_path


def escape(value):
    return ''.join({'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}.get(c, c)
                   for c in str(value))


def number(value):
    if value == '':
        return r'\textemdash'
    d = Decimal(value)
    if not d.is_finite():
        raise ValueError('A table value is nonfinite.')
    if d.is_zero():
        return '0'
    return escape(format(d, '.3g'))


def interval(lower, upper):
    lo, hi = Decimal(lower), Decimal(upper)
    if not lo.is_finite() or not hi.is_finite() or lo > hi:
        raise ValueError('A table interval is invalid.')
    return '[' + format(lo.quantize(Decimal('.001'), rounding=ROUND_FLOOR), '.3f') + ', ' + \
        format(hi.quantize(Decimal('.001'), rounding=ROUND_CEILING), '.3f') + ']'


def table_tex(title, headers, rows, note, fixture):
    if 'Nominal test target' in headers:
        widths = ([.15, .14, .04, .08, .12, .12, .16, .08] if len(headers) == 8
                  else [.18, .21, .04, .08, .14, .14, .08])
    else:
        widths = [.18, .06, .11, .11, .11, .11, .11, .11]
    columns = '@{}' + ''.join('>{\\raggedright\\arraybackslash}p{%.3f\\textwidth}' % width
                              for width in widths) + '@{}'
    lines = [r'\begingroup\small\setlength{\tabcolsep}{3pt}',
             r'\subsection*{' + escape(title + (' [DETERMINISTIC FIXTURE]' if fixture else '')) + '}',
             r'\begin{longtable}{' + columns + '}', r'\toprule',
             ' & '.join(escape(h) for h in headers) + r' \\\midrule\endhead']
    lines += [' & '.join(row) + r' \\' for row in rows]
    lines += [r'\bottomrule\end{longtable}', escape(note), r'\endgroup', '']
    return '\n'.join(lines)


def compact_label(method, group):
    if group == 'settings_counts':
        if method == 'two_sided':
            return 'Primary two-sided'
        return displays.label_method(method).replace('Two-sided set, ', '')
    if group == 'inflation_counts':
        k, b = method[len('inflation_K'):].split('_B')
        return 'Covariance factor ' + k + ', bias factor ' + b
    if group == 'gain_counts':
        gain, base = displays.split_method_setting(method, 'gain_')
        return 'Gain ' + gain + ', ' + {'common': 'common', 'diagonal': 'diagonal',
                                     'inverse_transport': 'inverse transport'}[base]
    return displays.label_method(method)


def compact_condition(cell):
    if cell['id'] in IDENTITY:
        return 'Identity, ' + ('reversible' if cell['id'].endswith('null') else 'rotation')
    if cell['id'] in NUMERICAL:
        return 'phi 97/100, ' + ('reversible' if '_null_' in cell['id'] else 'strong delay')
    if cell['id'] in CYCLE_CELLS:
        return 'Primary, ' + ('reversible' if '_null_' in cell['id'] else 'rotation')
    return displays.condition_label(cell)


def write_csv(path, rows):
    if not rows:
        raise ValueError('An empty table cannot be exported.')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_count_metadata(selected, numeric_rows, source_path, source_sha256):
    """Check finite counts. Normalize only an empty count for an absent metric."""
    index = {}
    for line, row in enumerate(numeric_rows, 2):
        key = row['cell'], row['method'], row['metric']
        if key in index:
            raise ValueError('Duplicate numeric summary identity.')
        index[key] = line, row
    changes = []
    for name, rows in selected.items():
        if not name.endswith('_counts'):
            continue
        for output_row, row in enumerate(rows, 2):
            full = int(row['n'])
            if full < 0:
                raise ValueError('A full denominator is negative.')
            for metric in displays.DECISION_METRICS:
                key = row['cell'], row['method'], metric
                if key not in index:
                    raise ValueError('A numeric count summary is missing.')
                line, summary = index[key]
                finite, missing = int(summary['finite_n']), int(summary['missing_n'])
                if int(summary['n']) != full or finite < 0 or missing < 0 or finite + missing != full:
                    raise ValueError('Numeric finite and missing counts differ from n.')
                for label in ('min', 'q25', 'median', 'q75', 'max'):
                    original, numeric = row[metric + '_' + label], summary[label]
                    if finite == 0:
                        if original != '' or numeric != '':
                            raise ValueError('An absent metric has a numeric summary.')
                    else:
                        if original == '' or numeric == '':
                            raise ValueError('A finite metric has a missing summary.')
                        a, b = Decimal(original), Decimal(numeric)
                        if not a.is_finite() or not b.is_finite() or a != b:
                            raise ValueError('Count and numeric descriptive summaries differ.')
                count_field = metric + '_n'
                original_count = row[count_field]
                if original_count == '':
                    if finite != 0:
                        raise ValueError('An empty finite count has available values.')
                    row[count_field] = '0'
                    changes.append(dict(table=name, output_row=output_row, field=count_field,
                                        original='', normalized='0', key=list(key),
                                        numeric_source_path=str(source_path), numeric_source_line=line,
                                        numeric_source_sha256=source_sha256,
                                        finite_n=summary['finite_n'], missing_n=summary['missing_n'],
                                        n=summary['n']))
                elif int(original_count) != finite:
                    raise ValueError('Count and numeric finite counts differ.')
    return changes


def export(root, output, fixture=False):
    if output.exists():
        raise FileExistsError(output)
    registry, marker, marker_path = checked_display(root, fixture)
    cells = {c['id']: c for c in registry['cells']}
    dependencies = [Path(__file__), Path(displays.__file__),
                    Path(__file__).resolve().parents[1] / 'summary_display/present_study.py',
                    Path(__file__).resolve().parents[1] / 'summary_display/validate_summaries.py',
                    Path(__file__).resolve().parents[1] / 'display_selection/PUBLICATION_SELECTION_001.md']
    selected, mappings, inputs = {}, {}, {p: digest(p) for p in [marker_path, *dependencies]}
    output_names = {entry['path'] for entry in marker['outputs']}
    for name, (source_name, key_fields, keys) in specification().items():
        if source_name not in output_names:
            raise ValueError('The display marker omits a consumed table.')
        source = root / 'output' / source_name
        with source.open() as f:
            rows = list(csv.DictReader(f))
        matches = select_exact(rows, key_fields, keys)
        selected[name] = [dict(row) for _, row in matches]
        mappings[name] = [dict(output_row=i, source_path=str(source.resolve()), source_line=line,
                               source_sha256=digest(source), key=list(key),
                               source_row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
                          for i, ((line, row), key) in enumerate(zip(matches, keys), 2)]
        inputs[source] = digest(source)
    numeric_source = root / 'output/complete_numeric_summaries.csv'
    if numeric_source.name not in output_names:
        raise ValueError('The display marker omits the numeric count summaries.')
    with numeric_source.open() as stream:
        numeric_rows = list(csv.DictReader(stream))
    inputs[numeric_source] = digest(numeric_source)
    normalized_counts = normalize_count_metadata(selected, numeric_rows, numeric_source.resolve(),
                                                  inputs[numeric_source])
    for name, rows in selected.items():
        if name.endswith('_counts'):
            for row in rows:
                row['nominal_target'] = displays.test_target(row['method'])
                for metric in displays.DECISION_METRICS:
                    full, finite = int(row['n']), int(row[metric + '_n'])
                    if not 0 <= finite <= full:
                        raise ValueError('A metric count is outside its denominator.')
                    row[metric + '_missing_n'] = str(full - finite)
    tex, tex_mapping = {}, {}
    paired_groups = {'inflation_counts': 'inflation_pairs', 'gain_counts': 'gain_pairs',
                     'settings_counts': 'settings_pairs'}
    for group in ('inflation_counts', 'gain_counts', 'settings_counts',
                  'violations_counts', 'near_boundary_counts'):
        pair = {(r['cell'], r['variant']): r for r in selected.get(paired_groups.get(group), [])}
        pair_mapping = {tuple(entry['key'][:2]): entry
                        for entry in mappings.get(paired_groups.get(group), [])}
        tex_mapping[group + '.tex'] = []
        rows = []
        for index, r in enumerate(selected[group]):
            cell = cells[r['cell']]
            condition = escape(compact_condition(cell)) + ', N=' + str(cell['N'])
            median = number(r['threshold_median']) + ' [' + r['threshold_n'] + '/' + r['n'] + ']'
            values = [condition, escape(compact_label(r['method'], group)), r['n'],
                      '/'.join(r[k] for k in ('reject', 'nonreject', 'abstain')),
                      escape(interval(r['rejection_lower'], r['rejection_upper'])), median]
            if group in paired_groups:
                p = pair.get((r['cell'], r['method']))
                if p is None:
                    if group != 'settings_counts' or r['method'] != 'two_sided':
                        raise ValueError('A declared paired table row is missing.')
                    values += ['Reference']
                else:
                    if int(p['n']) != int(r['n']):
                        raise ValueError('Paired and count denominators differ.')
                    values += [number(p['variant_difference']) + ' ' +
                               escape(interval(p['variant_lower'], p['variant_upper']))]
            values += [TARGET_CODES[r['nominal_target']]]
            rows.append(values)
            tex_mapping[group + '.tex'].append(dict(
                data_row=index + 1, count_source=mappings[group][index],
                paired_source=pair_mapping.get((r['cell'], r['method'])),
                registry_cell=r['cell'], method=r['method'],
                exact_nominal_target=r['nominal_target'],
                median_fields=['threshold_median', 'threshold_n', 'n'],
                interval_fields=['rejection_lower', 'rejection_upper']))
        headers = ['Condition', 'Method', 'n', 'R/NR/A', '95% rejection interval',
                   'Threshold median [finite/full]']
        note = NOTE + ' ' + TARGET_NOTE
        if group in paired_groups:
            headers += ['Variant minus reference [95% interval]']
            note += ' Paired differences use discordant outcomes. Inverse gain transport uses the diagonal reference; other gains use their named rule. Inflation uses the diagonal reference. Settings use the primary two-sided reference.'
        headers += ['Nominal test target']
        tex[group] = table_tex(group.replace('_', ' '), headers, rows, note, fixture)
    cycle_index = {(r['cell'], r['method'], r['edge'], r['component']): r
                   for r in selected['cycle_components']}
    cycle_mapping = {tuple(entry['key']): entry for entry in mappings['cycle_components']}
    for method in CYCLE_METHODS:
        rows = []
        tex_mapping['cycle_' + method + '.tex'] = []
        for c in CYCLE_CELLS:
            for e in EDGES:
                values = [escape(compact_condition(cells[c])) + ', N=' + str(cells[c]['N']),
                          {'0-1': '1--2', '1-2': '2--3', '2-0': '3--1'}[e]]
                for component in COMPONENTS:
                    r = cycle_index[c, method, e, component]
                    if int(r['finite_n']) + int(r['missing_n']) != int(r['n']):
                        raise ValueError('Cycle finite and missing counts differ from n.')
                    values += [number(r['median']) + ' [' + r['finite_n'] + '/' + r['n'] + ']']
                rows.append(values)
                tex_mapping['cycle_' + method + '.tex'].append(dict(
                    data_row=len(rows), registry_cell=c, method=method, edge=e,
                    components={term: cycle_mapping[c, method, e, term] for term in COMPONENTS},
                    displayed_fields=['median', 'finite_n', 'n']))
        tex['cycle_' + method] = table_tex(displays.label_method(method) + ' cycle radius terms',
                                          ['Condition', 'Edge', 'Tail bias', 'Centering',
                                           'Square-root sampling', 'Linear sampling', 'Recording', 'Total'],
                                          rows, 'Each entry is the descriptive median [finite/full]. '
                                          'These Decimal50 formula values are not saved component enclosures. '
                                          'All exact descriptive summaries and missing counts remain in the CSV. '
                                          'A dash means no finite value. Medians have three significant digits.', fixture)
    output.mkdir()
    for name, rows in selected.items():
        write_csv(output / (name + '.csv'), rows)
    for name, value in tex.items():
        (output / (name + '.tex')).write_text(value)
    plan = {name: dict(source=source, key_fields=fields, keys=keys)
            for name, (source, fields, keys) in specification().items()}
    (output / 'SELECTION_KEYS_001.json').write_text(json.dumps(plan, indent=2) + '\n')
    (output / 'TABLE_SOURCE_MAP_001.json').write_text(json.dumps(mappings, indent=2) + '\n')
    (output / 'TEX_SOURCE_MAP_001.json').write_text(json.dumps(tex_mapping, indent=2) + '\n')
    (output / 'FINITE_COUNT_NORMALIZATION_001.json').write_text(json.dumps(normalized_counts, indent=2) + '\n')
    receipt = dict(fixture=fixture, scope='Fixed publication table rows. No new inference.',
                   tables={name: len(rows) for name, rows in selected.items()},
                   inputs=[dict(path=str(p.resolve()), sha256=h) for p, h in inputs.items()],
                   outputs=[dict(path=p.name, sha256=digest(p)) for p in sorted(output.iterdir())])
    (output / 'TABLES_COMPLETE_001.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    export(args.input_root, args.output, args.fixture)


if __name__ == '__main__':
    main()
