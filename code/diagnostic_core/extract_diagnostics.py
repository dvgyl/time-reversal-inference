"""Extract completed decision receipts without calling inference code."""

import argparse
from collections import Counter
import csv
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import gzip
import hashlib
import json
from pathlib import Path

CATEGORIES = ('input_numerical', 'empty_source_set', 'hull_reaching_one',
              'no_finite_covariance', 'no_positive_scale_denominator',
              'available_insufficient_margin', 'rejection')
ANALYSIS_FILES = ('decisions.csv', 'source_sets.csv', 'source_summary.csv',
                  'cell_summary.csv', 'paired_comparisons.csv')
FORMULA_FILES = ('study/evaluate_methods.py', 'study/analyze_study.py', 'study/run_study.py',
                 'cycle/calibrated_adapters.py', 'cycle/calibrated_threshold.py',
                 'cycle/cycle_diagonal.py', 'cycle/octant_cycle.py',
                 'cycle/vendor/cycle_reference.py',
                 'source_inference/implementation_001/two_sided_source.py')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def exact(value):
    if isinstance(value, dict) and set(value) == {'__fraction__'}:
        return Q(value['__fraction__'])
    if type(value) is int:
        return Q(value)
    raise ValueError('An exact rational receipt was required.')


def encoded(value):
    return {'__fraction__': str(value)}


def dec(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def numerical(reason):
    return any(part in reason.lower() for part in (
        'max_', 'resource_limit', 'work_limit', 'compilation', 'moments_unavailable',
        'nonfinite', 'input', 'recording_moments', 'saturation', 'rational_bits',
        'uncertain_cutoff', 'arithmetic', 'precision'))


def status(value):
    if value.get('__type__') == 'MemberDecision':
        return 'reject' if value['reject'] else 'nonreject'
    raw = value.get('decision', value.get('status'))
    mapping = {'REJECT': 'reject', 'DO_NOT_REJECT': 'nonreject', 'ABSTAIN': 'abstain',
               'reject': 'reject', 'nonreject': 'nonreject', 'abstain': 'abstain'}
    if raw in mapping:
        return mapping[raw]
    if value.get('__type__') in ('Unavailable', 'CompileFailure', 'CompilationFailure'):
        return 'abstain'
    raise ValueError('Unsupported decision type: ' + str(value.get('__type__', raw)))


def geometry(hull, phi, coefficients):
    result = dict(saved_hull=hull, fallback=hull.get('status') == 'FULL_FALLBACK',
                  returned_set_contains_truth=None, retained_cells_contain_truth=None,
                  unresolved_cell_contains_truth=None, geometry=None)
    state = hull.get('status')
    if state == 'FULL_FALLBACK':
        result['returned_set_contains_truth'] = True
        result['coverage_kind'] = 'fallback_inclusion_only'
        return result
    if state not in ('OUTER_HULL', 'EMPTY_CERTIFIED'):
        raise ValueError('Unsupported source-set status: ' + str(state))
    def contains(parts):
        return any(exact(p['lower']) <= phi <= exact(p['upper']) for p in parts)
    covered = contains(hull.get('retained_cells', []))
    result.update(returned_set_contains_truth=covered, retained_cells_contain_truth=covered,
                  unresolved_cell_contains_truth=contains(hull.get('unresolved_cells', [])),
                  coverage_kind='retained_outer_cells')
    if state == 'EMPTY_CERTIFIED':
        return result
    lo, hi = exact(hull['lower']), exact(hull['upper'])
    if not 0 <= lo <= hi <= 1:
        raise ValueError('Source endpoints are outside the declared range.')
    if hi == 1:
        return result
    tl, tu = (1 + lo) / (1 - lo), (1 + hi) / (1 - hi)
    span = tu / tl
    members = []
    for coefficient in coefficients:
        a = exact(coefficient)
        if not 0 <= a < 1:
            raise ValueError('Invalid declared bank coefficient.')
        ta = (1 + a) / (1 - a)
        envelope = max((tu / ta)**2, (ta / tl)**2)
        members.append(dict(coefficient=coefficient, source_envelope=encoded(envelope),
                            placement_factor=encoded(envelope / span)))
    best = min((exact(m['source_envelope']) for m in members), default=None)
    with localcontext() as context:
        context.prec = 50
        logs = dict(log_persistence_span=str(dec(span).ln()),
                    bank_log_distance=str(dec(best / span).ln() / 2) if best else None)
    result['geometry'] = dict(T_lower=encoded(tl), T_upper=encoded(tu),
        width=encoded(hi - lo), continuous_best_envelope=encoded(span),
        bank_best_envelope=encoded(best) if best else None,
        bank_placement_factor=encoded(best / span) if best else None,
        members=members, decimal_display=logs,
        display_scope='Decimal50 descriptive logarithms; exact ratios are authoritative')
    return result


def member_diagnostic(member, envelope=None, calibration=None, method=''):
    receipt = member.get('receipt', {})
    coefficient = member.get('coefficient', receipt.get('coefficient'))
    output = dict(saved_member=member, coefficient=coefficient,
                  available=member.get('__type__') == 'MemberDecision',
                  reason=member.get('reason'), source_envelope_saved=member.get('source_ratio'),
                  q_saved=member.get('covariance_ratio'), source_envelope_derived=envelope,
                  scale_denominator=receipt.get('scale_denominator'),
                  threshold_components={key: receipt.get(key) for key in
                    ('variance_scale', 'null_tolerance', 'sampling_radius', 'rounding_contrast',
                     'variance_log', 'tail_log', 'operator_factor', 'scale_mode',
                     'observed_variances', 'ideal_variance_upper', 'variances',
                     'threshold_excess_upper')},
                  margin=None, q_derived=None, mathematical_scale_display=None, missing_saved_fields=[])
    if output['available']:
        output['margin'] = encoded(exact(member['statistic_absolute']) - exact(member['threshold']['upper']))
    if envelope is not None and calibration and 'shape_upper' in calibration:
        factor = Q(method.split('two_sided_shape_', 1)[1]) if 'two_sided_shape_' in method else Q(1)
        value = max(Q(1), 2 * exact(envelope) * max(Q(1), exact(calibration['shape_upper'])) * factor)
        output['q_derived'] = encoded(value)
        output['q_derived_scope'] = 'Fixed formula from saved hull, calibration shape, and declared variant factor'
    for field in ('source_ratio', 'covariance_ratio', 'statistic_absolute', 'threshold'):
        if field not in member:
            output['missing_saved_fields'].append(field)
    return output


def classify_source(decision):
    members = decision.get('members', [])
    available = [m for m in members if m.get('__type__') == 'MemberDecision']
    expected = 'reject' if any(m['reject'] for m in available) else ('nonreject' if available else 'abstain')
    if status(decision) != expected:
        raise ValueError('The bank decision differs from its saved members.')
    if expected == 'reject':
        return 'rejection'
    source = decision.get('source', {})
    reason = source.get('reason', '')
    if source.get('status') == 'FULL_FALLBACK':
        return 'input_numerical' if numerical(reason) else 'no_finite_covariance'
    if source.get('status') == 'EMPTY_CERTIFIED':
        return 'empty_source_set'
    if source.get('upper') is not None and exact(source['upper']) == 1:
        return 'hull_reaching_one'
    if available:
        return 'available_insufficient_margin'
    reasons = [m.get('reason', '') for m in members]
    if any(numerical(reason) for reason in reasons):
        return 'input_numerical'
    if reasons and all(reason == 'nonpositive_scale_denominator' for reason in reasons):
        return 'no_positive_scale_denominator'
    calibration = decision.get('calibration', {})
    if calibration.get('__type__') == 'Unavailable' or 'no_finite_source_certificate' in reasons:
        return 'no_finite_covariance'
    raise ValueError('Cannot classify source refusal from its saved receipts: ' + str(reasons))


def cycle_components(container, method, decision):
    compiled = container.get('compilation', {})
    bounds = container.get('bounds', {})
    perturbation = container.get('perturbation', {})
    if not all(key in compiled for key in ('design', 'centered_energy')) or 'covariance' not in bounds:
        return dict(supported=False, reason='Required saved design, energy, or bounds are absent')
    k = list(map(exact, bounds['covariance']))
    b = list(map(exact, bounds['bias']))
    energies = list(map(exact, compiled['centered_energy']))
    errors = list(map(exact, perturbation['error']))
    local = method.rsplit('/', 1) if method.startswith('gain_') else [method]
    if local[0].startswith('gain_'):
        gain = Q(local[0][5:])
        if local[-1] != 'inverse_transport':
            k[2] *= gain**2
            energies[2] *= gain**2
            errors[2] *= abs(gain)
            b[1] *= abs(gain)
            b[2] *= abs(gain)
    elif local[0].startswith('inflation_K'):
        ck, cb = local[0][11:].split('_B')
        k = [x * Q(ck) for x in k]
        b = [x * Q(cb) for x in b]
    elif local[0].startswith('underestimated_'):
        _, component, factor = local[0].split('_')
        if component == 'K':
            k = [x * Q(factor) for x in k]
        elif component == 'B':
            b = [x * Q(factor) for x in b]
        else:
            return dict(supported=False, reason='Unknown bound transform')
    elif local[0] not in ('diagonal', 'common', 'combined_frequencies'):
        return dict(supported=False, reason='Unknown cycle method context')
    if local[-1] == 'common':
        k, b = [max(k)] * 3, [max(b)] * 3
    design = compiled['design']
    count = 2 if local[-1] == 'combined_frequencies' else len(design['frequencies'])
    with localcontext() as context:
        context.prec = 50
        n, h = Decimal(design['n']), Decimal(2 * design['lag'] + 1)
        t = dec(Q(12 * count) / exact(design['alpha'])).ln()
        output = []
        for edge, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            scale = dec(k[i] * k[j]).sqrt()
            parts = dict(tail_bias=dec(b[edge]), centering=3 * scale * h / n,
                         stochastic_sqrt=2 * scale * (h * t / n).sqrt(),
                         stochastic_linear=Decimal(2).sqrt() * scale * h * t / n,
                         recording=h * (dec(errors[i]) * (dec(energies[j]) / n).sqrt()
                           + dec(errors[j]) * (dec(energies[i]) / n).sqrt()
                           + dec(errors[i] * errors[j])))
            output.append(dict(edge=[i, j], components={key: str(value) for key, value in parts.items()},
                               total_display=str(sum(parts.values()))))
    return dict(supported=True, scope='Decimal50 fixed-formula description; not saved component enclosures',
                frequency_count=count, edges=output,
                transformed_bounds=dict(covariance=list(map(encoded, k)), bias=list(map(encoded, b))),
                missing_saved_fields=['individual radius component enclosures'])


def flatten(container):
    result = dict(container['methods'])
    for name, value in container.get('paired', {}).items():
        if name.startswith('gain_'):
            for subname, decision in value.items():
                key = name + '/' + subname
                if key in result:
                    raise ValueError('Duplicate method path.')
                result[key] = decision
        else:
            if name in result:
                raise ValueError('Duplicate method path.')
            result[name] = value
    return result


def normalize(cell, replicate, prefix, container):
    result = []
    for method, decision in flatten(container).items():
        row = dict(cell=cell['id'], replicate=replicate, family=cell['family'], N=cell['N'],
                   method=prefix + method, status=status(decision), saved_decision=decision,
                   first_failure=None, source=None, members=[], cycle=None,
                   context={key: container[key] for key in ('calibration', 'recording_identity',
                     'compilation_receipt', 'compilation_failure', 'bounds', 'perturbation', 'recorder',
                     'limits', 'original_compilation_identity')
                     if key in container})
        if decision.get('__type__') == 'SourceBankResult':
            coefficients = [m.get('coefficient', m.get('receipt', {}).get('coefficient')) for m in decision['members']]
            if any(a is None for a in coefficients):
                raise ValueError('A source member lacks its coefficient.')
            row['source'] = geometry(decision['source'], Q(cell['parameters']['phi']), coefficients)
            geo = row['source']['geometry']
            envelopes = [m['source_envelope'] for m in geo['members']] if geo else [None] * len(coefficients)
            row['members'] = [member_diagnostic(m, e, decision.get('calibration'), method)
                              for m, e in zip(decision['members'], envelopes)]
            row['first_failure'] = classify_source(decision)
            row['source_scope'] = True
            with localcontext() as context:
                context.prec = 50
                for detail in row['members']:
                    q = detail['q_saved'] or detail['q_derived']
                    if q is not None:
                        length = cell['N'] - 1
                        ratio = dec(exact(q) / length)
                        log = Decimal(200 * len(coefficients)).ln()
                        value = 1 - ratio - 2 * (ratio * log).sqrt()
                        detail['mathematical_scale_display'] = dict(value=str(value), positive=value > 0,
                            retained_length=length, bank_size=len(coefficients),
                            scope='Decimal50 fixed-formula diagnostic; saved interval controls availability')
        elif decision.get('__type__') == 'MemberDecision':
            row['members'] = [member_diagnostic(decision)]
        if 'frequencies' in decision:
            row['cycle'] = dict(saved_frequencies=decision['frequencies'],
                                radius_decomposition=cycle_components(container, method, decision),
                                saved_compilation=container.get('compilation'))
        result.append(row)
    return result


def gate(run, analysis, registry_path, freeze_path, authorization=None, fixture=False):
    if not (run / 'RUN_COMPLETE.json').is_file():
        raise ValueError('RUN_COMPLETE.json is required before reading decisions.')
    if not (analysis / 'ANALYSIS.json').is_file() or any(not (analysis / f).is_file() for f in ANALYSIS_FILES):
        raise ValueError('The complete analysis marker and tables are required.')
    registry = read_json(registry_path)
    expected = sum(cell['replicates'] for cell in registry['cells'])
    start, done, report = [read_json(p) for p in (run / 'RUN_START.json', run / 'RUN_COMPLETE.json', analysis / 'ANALYSIS.json')]
    identity = digest(registry_path)
    if start['registry_sha256'] != identity or report['registry_sha256'] != identity:
        raise ValueError('Registry identities differ.')
    if start['freeze_sha256'] != digest(freeze_path):
        raise ValueError('Freeze identity differs.')
    if done['records'] != expected or report['records'] != expected or report.get('failures') != 0:
        raise ValueError('Run or analysis is incomplete.')
    if report['cells'] != len(registry['cells']):
        raise ValueError('Analysis cell count differs.')
    if fixture:
        if registry.get('status') != 'DETERMINISTIC_FIXTURE':
            raise ValueError('Fixture mode cannot read a scientific registry.')
    else:
        if authorization is None:
            raise ValueError('Parent authorization is required for scientific extraction.')
        allowed = read_json(authorization)
        if (allowed.get('authorized') is not True or allowed.get('registry_sha256') != identity
                or Path(allowed.get('run_path', '')).resolve() != run.resolve()
                or allowed.get('purpose') != 'presentation-only diagnostics'):
            raise ValueError('The authorization does not match this extraction.')
    for filename in ANALYSIS_FILES:
        with (analysis / filename).open(newline='') as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if filename == 'source_sets.csv' and len(rows) != report['source_sets']:
                raise ValueError('Source analysis is incomplete.')
            if filename == 'cell_summary.csv' and sum(int(row['n']) for row in rows) != report['decisions']:
                raise ValueError('Cell analysis is incomplete.')
    return registry, start, report


def extract(run, analysis, registry_path, freeze_path, output, authorization=None, fixture=False):
    registry, start, report = gate(run, analysis, registry_path, freeze_path, authorization, fixture)
    if output.exists():
        raise FileExistsError(output)
    manifest = []
    def bind(path, role):
        entry = dict(path=str(path.resolve()), sha256=digest(path), bytes=path.stat().st_size, role=role)
        manifest.append(entry)
        return entry
    for path in (registry_path, freeze_path, run / 'RUN_START.json', run / 'RUN_COMPLETE.json',
                 analysis / 'ANALYSIS.json', *[analysis / f for f in ANALYSIS_FILES], Path(__file__)):
        bind(path, 'control_or_analysis')
    if authorization:
        bind(authorization, 'authorization')
    root = Path(__file__).resolve().parent.parent
    formula_binding = Path(__file__).with_name('FORMULA_BINDING_001.json')
    bind(formula_binding, 'formula_contract')
    expected_formulas = read_json(formula_binding)
    freeze_entries = {entry['path']: entry['sha256'] for entry in read_json(freeze_path).get('files', [])}
    for relative in FORMULA_FILES:
        item = bind(root / relative, 'formula_source')
        if item['sha256'] != expected_formulas[relative]:
            raise ValueError('A formula source differs from the extraction contract.')
        if not fixture and freeze_entries.get(relative) != item['sha256']:
            raise ValueError('A formula source differs from the scientific freeze.')
    expected_jobs = {f"{c['id']}__{r:04d}" for c in registry['cells'] for r in range(c['replicates'])}
    if {p.name for p in run.iterdir() if p.is_dir()} != expected_jobs:
        raise ValueError('The run directory set differs from the registry.')
    output.mkdir()
    counts = Counter()
    coverage_counts = Counter()
    source_groups = set()
    coverage_groups = set()
    keys = {}
    records = 0
    try:
        with gzip.open(output / 'diagnostics.jsonl.gz', 'wt') as stream:
            for cell in registry['cells']:
                for replicate in range(cell['replicates']):
                    job = run / f"{cell['id']}__{replicate:04d}"
                    if (job / 'FAILED.json').exists():
                        raise ValueError('A failed job is present.')
                    completion = read_json(job / 'COMPLETE.json')
                    if (completion['cell'], completion['replicate']) != (cell['id'], replicate):
                        raise ValueError('Job identity differs.')
                    bind(job / 'COMPLETE.json', 'job_completion')
                    entries = {e['path']: e for e in completion['files']}
                    if len(entries) != len(completion['files']) or any(Path(name).name != name for name in entries):
                        raise ValueError('Invalid completion file names.')
                    if set(entries) != {p.name for p in job.iterdir() if p.is_file()} - {'COMPLETE.json'}:
                        raise ValueError('Job file inventory differs.')
                    def read_receipt(name):
                        if name not in entries:
                            raise ValueError('Required receipt missing from completion: ' + name)
                        path = job / name
                        entry = bind(path, 'decision_receipt')
                        if entry['sha256'] != entries[name]['sha256']:
                            raise ValueError('A decision receipt hash differs: ' + name)
                        return read_json(path)
                    job_start = read_receipt('START.json')
                    if job_start['freeze_sha256'] != start['freeze_sha256']:
                        raise ValueError('Job freeze differs.')
                    primary = read_receipt('PRIMARY.json.gz')
                    rows = normalize(cell, replicate, '', primary)
                    rows += normalize(cell, replicate, 'baseline/', {'methods': read_receipt('BASELINES.json')})
                    if 'NUMERICAL_VARIANTS.json.gz' in entries:
                        for name, container in read_receipt('NUMERICAL_VARIANTS.json.gz').items():
                            rows += normalize(cell, replicate, 'numerical/' + name + '/', container)
                    for name in sorted(entries):
                        if name.startswith('FRACTIONAL_') and name.endswith('.json.gz'):
                            rows += normalize(cell, replicate, name.split('.')[0].lower() + '/', read_receipt(name))
                        if name.startswith('RECORDER_') and name.endswith('.json.gz'):
                            recorded = read_receipt(name)
                            container = dict(recorded['result'], recorder=recorded['recorder'])
                            if 'methods' not in container:
                                container['methods'] = {key: recorded['result'] for key in flatten(primary)}
                            extra = normalize(cell, replicate, name.split('.')[0].lower() + '/', container)
                            if recorded['result'].get('reason') == 'recorder_saturation_without_error_certificate':
                                for row in extra:
                                    row['first_failure'] = 'input_numerical'
                                    if flatten(primary)[row['method'].split('/', 1)[1]].get('__type__') == 'SourceBankResult':
                                        row['source_scope'] = True
                                        row['source_coverage_unavailable'] = 'No valid recorder certificate; no source method was evaluated'
                            rows += extra
                    for row in rows:
                        key = (row['cell'], row['replicate'], row['method'])
                        if key in keys:
                            raise ValueError('Duplicate diagnostic identity.')
                        keys[key] = row['status']
                        if row['first_failure']:
                            source_groups.add((row['cell'], row['method']))
                            counts[(row['cell'], row['method'], row['first_failure'])] += 1
                        if row.get('source_scope'):
                            coverage_groups.add((row['cell'], row['method']))
                            coverage_counts[(row['cell'], row['method'], 'all_declared_outcomes')] += 1
                            if row['source'] is None:
                                coverage_counts[(row['cell'], row['method'], 'source_not_evaluated')] += 1
                        if row['source'] is not None:
                            for field in ('fallback', 'returned_set_contains_truth',
                                          'retained_cells_contain_truth', 'unresolved_cell_contains_truth'):
                                if row['source'][field] is True:
                                    coverage_counts[(row['cell'], row['method'], field)] += 1
                        stream.write(json.dumps(row, sort_keys=True) + '\n')
                    manifest.append(dict(path=str(job.resolve()), role='complete_inventory_including_unread_files',
                        files=completion['files']))
                    records += 1
        analysis_statuses = {}
        with (analysis / 'decisions.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'cell', 'replicate', 'method', 'status'} <= set(reader.fieldnames or ()):
                raise ValueError('The analysis decision identity and status columns are required.')
            for row in reader:
                key = (row['cell'], int(row['replicate']), row['method'])
                if key in analysis_statuses:
                    raise ValueError('Duplicate analysis decision identity.')
                if row['status'] not in ('reject', 'nonreject', 'abstain'):
                    raise ValueError('Missing or unknown analysis decision status: ' + str(key))
                analysis_statuses[key] = row['status']
        if analysis_statuses.keys() != keys.keys() or report['decisions'] != len(keys):
            raise ValueError('Diagnostic methods differ from the completed analysis.')
        for key, receipt_status in keys.items():
            if analysis_statuses[key] != receipt_status:
                raise ValueError('The analysis status differs from its saved receipt: ' + str(key))
        rows = [dict(cell=c, method=m, first_failure=k, count=counts[(c, m, k)])
                for c, m in sorted(source_groups) for k in CATEGORIES]
        coverage = [dict(cell=c, method=m, field=k, count=coverage_counts[(c, m, k)])
                    for c, m in sorted(coverage_groups) for k in ('all_declared_outcomes', 'source_not_evaluated',
                    'fallback', 'returned_set_contains_truth',
                    'retained_cells_contain_truth', 'unresolved_cell_contains_truth')]
        (output / 'source_coverage_counts.json').write_text(json.dumps(coverage, indent=2) + '\n')
        (output / 'first_failure_counts.json').write_text(json.dumps(rows, indent=2) + '\n')
        for item in manifest:
            if 'sha256' in item and digest(Path(item['path'])) != item['sha256']:
                raise ValueError('An input changed during extraction.')
        (output / 'INPUT_BINDING.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result = dict(status='COMPLETE', records=records, methods=len(keys), fixture=fixture,
            observations_read=False, methods_reevaluated=False, random_draws=0,
            output_files=[dict(path=p.name, sha256=digest(p)) for p in sorted(output.iterdir())])
        (output / 'EXTRACTION_COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    except Exception as error:
        (output / 'EXTRACTION_FAILED.json').write_text(json.dumps(dict(error=repr(error)), indent=2) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'analysis', 'registry', 'freeze', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    extract(args.run, args.analysis, args.registry, args.freeze, args.output, args.authorization, args.fixture)


if __name__ == '__main__':
    main()
