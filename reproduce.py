"""Reproduce displays from complete saved analysis without generating observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
ANALYSIS_FILES = ('ANALYSIS.json', 'decisions.csv', 'source_sets.csv', 'source_summary.csv',
                  'cell_summary.csv', 'paired_comparisons.csv')
DIAGNOSTIC_FILES = ('diagnostics.jsonl.gz', 'first_failure_counts.json', 'source_coverage_counts.json')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2) + '\n')


def contained(path):
    path = path.resolve()
    if PACKAGE == path or PACKAGE in path.parents:
        raise ValueError('Use a reproduction output directory outside the package.')
    return path


def prepare_view(work):
    registry = PACKAGE / 'study/REGISTRY_001.json'
    analysis = PACKAGE / 'study/analysis_001'
    provenance = json.loads((PACKAGE / 'diagnostics/PROVENANCE.json').read_text())
    expected = {'records': 20800, 'methods': 627400, 'fixture': False}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved diagnostic scope differs from the complete study.')
    (work / 'REGISTRY_001.json').symlink_to(registry)
    (work / 'analysis').symlink_to(analysis, target_is_directory=True)
    view = work / 'diagnostics'
    view.mkdir()
    outputs = []
    for name in DIAGNOSTIC_FILES:
        (view / name).symlink_to(PACKAGE / 'diagnostics' / name)
        outputs.append({'path': name, 'sha256': provenance['outputs'][name]})
    inputs = [registry, *[analysis / name for name in ANALYSIS_FILES]]
    binding = [{'path': str(p.resolve()), 'sha256': digest(p)} for p in inputs]
    write_json(view / 'INPUT_BINDING.json', binding)
    outputs.append({'path': 'INPUT_BINDING.json', 'sha256': digest(view / 'INPUT_BINDING.json')})
    marker = dict(status='COMPLETE', records=20800, methods=627400, fixture=False,
                  observations_read=False, methods_reevaluated=False, random_draws=0,
                  output_files=outputs, receipt_kind='SAVED_DIAGNOSTICS_PATH_VIEW',
                  extraction_executed=False, included_payloads_rehashed=False,
                  original_extraction_marker_sha256=provenance['original_extraction_marker_sha256'],
                  scope='Relative package input view of saved diagnostics. This does not repeat extraction.')
    write_json(view / 'EXTRACTION_COMPLETE.json', marker)


def execute(stage, work):
    work = contained(work)
    code = PACKAGE / 'code'
    registry = PACKAGE / 'study/REGISTRY_001.json'
    analysis = PACKAGE / 'study/analysis_001'
    if stage == 'operating':
        work.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(code / 'operating_display/present_study.py'),
                   '--registry', str(registry), '--run', str(PACKAGE / 'study/run_001'),
                   '--analysis', str(analysis), '--output', str(work / 'presentation')]
    elif stage == 'diagnostics':
        if not (work / 'presentation/PRESENTATION.json').is_file():
            raise ValueError('Run the operating stage before the diagnostics stage.')
        prepare_view(work)
        command = [sys.executable, str(code / 'diagnostic_display/present_diagnostics.py'),
                   '--registry', str(work / 'REGISTRY_001.json'), '--analysis', str(work / 'analysis'),
                   '--presentation', str(work / 'presentation'), '--diagnostics', str(work / 'diagnostics'),
                   '--output', str(work / 'output')]
    else:
        if not (work / 'output/DISPLAY_COMPLETE.json').is_file():
            raise ValueError('Run the diagnostics stage before the tables stage.')
        command = [sys.executable, str(code / 'table_selection/select_tables.py'),
                   '--input-root', str(work), '--output', str(work / 'tables')]
    record = dict(stage=stage, command=command, scientific_scope='Saved-analysis displays only.',
                  observations_generated=0, methods_reevaluated=False, extraction_executed=False)
    write_json(work / (stage + '_command.json'), record)
    with (work / (stage + '.stdout')).open('x') as out, (work / (stage + '.stderr')).open('x') as err:
        result = subprocess.run(command, stdout=out, stderr=err,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    record['exit_code'] = result.returncode
    write_json(work / (stage + '_result.json'), record)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('operating', 'diagnostics', 'tables'), required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    execute(args.stage, args.work.resolve())


if __name__ == '__main__':
    main()
