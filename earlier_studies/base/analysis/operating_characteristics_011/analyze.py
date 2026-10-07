"""Report every fixed cell, including abstentions and missing outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from scipy.stats import beta


def interval(count, total, alpha):
    return [0.0 if count == 0 else float(beta.ppf(alpha/2,count,total-count+1)),
            1.0 if count == total else float(beta.ppf(1-alpha/2,count+1,total-count))]


def main(directory):
    summary = directory/'ANALYSIS.json'
    if summary.exists():
        raise RuntimeError('Preserve the existing analysis. Use a separate attempt.')
    parameters = json.loads((directory/'raw_inputs/PARAMETERS.json').read_text())
    designs = parameters['design']
    cells, paired = [], []
    results = {}
    for replicate in range(400):
        for design in designs:
            number = design['design_id']
            path = directory/'results'/f'rep{replicate:03d}_design{number:02d}.json'
            result = json.loads(path.read_text()) if path.exists() else None
            results[number,replicate] = result
    for design in designs:
        number = design['design_id']
        lanes = ['primary'] if number == 7 else ['primary','marginal','prefilter']
        for length in design['lengths']:
            for lane in lanes:
                counts = dict(rejections=0,nonrejections=0,abstentions=0,missing_or_errors=0,
                              scale_coverage_available=0,scale_coverage_both=0)
                for replicate in range(400):
                    result = results[number,replicate]
                    value = next((row for row in result['outcomes'] if row['lane'] == lane
                                 and row['original_observed_length'] == length),None) if result else None
                    if value is None or (result and (result.get('resource_error') or result.get('error_type'))):
                        counts['missing_or_errors'] += 1
                        continue
                    if value['status'] == 'ABSTAIN':
                        counts['abstentions'] += 1
                    elif value['status'] == 'DECISION':
                        counts['rejections' if value['rejection'] else 'nonrejections'] += 1
                    else:
                        raise ValueError('A stored outcome has an unsupported status.')
                    if value.get('scale_coverage') is not None:
                        counts['scale_coverage_available'] += 1
                        counts['scale_coverage_both'] += int(all(value['scale_coverage']))
                complete = counts['missing_or_errors'] == 0
                cells.append(dict(design_id=number,hypothesis=design['hypothesis'],
                     lane=lane,original_observed_length=length,
                     retained_length=length-(lane == 'prefilter'),replicates=400,
                     complete=complete,**counts,rejection_rate=counts['rejections']/400,
                     pointwise_95=interval(counts['rejections'],400,.05) if complete else None,
                     simultaneous_95=interval(counts['rejections'],400,.05/57) if complete else None))
            for lane in lanes[1:]:
                gain,loss,missing = 0,0,0
                for replicate in range(400):
                    result = results[number,replicate]
                    rows = {row['lane']:row for row in result['outcomes']
                            if row['original_observed_length'] == length} if result else {}
                    if 'primary' not in rows or lane not in rows or result.get('resource_error') or result.get('error_type'):
                        missing += 1
                        continue
                    primary = bool(rows['primary']['rejection'])
                    control = bool(rows[lane]['rejection'])
                    gain += int(control and not primary)
                    loss += int(primary and not control)
                difference_interval = None
                if missing == 0:
                    gl,gu = interval(gain,400,.025)
                    ll,lu = interval(loss,400,.025)
                    difference_interval = [max(-1,gl-lu),min(1,gu-ll)]
                paired.append(dict(design_id=number,original_observed_length=length,
                                   control=lane,gain=gain,loss=loss,missing_or_errors=missing,
                                   rejection_difference=(gain-loss)/400,
                                   conservative_pointwise_95=difference_interval,
                                   broad_cap_invariant_pass=(loss == 0) if lane == 'marginal' else None))
    calibration = []
    for design in designs:
        rows = [results[design['design_id'],r] for r in range(400)]
        available = [row for row in rows if row and 'calibration_coverage' in row]
        calibration.append(dict(design_id=design['design_id'],available=len(available),
                                missing=400-len(available),
                                channel_coverage_counts=[sum(int(row['calibration_coverage'][i])
                                                         for row in available) for i in [0,1]],
                                both_coverage_count=sum(int(all(row['calibration_coverage']))
                                                        for row in available)))
    if len(cells) != 57 or len(paired) != 36:
        raise ValueError('The analysis does not cover the declared fixed grid.')
    report = dict(status='COMPLETE' if all(row['complete'] for row in cells) else 'INCOMPLETE',
                  cells=cells,paired_comparisons=paired,calibration_coverage=calibration,
                  interpretation='These are finite-design rates. They do not establish uniform size or power. '
                  'Abstention is a separate status. Supplied-marginal controls use additional model information. '
                  'Scale coverage among available decisions is a diagnostic, not unconditional event coverage.',
                  parameters_sha256=hashlib.sha256((directory/'raw_inputs/PARAMETERS.json').read_bytes()).hexdigest(),
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    summary.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (directory/'rates.csv').open('w',newline='') as stream:
        fields = list(cells[0])
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in cells:
            writer.writerow({key:json.dumps(value) if isinstance(value,list) else value
                             for key,value in row.items()})
    print(json.dumps({'status':report['status'],'cells':len(cells),'paired_comparisons':len(paired)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    main(parser.parse_args().directory)
