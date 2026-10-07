"""Summarize every declared record and all paired decisions."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest


def read_json(path):
    if path.suffix == '.gz':
        with gzip.open(path,'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def number(value):
    if isinstance(value,dict) and '__fraction__' in value:
        return float(Fraction(value['__fraction__']))
    return value


def exact(value):
    return Fraction(value['__fraction__'])


def endpoint(value,side):
    return number(value[side]) if isinstance(value,dict) and side in value else None


def status(decision):
    if decision.get('__type__') == 'MemberDecision':
        return 'reject' if decision['reject'] else 'nonreject'
    raw=decision.get('decision',decision.get('status'))
    mapping={'REJECT':'reject','DO_NOT_REJECT':'nonreject','ABSTAIN':'abstain',
             'reject':'reject','nonreject':'nonreject','abstain':'abstain'}
    if raw in mapping:
        return mapping[raw]
    if decision.get('__type__') in ('Unavailable','CompileFailure','CompilationFailure'):
        return 'abstain'
    raise ValueError('Unrecognized decision: '+str(raw or decision.get('__type__')))


def clopper_pearson(count,total,confidence=.95):
    if not 0 <= count <= total or total < 1:
        raise ValueError('A binomial interval needs 0 <= count <= total and total > 0.')
    tail=(1-confidence)/2
    return [0. if count==0 else float(beta.ppf(tail,count,total-count+1)),
            1. if count==total else float(beta.ppf(1-tail,count+1,total-count))]


def paired_interval(first_only,second_only,total):
    first=clopper_pearson(first_only,total,.975)
    second=clopper_pearson(second_only,total,.975)
    return [max(-1.,first[0]-second[1]),min(1.,first[1]-second[0])]


def flatten_methods(primary):
    result=dict(primary['methods'])
    for name,value in primary.get('paired',{}).items():
        if name.startswith('gain_'):
            result.update({name+'/'+method:decision for method,decision in value.items()})
        else:
            result[name]=value
    return result


def all_methods(job,primary):
    result=flatten_methods(primary)
    result.update({'baseline/'+name:value for name,value in read_json(job/'BASELINES.json').items()})
    numerical=job/'NUMERICAL_VARIANTS.json.gz'
    if numerical.exists():
        for variant,values in read_json(numerical).items():
            result.update({'numerical/'+variant+'/'+name:value for name,value in values['methods'].items()})
    for path in sorted(job.glob('FRACTIONAL_*.json.gz')):
        prefix=path.name.split('.')[0].lower()
        result.update({prefix+'/'+name:value for name,value in flatten_methods(read_json(path)).items()})
    for path in sorted(job.glob('RECORDER_*.json.gz')):
        prefix=path.name.split('.')[0].lower()
        record=read_json(path)['result']
        if 'methods' in record:
            result.update({prefix+'/'+name:value for name,value in flatten_methods(record).items()})
        else:
            result.update({prefix+'/'+name:record for name in flatten_methods(primary)})
    return result


def all_source_sets(job,primary):
    result=dict(primary.get('source_sets',{}))
    numerical=job/'NUMERICAL_VARIANTS.json.gz'
    if numerical.exists():
        for variant,record in read_json(numerical).items():
            result.update({variant+'/'+name:hull for name,hull in record['source_sets'].items()})
    for path in sorted(job.glob('FRACTIONAL_*.json.gz')):
        prefix=path.name.split('.')[0].lower()
        result.update({prefix+'/'+name:hull for name,hull in read_json(path).get('source_sets',{}).items()})
    for path in sorted(job.glob('RECORDER_*.json.gz')):
        prefix=path.name.split('.')[0].lower()
        record=read_json(path)['result']
        if 'source_sets' in record:
            result.update({prefix+'/'+name:hull for name,hull in record['source_sets'].items()})
        elif record.get('reason')=='recorder_saturation_without_error_certificate':
            fallback=dict(status='FULL_FALLBACK',lower={'__fraction__':'0'},upper={'__fraction__':'1'},
                          reason='recorder_saturation_without_error_certificate')
            result.update({prefix+'/'+name:fallback for name in primary.get('source_sets',{})})
    return result


def diagnostics(decision):
    output=dict(reason=decision.get('reason',''))
    members=decision.get('members',[decision] if decision.get('__type__')=='MemberDecision' else [])
    available=[member for member in members if member.get('__type__')=='MemberDecision']
    output['available_members']=len(available) if members else None
    if available:
        selected=max(available,key=lambda member:exact(member['statistic_absolute'])-exact(member['threshold']['upper']))
        receipt=selected['receipt']
        statistic=number(selected['statistic_absolute'])
        threshold=endpoint(selected['threshold'],'upper')
        output.update(selected_coefficient=number(selected['coefficient']),statistic=statistic,threshold=threshold,
                      margin=float(exact(selected['statistic_absolute'])-exact(selected['threshold']['upper'])),statistic_threshold_ratio=statistic/threshold if threshold else None,
                      source_envelope=number(selected['source_ratio']),q=number(selected['covariance_ratio']),
                      scale_denominator=endpoint(receipt.get('scale_denominator'),'lower'),
                      variance_scale=endpoint(receipt.get('variance_scale'),'upper'),
                      phase_allowance=endpoint(receipt.get('null_tolerance'),'upper'),
                      sampling_radius=endpoint(receipt.get('sampling_radius'),'upper'),
                      rounding_allowance=endpoint(receipt.get('rounding_contrast'),'upper'))
        output['minimum_available_envelope']=min(number(member['source_ratio']) for member in available)
    elif members:
        output['reason']=';'.join(sorted({member.get('reason','') for member in members}))
    frequencies=decision.get('frequencies',[])
    if frequencies:
        margins=[]
        for entry in frequencies:
            if 'absolute_imaginary_product' in entry:
                raw_statistic=entry['absolute_imaginary_product']
                statistic=number(raw_statistic) if '__fraction__' in raw_statistic else endpoint(raw_statistic,'lower')
            else:
                statistic=endpoint(entry.get('absolute_imaginary'),'lower')
            threshold=endpoint(entry.get('threshold'),'upper')
            if statistic is not None and threshold is not None:
                margins.append((statistic-threshold,statistic,threshold))
        if margins:
            margin,statistic,threshold=max(margins)
            output.update(statistic=statistic,threshold=threshold,margin=margin,
                          statistic_threshold_ratio=statistic/threshold if threshold else None)
    if not members and not frequencies:
        for key in ('statistic','threshold','pvalue','phase_tolerance','imaginary_coherency'):
            if key in decision:
                output[key]=number(decision[key])
    return output


def source_diagnostics(cell,replicate,name,hull):
    row=dict(cell=cell['id'],replicate=replicate,method=name,status=hull['status'])
    lower,upper=number(hull.get('lower')),number(hull.get('upper'))
    phi=Fraction(cell['parameters']['phi'])
    components=hull.get('retained_cells',[])
    if hull['status']=='FULL_FALLBACK':
        coverage=True
    else:
        coverage=any(exact(part['lower'])<=phi<=exact(part['upper']) for part in components)
    finite='upper' in hull and exact(hull['upper'])<1
    row.update(lower=lower,upper=upper,coverage=coverage,finite_endpoint=finite,
               components=len(components),unresolved_components=len(hull.get('unresolved_cells',[])),
               reason=hull.get('reason',''))
    if finite:
        lo,hi=exact(hull['lower']),exact(hull['upper'])
        tl,tu=(1+lo)/(1-lo),(1+hi)/(1-hi)
        span=float(tu/tl)
        row.update(width=float(hi-lo),persistence_span_ratio=span,log_persistence_width=float(np.log(span)))
        for label,bank in (('fixed',[Fraction(x) for x in ('0','1/2','4/5','9/10','97/100','199/200')]),
                           ('length_dependent',[1-Fraction(1,2**j) for j in range(math.ceil(math.log2(cell['N']-1))+1)])):
            envelope=min(max((tu/((1+a)/(1-a)))**2,(((1+a)/(1-a))/tl)**2) for a in bank)
            row[label+'_minimum_envelope']=float(envelope)
            row[label+'_placement_factor']=float(envelope/(tu/tl))
    return row


def write_csv(path,rows):
    names=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['cell'],row['method'])].append(row)
    result=[]
    for (cell,method),values in sorted(grouped.items()):
        n=len(values)
        counts={label:sum(row['status']==label for row in values) for label in ('reject','nonreject','abstain')}
        if sum(counts.values())!=n:
            raise ValueError('A method has an uncounted outcome.')
        available=counts['reject']+counts['nonreject']
        interval=clopper_pearson(counts['reject'],n)
        row=dict(cell=cell,method=method,n=n,**counts,rejection_fraction=counts['reject']/n,
                 rejection_lower=interval[0],rejection_upper=interval[1],available=available,
                 rejection_given_available=counts['reject']/available if available else None)
        for key in ('threshold','margin','statistic_threshold_ratio','source_envelope','q','scale_denominator',
                    'variance_scale','phase_allowance','sampling_radius','rounding_allowance'):
            finite=[value[key] for value in values if value.get(key) is not None]
            if finite:
                quantiles=np.quantile(finite,[0,.25,.5,.75,1])
                row.update({key+'_'+label:float(value) for label,value in zip(('min','q25','median','q75','max'),quantiles)})
                row[key+'_n']=len(finite)
        result.append(row)
    return result


def pair_summaries(rows):
    cells=defaultdict(lambda:defaultdict(dict))
    for row in rows:
        cells[row['cell']][row['method']][row['replicate']]=row['status']
    output=[]
    for cell,methods in sorted(cells.items()):
        for first,second in itertools.combinations(sorted(methods),2):
            a,b=methods[first],methods[second]
            if a.keys()!=b.keys():
                raise ValueError('Paired methods have different record sets.')
            n=len(a)
            ab=sum(a[i]=='reject' and b[i]!='reject' for i in a)
            ba=sum(b[i]=='reject' and a[i]!='reject' for i in a)
            interval=paired_interval(ab,ba,n)
            output.append(dict(cell=cell,first=first,second=second,n=n,first_only=ab,second_only=ba,
                difference=(ab-ba)/n,lower=interval[0],upper=interval[1],
                discordance_pvalue=float(binomtest(ab,ab+ba,.5).pvalue) if ab+ba else 1.,
                status_disagreements=sum(a[i]!=b[i] for i in a)))
    return output


def source_summaries(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['cell'],row['method'])].append(row)
    output=[]
    for (cell,method),values in sorted(grouped.items()):
        n=len(values)
        covered=sum(row['coverage'] for row in values)
        finite=sum(row['finite_endpoint'] for row in values)
        interval=clopper_pearson(covered,n)
        summary=dict(cell=cell,method=method,n=n,covered=covered,coverage=covered/n,
                     coverage_lower=interval[0],coverage_upper=interval[1],finite_endpoint=finite,
                     finite_fraction=finite/n,empty=sum(row['status']=='EMPTY_CERTIFIED' for row in values),
                     fallback=sum(row['status']=='FULL_FALLBACK' for row in values))
        for key in ('width','persistence_span_ratio','log_persistence_width','fixed_minimum_envelope',
                    'fixed_placement_factor','length_dependent_minimum_envelope','length_dependent_placement_factor'):
            data=[row[key] for row in values if key in row]
            if data:
                summary.update({key+'_'+label:float(value) for label,value in
                    zip(('min','q25','median','q75','max'),np.quantile(data,[0,.25,.5,.75,1]))})
        output.append(summary)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    registry=read_json(args.registry)
    if not (args.run/'RUN_COMPLETE.json').exists():
        raise ValueError('Final analysis requires all declared records. Preserve failures separately.')
    from run_study import completed_jobs,file_hash
    start=read_json(args.run/'RUN_START.json')
    if file_hash(args.registry)!=start['registry_sha256']:
        raise ValueError('The analysis registry differs from the run.')
    done=completed_jobs(args.run,registry,start['freeze_sha256'])
    if len(done)!=sum(cell['replicates'] for cell in registry['cells']):
        raise ValueError('The run omits declared records.')
    args.output.mkdir(exist_ok=False)
    decisions=[]
    sources=[]
    for cell in registry['cells']:
        for replicate in range(cell['replicates']):
            job=args.run/('%s__%04d'%(cell['id'],replicate))
            primary=read_json(job/'PRIMARY.json.gz')
            for name,decision in all_methods(job,primary).items():
                decisions.append(dict(cell=cell['id'],replicate=replicate,family=cell['family'],
                    target=cell['target'],method=name,status=status(decision),**diagnostics(decision)))
            for name,hull in all_source_sets(job,primary).items():
                sources.append(source_diagnostics(cell,replicate,name,hull))
    write_csv(args.output/'decisions.csv',decisions)
    write_csv(args.output/'source_sets.csv',sources)
    write_csv(args.output/'source_summary.csv',source_summaries(sources))
    summaries=summarize(decisions)
    write_csv(args.output/'cell_summary.csv',summaries)
    write_csv(args.output/'paired_comparisons.csv',pair_summaries(decisions))
    (args.output/'ANALYSIS.json').write_text(json.dumps(dict(registry_sha256=file_hash(args.registry),
        records=len(done),decisions=len(decisions),source_sets=len(sources),cells=len(registry['cells']),
        uncertainty='Two-sided95% Clopper-Pearson. Paired differences subtract simultaneous97.5% intervals.',
        failures=0,conditional_fractions='Descriptive; not a conditional size guarantee'),indent=2)+'\n')


if __name__=='__main__':
    main()
