"""Analyze all planned outcomes with exact binomial intervals."""
from functools import lru_cache
from fractions import Fraction as Q
from math import comb
from collections import Counter
from contracts import load_calls
from pathlib import Path
from study_common import CELLS,LENGTHS,REPLICATES,read,save

@lru_cache(None)
def binomial_interval(count, total, alpha=Q(1,20), bits=40):
    if not 0 <= count <= total or total < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid binomial interval inputs.')
    denominator=1 << bits
    target=alpha/2
    powers=denominator**total
    def tail(integer, lower, upper):
        complement=denominator-integer
        return sum(comb(total,j)*integer**j*complement**(total-j) for j in range(lower,upper+1))
    lower=0
    if count:
        left,right=0,denominator
        while right-left>1:
            middle=(left+right)//2
            value=tail(middle,count,total)
            if value*target.denominator >= target.numerator*powers: right=middle
            else: left=middle
        lower=left
    upper=denominator
    if count<total:
        left,right=0,denominator
        while right-left>1:
            middle=(left+right)//2
            value=tail(middle,0,count)
            if value*target.denominator <= target.numerator*powers: right=middle
            else: left=middle
        upper=right
    return (Q(lower,denominator),Q(upper,denominator))


def missing_interval(count, missing, total, alpha):
    return (binomial_interval(count,total,alpha)[0],binomial_interval(count+missing,total,alpha)[1])


def distribution(values, missing=0):
    finite=sorted(value for value in values if value is not None)
    n=len(finite)
    counts=dict(planned=len(values),completed_count=len(values)-missing,defined_count=n,undefined_count=len(values)-missing-n,missing_count=missing)
    if not n: return counts
    def quantile(q):
        position=(n-1)*q
        low=position.numerator//position.denominator
        high=min(n-1,low+1)
        return finite[low]+(finite[high]-finite[low])*(position-low)
    return dict(**counts,minimum=finite[0],maximum=finite[-1],
                first_quartile=quantile(Q(1,4)),median=quantile(Q(1,2)),third_quartile=quantile(Q(3,4)))


def dominance(first, second, phi):
    if 'missing_status' in first or 'missing_status' in second:
        return dict(scope='missing_pair',checks=[],primary_only=None)
    p,c=first['result'],second['result']
    source=p['source']
    covered=source['status']=='CERTIFIED_HULL' and source['lower']<=phi<=source['upper']
    checks=[]
    causes=[]
    primary_only=p['decision']=='REJECT' and c['decision']!='REJECT'
    for index,(pm,cm) in enumerate(zip(p['members'],c['members'])):
        pa,ca='reject' in pm,'reject' in cm
        if covered and pa and ca:
            valid=(pm['source_ratio']>=cm['source_ratio'] and pm['threshold']['upper']>=cm['threshold']['upper']
                   and pm['statistic_absolute']==cm['statistic_absolute']
                   and pm['receipt']['signed_statistic']==cm['receipt']['signed_statistic']
                   and pm['receipt']['variances']==cm['receipt']['variances'] and p['calibration']==c['calibration'])
            checks.append(dict(member_index=index,passed=valid))
        if primary_only and pa and pm['reject']:
            reason='MISSED_SOURCE_COVERAGE' if not covered else ('CONTROL_MEMBER_UNAVAILABLE' if not ca else 'DEFECT')
            causes.append(dict(member_index=index,reason=reason,control_reason=cm.get('reason')))
    return dict(scope='completed_pair',source_covers_phi=covered,checks=checks,primary_only=primary_only,causes=causes)


def analyze(attempt):
    attempt=Path(attempt)
    method_rows=[]
    paired_rows=[]
    contract_issues=[]
    dominance_rows=[]
    for cell in CELLS:
        jobs=[]
        for replicate in range(REPLICATES):
            directory=attempt/'jobs'/f'r{replicate:03d}_c{cell["cell_id"]:02d}'
            calls,issues=load_calls(directory,cell['cell_id'],replicate)
            jobs.append(calls)
            contract_issues.extend(dict(cell_id=cell['cell_id'],replicate=replicate,**issue) for issue in issues)
        for length in LENGTHS:
            collected={lane:[calls[(length,lane)] for calls in jobs] for lane in ('primary','control')}
            for lane,calls in collected.items():
                counts={name:0 for name in ('REJECT','DO_NOT_REJECT','ABSTAIN','ERROR','UNEXECUTED')}
                for call in calls:
                    counts[call['missing_status'] if 'missing_status' in call else call['result']['decision']]+=1
                missing=counts['ERROR']+counts['UNEXECUTED']
                entry=dict(cell=cell,length=length,lane=lane,planned=REPLICATES,counts=counts,missing=missing,
                           empirical_rejection_range=(Q(counts['REJECT'],REPLICATES),Q(counts['REJECT']+missing,REPLICATES)),
                           rejection_pointwise_interval=missing_interval(counts['REJECT'],missing,REPLICATES,Q(1,20)),
                           rejection_simultaneous_interval=missing_interval(counts['REJECT'],missing,REPLICATES,Q(1,1440)),
                           abstention_pointwise_interval=missing_interval(counts['ABSTAIN'],missing,REPLICATES,Q(1,20)))
                diagnostics=[call.get('diagnostics') for call in calls]
                entry['continuous']={key:distribution([d.get(key) if d is not None else None for d in diagnostics],missing) for key in
                                     ('source_width','source_radius_width','source_acceptance_excess','calibration_squared_error','calibration_radius_upper')}
                for key in ('source_covers_phi','calibration_covers_coefficients'):
                    values=[d.get(key) if d is not None else None for d in diagnostics]
                    defined=[v for v in values if v is not None]
                    unknown=REPLICATES-len(defined)
                    successes=sum(defined)
                    entry[key]=dict(planned=REPLICATES,completed_count=REPLICATES-missing,defined_count=len(defined),
                        undefined_count=unknown-missing,missing_count=missing,successes=successes,
                        descriptive_defined_rate=Q(successes,len(defined)) if defined else None,
                        binary_completion_range=(Q(successes,REPLICATES),Q(successes+unknown,REPLICATES)),
                        binary_completion_outer_interval=missing_interval(successes,unknown,REPLICATES,Q(1,20)),
                        scope='No coverage event is defined for unavailable or supplied-parameter certificates. The outer interval spans all binary completions. Defined-only rates are descriptive; no selected-sample coverage guarantee is asserted.')
                entry['members']=[]
                for member in range(4):
                    items=[d['members'][member] if d is not None else None for d in diagnostics]
                    available=sum(x is not None and x['available'] for x in items)
                    reasons=Counter(x['reason'] for x in items if x is not None and not x['available'])
                    entry['members'].append(dict(member_index=member,available_count=available,
                        unavailable_count=sum(reasons.values()),unavailability_reasons=dict(reasons),
                        planned=REPLICATES,completed_count=REPLICATES-missing,missing_count=missing,
                        diagnostic_summaries={key:distribution([x.get(key) if x is not None else None for x in items],missing) for key in
                        ('observed_threshold_ratio','nominal_population_threshold_ratio','strict_margin','threshold_excess','statistic_loss')}))
                method_rows.append(entry)
            gain=loss=missing=0
            for replicate,(first,second) in enumerate(zip(collected['primary'],collected['control'])):
                audit=dominance(first,second,cell['phi'])
                dominance_rows.append(dict(cell_id=cell['cell_id'],replicate=replicate,length=length,**audit))
                if 'missing_status' in first or 'missing_status' in second: missing+=1;continue
                p=first['result']['decision']=='REJECT';c=second['result']['decision']=='REJECT'
                gain+=int(p and not c);loss+=int(c and not p)
            gain_interval=missing_interval(gain,missing,REPLICATES,Q(1,40))
            loss_interval=missing_interval(loss,missing,REPLICATES,Q(1,40))
            paired_rows.append(dict(cell_id=cell['cell_id'],length=length,planned=REPLICATES,gain=gain,loss=loss,missing=missing,
                gain_interval=gain_interval,loss_interval=loss_interval,
                primary_minus_control_interval=(gain_interval[0]-loss_interval[1],gain_interval[1]-loss_interval[0])))
    return dict(method_rows=method_rows,paired_rows=paired_rows,contract_issues=contract_issues,dominance_audit=dominance_rows,
                planned_jobs=len(CELLS)*REPLICATES,
                planned_method_outcomes=len(CELLS)*REPLICATES*len(LENGTHS)*2,
                planned_member_outcomes=len(CELLS)*REPLICATES*len(LENGTHS)*2*4,
                scope='Finite implementation assessment. POWER_UNVERIFIED. No recorder-rounding transfer proof.')

if __name__=='__main__':
    import sys
    save(Path(sys.argv[1])/'ANALYSIS.json',analyze(sys.argv[1]))
