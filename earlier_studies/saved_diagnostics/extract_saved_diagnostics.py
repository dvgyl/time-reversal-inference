from pathlib import Path
from fractions import Fraction as Q
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,math,statistics

def main():
 ROOT=Path(__file__).resolve().parents[2]
 parser=argparse.ArgumentParser(description='Summarize saved source certificates. This is a post hoc descriptive extraction.')
 parser.add_argument('--output',type=Path,required=True,help='New output directory outside the scientific package.')
 args=parser.parse_args()
 OUT=args.output.resolve()
 if OUT==ROOT or ROOT in OUT.parents:
  parser.error('The output directory must be outside the scientific package.')
 if OUT.exists():
  parser.error('The output directory already exists. Use a new path.')
 RUN=ROOT/'earlier_studies/mechanisms/study/run_001'
 if not RUN.is_dir():
  parser.error('The saved mechanism study directory is absent.')
 OUT.mkdir(parents=True,exist_ok=False)
 BANK=[Q(0),Q(1,2),Q(4,5),Q(9,10),Q(97,100),Q(199,200)]
 def q(x): return Q(*map(int,x['q']))
 def T(x): return (1+x)/(1-x)
 def bound(x,side): return q(x['fields'][side])
 rows=[]; members=[]; binding=[]
 for p in sorted(RUN.glob('source_*/N*.json')):
  if p.stem.endswith('_supplied'):continue
  raw=p.read_bytes(); d=json.loads(raw); assert d['execution_status']=='completed'
  method=d['method']; assert method in ('lag','bridge','intersection')
  binding.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()})
  f=d['result']['fields'];h=f['source']['fields'];lo=q(h['lower']);hi=q(h['upper']);assert 0<=lo<=hi<=1
  finite=hi<1; available=[m for m in f['members'] if m['type']=='MemberDecision']
  reasons=Counter(m['fields']['reason'] for m in f['members'] if m['type']=='Unavailable')
  stage='REJECT' if f['decision']=='REJECT' else 'AVAILABLE_NONREJECTION' if available else 'NO_FINITE_SOURCE' if not finite else 'SCALE_UNAVAILABLE'
  if stage=='SCALE_UNAVAILABLE':assert reasons=={'nonpositive_scale_denominator':6},reasons
  r={'job_id':d['job']['id'],'cell_id':d['job']['cell_id'],'replicate':d['job']['replicate'],'length':d['n'],'method':method,'decision':f['decision'],'stage':stage,'lower_exact':str(lo),'upper_exact':str(hi),'lower':float(lo),'upper':float(hi),'finite':int(finite),'available_members':len(available)}
  fields=['span_exact','bank_min_exact','bank_penalty_exact','span','log_span','bank_min','bank_penalty','min_covariance_ratio','ideal_min_denominator','saved_max_denominator_lower','largest_statistic_threshold_ratio','smallest_phase_fraction']
  r.update({k:None for k in fields})
  if finite:
   span=T(hi)/T(lo); ss=[max((T(hi)/T(a))**2,(T(a)/T(lo))**2) for a in BANK]; best=min(ss);penalty=best/span
   assert best>=span
   ph=max(Q(1),q(f['calibration']['fields']['shape_upper']));minq=2*ph*best;b=d['n']-1;t=math.log(1200)
   r.update(span_exact=str(span),bank_min_exact=str(best),bank_penalty_exact=str(penalty),span=float(span),log_span=math.log(float(span)),bank_min=float(best),bank_penalty=float(penalty),min_covariance_ratio=float(minq),ideal_min_denominator=1-float(minq)/b-2*math.sqrt(float(minq)*t/b))
   denoms=[]; ratios=[]; phases=[]
   for j,m in enumerate(f['members']):
    mf=m['fields'];receipt=mf['receipt'];den=bound(receipt['scale_denominator'],'lower');denoms.append(den)
    mr={'job_id':d['job']['id'],'length':d['n'],'method':method,'member':j,'coefficient_exact':str(BANK[j]),'source_ratio_exact':str(ss[j]),'denominator_lower':float(den),'available':int(m['type']=='MemberDecision'),'phase_upper':None,'threshold_upper':None,'statistic_absolute':None,'statistic_threshold_ratio':None,'phase_fraction':None}
    if m['type']=='MemberDecision':
     assert q(mf['source_ratio'])==ss[j]
     th=bound(mf['threshold'],'upper');stat=q(mf['statistic_absolute']);phase=2*q(f['calibration']['fields']['phase_upper'])*bound(receipt['variance_scale'],'upper')
     assert th>0 and th>=phase
     ratios.append(float(stat/th));phases.append(float(phase/th))
     mr.update(phase_upper=float(phase),threshold_upper=float(th),statistic_absolute=float(stat),statistic_threshold_ratio=float(stat/th),phase_fraction=float(phase/th))
    members.append(mr)
   r['saved_max_denominator_lower']=float(max(denoms))
   if available:r.update(largest_statistic_threshold_ratio=max(ratios),smallest_phase_fraction=min(phases))
  rows.append(r)
 assert len(rows)==1728
 counts={m:dict(Counter(r['stage'] for r in rows if r['method']==m)) for m in ('lag','bridge','intersection')}
 assert counts['lag']=={'NO_FINITE_SOURCE':344,'AVAILABLE_NONREJECTION':83,'REJECT':149},counts
 assert counts['bridge']=={'NO_FINITE_SOURCE':302,'SCALE_UNAVAILABLE':138,'AVAILABLE_NONREJECTION':136},counts
 assert counts['intersection']=={'NO_FINITE_SOURCE':298,'SCALE_UNAVAILABLE':2,'AVAILABLE_NONREJECTION':128,'REJECT':148},counts
 metrics=['lower','upper','span','log_span','bank_min','bank_penalty','min_covariance_ratio','ideal_min_denominator','saved_max_denominator_lower','largest_statistic_threshold_ratio','smallest_phase_fraction']
 groups=[]
 for key in sorted({(r['cell_id'],r['length'],r['method']) for r in rows}):
  rr=[r for r in rows if (r['cell_id'],r['length'],r['method'])==key];finite=[r for r in rr if r['finite']]
  g={'cell_id':key[0],'length':key[1],'method':key[2],'total':len(rr),'finite':len(finite),'usable':sum(r['available_members']>0 for r in rr),'rejections':sum(r['decision']=='REJECT' for r in rr),'stages':dict(Counter(r['stage'] for r in rr))}
  for k in metrics:
   vals=[r[k] for r in finite if r[k] is not None];g[k]={'defined':len(vals),'median':statistics.median(vals) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None}
  groups.append(g)
 def csvwrite(path,rr):
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 csvwrite(OUT/'source_diagnostics.csv',rows);csvwrite(OUT/'member_diagnostics.csv',members)
 summary={'scope':'Descriptive summaries of saved certificates. No raw arrays, inference routines or random generation were used. No decision was changed. Medians of source geometry use finite hulls only; threshold metrics require available members. Floating summaries are not probability certificates.','counts':counts,'groups':groups}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (OUT/'INPUT_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
 cumulative_members=[x for x in members if x['method']=='bridge' and x['available']]
 assert len(cumulative_members)==243
 assert max(x['phase_fraction'] for x in cumulative_members)<0.0118
 assert max(x['statistic_threshold_ratio'] for x in cumulative_members)<0.847

if __name__=='__main__':
 main()
