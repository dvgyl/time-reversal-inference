"""Calculate declared population quantities without observations."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import adapters as A
from study_common import BANK,CELLS,DESIGN,LENGTHS,save

B=A.B
path=Path(__file__).resolve().parent.parent/'population/population_oracle.py'
spec=importlib.util.spec_from_file_location('_study_population_oracle',path)
oracle=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=oracle
spec.loader.exec_module(oracle)


def planned_rows():
    limits=B.WorkLimits()
    design=A.certify_design(A.CalibrationDesign(DESIGN,(0,1)),limits)
    output=[]
    for cell in CELLS:
        response=A.plan_response(cell['response'],B.Q(1,10000),design,B.Arithmetic(limits))
        true_shape,true_phase=B.response_bounds(cell['response'],(0,1),B.Q(0),B.Arithmetic(limits))
        for length in LENGTHS:
            source=B.planned_source_bound(cell['phi'],length,B.Q(1,2**20),B.Arithmetic(limits))
            for coefficient in BANK:
                population=oracle.Population(cell['phi'],cell['correlation'],cell['delay'],B.Q(1),tuple(enumerate(cell['response'])),coefficient)
                moments=oracle.moments(population)
                entry=dict(cell=cell,length=length,coefficient=coefficient,population=moments,
                           planned_response=response,planned_source=source,true_shape_upper=true_shape,true_phase_upper=true_phase)
                entry['null_phase_term_upper']=2*true_phase*moments['maximum_variance']
                phase_term=entry['null_phase_term_upper']
                entry['phase_nearness_ratio']=abs(moments['signed_contrast'])/phase_term if phase_term else None
                lanes={}
                for lane in ('primary','control'):
                    if lane=='primary' and (isinstance(source,B.Unavailable) or source['upper']>=1):
                        lanes[lane]=dict(status='POWER_UNVERIFIED',reason=getattr(source,'reason','planned_source_hull_reaches_one'))
                        continue
                    lower=source['lower'] if lane=='primary' else cell['phi']
                    upper=source['upper'] if lane=='primary' else cell['phi']
                    ar=B.Arithmetic(limits)
                    ratio=B.source_ratio(lower,upper,coefficient,ar)
                    q=ar.mulq(2,ar.mulq(ratio,max(B.Q(1),response['shape_upper'])))
                    margin=B.conditional_power_margin(q,response['phase_upper'],moments['maximum_variance'],moments['total_variance'],
                                                      abs(moments['signed_contrast']),length-1,len(BANK),ar)
                    lanes[lane]=dict(source_ratio=ratio,covariance_ratio=q,margin=margin)
                entry['lanes']=lanes
                output.append(entry)
    return dict(design_certificate=design,rows=output,oracle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='Static rational population analysis. No observations. POWER_UNVERIFIED.')

if __name__=='__main__':
    save(Path(__file__).with_name('STATIC_PLANNING.json'),planned_rows())
