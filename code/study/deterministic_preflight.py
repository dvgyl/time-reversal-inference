"""Exercise complete study adapters with fixed arrays and no scientific draw."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import run_study as runner
import build_registry
import analyze_study as analysis


class FixedInput:
    """Supply periodic numbers for deterministic integration checks."""
    def standard_normal(self,size):
        dimensions=(size,) if isinstance(size,int) else size
        count=int(np.prod(dimensions))
        values=np.resize(np.array([-1.,.25,.75,-.5,1.,-.25,-.75,.5]),count)
        return values.reshape(dimensions)

    def standard_t(self,df,size):
        if df!=5:
            raise ValueError('The fixture expects five degrees of freedom.')
        return self.standard_normal(size)

    def integers(self,low,high,size):
        return (np.arange(int(np.prod(size))).reshape(size)%(high-low))+low


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    registry=build_registry.build()
    identifiers=['source_4_5_null_16385','source_199_200_strong_2049',
                 'cycle_null_8192','brownian_T1_32769','phase_boundary_99_100',
                 'source_reference_snr_10_null','source_student_t_5_null','cycle_correlated_error_null']
    outcomes=[]
    for identifier in identifiers:
        cell=next(item for item in registry['cells'] if item['id']==identifier)
        started=time.monotonic()
        streams={role:FixedInput() for role in registry['execution']['roles']}
        raw=runner.generation.generate_record(cell,streams)
        integers,error=runner.generation.quantize_channels(raw,20)
        calibration=(None,None,runner.Q(0))
        if cell['family'] in ('source','brownian'):
            design,response=runner.physical_models.calibration_record(256,
                [float(runner.Q(x)) for x in cell['parameters']['response']],.01,FixedInput())
            rounded,allowance=runner.generation.quantize_channels(np.column_stack((response,response)),20)
            calibration=(design.astype(int).tolist(),[runner.Q(int(x),2**20) for x in rounded[:,0]],allowance)
        primary,compiled=runner._evaluate(cell,integers,20,error,calibration,registry,
                                         runner.ROOT/'source_inference/records/cutoffs_001')
        encoded=runner.encode(primary)
        decisions={key:analysis.status(value) for key,value in analysis.flatten_methods(encoded).items()}
        for value in analysis.flatten_methods(encoded).values():
            analysis.diagnostics(value)
        comparators=runner.baselines.lag_baselines(raw,FixedInput())
        comparators['imaginary_coherency']=runner.baselines.imaginary_coherency(raw,FixedInput())
        variants={}
        if identifier=='source_4_5_null_16385':
            numerical=runner.methods.numerical_variants(compiled,error,registry,
                runner.ROOT/'source_inference/records/cutoffs_001',*calibration)
            variants={key:{name:analysis.status(runner.encode(value)) for name,value in record['methods'].items()}
                      for key,record in numerical.items()}
        outcomes.append(dict(cell=identifier,decisions=decisions,variants=variants,
                             seconds=time.monotonic()-started))
        print(json.dumps(outcomes[-1]),flush=True)
        runner.write_json(args.output.with_name(args.output.stem+'_'+identifier+'.json'),outcomes[-1])
    runner.write_json(args.output,dict(scope='Deterministic integration fixtures. No scientific random draw.',
                                      outcomes=outcomes))


if __name__=='__main__':
    main()
