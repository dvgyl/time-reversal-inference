"""Evaluate declared methods without selecting settings from their outcomes."""

from dataclasses import replace
from fractions import Fraction as Q
from math import ceil, log2
from pathlib import Path

import numpy as np

import two_sided_source as source
import fast_dyadic as source_fast
import cycle_diagonal as cycle
import exact_dyadic as cycle_fast
import octant_cycle
import physical_models
from calibrated_adapters import fit_rounded_calibration, member_evaluator, supplied_physical_test


def _supplied_source(phi, identifier):
    return source.base.SourceOuterHull(phi,phi,(source.Interval(phi,phi),),(),(),None,None,
                                       dict(supplied_phi=phi,population_id=identifier))


def _bank(coefficients,limits):
    return source.BankPlan(tuple(map(Q,coefficients)),limits)


def _calibration(design,response,limits,error):
    data=source.CalibrationData(design,response,(0,1))
    return data,fit_rounded_calibration(source.base.reference,data,limits,error)


def _compile_source(integers,bits,limits):
    try:
        return source_fast.compile_dyadic(integers,bits,limits)
    except source.base.WorkLimit as refusal:
        failure=source.base.PhaseFailure('compilation',str(refusal),dict(refusal=str(refusal)))
        return source.base.CompilationFailure(limits,'dyadic',source.base.DEPENDENCIES,failure)


def _compilation_metadata(compiled):
    if isinstance(compiled,source.base.CompilationFailure):
        return dict(recording_identity=None,compilation_receipt=compiled.failure.receipt,
                    compilation_failure=compiled)
    return dict(recording_identity=compiled.identity,compilation_receipt=compiled.preparation_receipt)


def _source_hulls(compiled,limits,rounding,cutoff_directory,*,inversion='roots',root_bits=48):
    if isinstance(compiled,source.base.CompilationFailure):
        return {name:source.base.SourceFallback(compiled.failure.reason,compiled.failure.receipt)
                for name in ('lag','lower_cn','lower_cstar','lower_intersection','two_sided','two_sided_intersection')}
    n=compiled.identity.original_length-1
    baseline=source.base
    plans=dict(lag=baseline.LagPlan(),lower_cn=baseline.BridgePlan(cutoff='s18_cn'),
               lower_cstar=baseline.BridgePlan(cutoff='s18_cstar'),
               lower_intersection=baseline.IntersectionPlan())
    hulls={name:source.source_confidence_baseline(plan,compiled,limits,rounding,
                                                inversion=inversion,root_bits=root_bits)
           for name,plan in plans.items()}
    for name,denominator,alpha_lag in (('two_sided',200,Q(0)),('two_sided_intersection',400,Q(1,400))):
        plan=source.plan_from_certificate(cutoff_directory/('pivot_n%d_alpha1_%d.json'%(n,denominator)),
                                         alpha_lag=alpha_lag,root_bits=root_bits)
        hulls[name]=source.source_confidence(plan,compiled,limits,rounding_error=rounding,inversion=inversion)
    return hulls


def evaluate_source(cell,integers,bits,rounding,calibration_design,calibration_response,calibration_error,
                    registry,cutoff_directory):
    limits=source.WorkLimits(**registry['source']['limits'])
    compiled=_compile_source(integers,bits,limits)
    bank=_bank(registry['source']['bank'],limits)
    data,fitted=_calibration(calibration_design,calibration_response,limits,calibration_error)
    hulls=_source_hulls(compiled,limits,rounding,cutoff_directory,root_bits=registry['source']['root_bits'])
    hulls['supplied']=_supplied_source(Q(cell['parameters']['phi']),cell['id'])
    primary=member_evaluator(source,rounding_error=rounding)
    def apply(hull,chosen_bank=bank,evaluator=primary,name=''):
        return source.evaluate_compiled_bank(chosen_bank,compiled,hull,data,member_evaluator=evaluator,
                                             fitted_calibration=fitted,source_plan=name,
                                             source_budget=Q(0) if name=='supplied' else Q(1,200))
    results={name:apply(hull,name=name) for name,hull in hulls.items()}
    for name in ('two_sided','two_sided_intersection'):
        results[name+'_maximum_scale']=apply(hulls[name],evaluator=member_evaluator(
            source,rounding_error=rounding,scale_mode='maximum'),name=name)
        results[name+'_old_operator']=apply(hulls[name],evaluator=member_evaluator(
            source,rounding_error=rounding,operator_factor=2),name=name)
    n=cell['N']-1
    adaptive=_bank([Q(1)-Q(1,2**j) for j in range(ceil(log2(n))+1)],limits)
    results['two_sided_adaptive_bank']=apply(hulls['two_sided'],chosen_bank=adaptive,name='two_sided')
    for factor in registry['ablations']['underestimated_source_shape']:
        results['two_sided_shape_'+factor]=apply(hulls['two_sided'],evaluator=member_evaluator(
            source,rounding_error=rounding,shape_factor=Q(factor)),name='two_sided')
    return dict(methods=results,source_sets=hulls,calibration=fitted,**_compilation_metadata(compiled)),compiled


def numerical_variants(compiled,rounding,registry,cutoff_directory,calibration_design,
                       calibration_response,calibration_error):
    failed=isinstance(compiled,source.base.CompilationFailure)
    original=compiled.limits if failed else compiled.identity.limits
    output={}
    variants=[('depth_%d'%depth,replace(original,max_depth=depth),'dyadic',registry['source']['root_bits'])
              for depth in registry['ablations']['depth']]
    variants += [('precision_%d'%bits,replace(original,sqrt_bits=bits),'roots',registry['source']['root_bits'])
                 for bits in registry['ablations']['arithmetic_bits']]
    for name,limits,inversion,root_bits in variants:
        changed={key for key,value in original.__dict__.items() if limits.__dict__[key]!=value}
        if not changed <= {'max_depth','sqrt_bits'}:
            raise ValueError('Only inversion and threshold precision can change after exact compilation.')
        if failed:
            updated=replace(compiled,limits=limits)
        else:
            identity=replace(compiled.identity,limits=limits)
            updated=replace(compiled,identity=identity,
                            bank_polynomials=(compiled.bank_polynomials if isinstance(compiled.bank_polynomials,source.base.PhaseFailure)
                                              else replace(compiled.bank_polynomials,identity=identity)),
                            bridge_polynomial=(compiled.bridge_polynomial if isinstance(compiled.bridge_polynomial,source.base.PhaseFailure)
                                               else replace(compiled.bridge_polynomial,identity=identity)))
        hulls=_source_hulls(updated,limits,rounding,cutoff_directory,inversion=inversion,root_bits=root_bits)
        data,fitted=_calibration(calibration_design,calibration_response,limits,calibration_error)
        bank=_bank(registry['source']['bank'],limits)
        evaluator=member_evaluator(source,rounding_error=rounding)
        results={method:source.evaluate_compiled_bank(bank,updated,hull,data,member_evaluator=evaluator,
                                                       fitted_calibration=fitted,source_plan=method)
                 for method,hull in hulls.items()}
        output[name]=dict(source_sets=hulls,methods=results,calibration=fitted,limits=limits,
                          compilation_reuse='Exact integer moments do not use subdivision depth or square-root precision',
                          original_compilation_identity=None if failed else compiled.identity)
    return output


def evaluate_physical(cell,integers,bits,rounding,calibration_design,calibration_response,calibration_error,registry):
    limits=source.WorkLimits(**registry['source']['limits'])
    compiled=_compile_source(integers,bits,limits)
    if cell['family']=='phase_boundary':
        reference=source.base.reference
        fitted=reference.CalibrationCertificate((Q(0),Q(1)),(0,1),source.Interval(Q(0),Q(0)),
                                                Q(1),Q(1),dict(supplied_exact_phase=True,failure_budget=Q(0)))
    else:
        _,fitted=_calibration(calibration_design,calibration_response,limits,calibration_error)
    q=Q(cell['parameters']['q'])
    methods={}
    for name,scale,operator in (('calibrated','geometric',1),('maximum_scale','maximum',1),('old_operator','geometric',2)):
        methods[name]=supplied_physical_test(source,compiled,fitted,q,limits,rounding_error=rounding,
                                             scale_mode=scale,operator_factor=operator)
    methods['missing_bound']=source.base.Unavailable('physical_covariance_bound_not_supplied',{})
    methods['missing_response']=source.base.Unavailable('relative_phase_bound_not_supplied',{})
    if isinstance(fitted,source.base.reference.CalibrationCertificate):
        ignored=replace(fitted,phase_upper=Q(0))
        decision=supplied_physical_test(source,compiled,ignored,q,limits,rounding_error=rounding)
        decision=replace(decision,receipt=dict(decision.receipt,
            assumption_violation='Relative response phase set to zero',size_bound=None))
        methods['phase_ignored']=decision
    else:
        methods['phase_ignored']=fitted
    for factor in registry['ablations']['underestimated_source_shape']:
        methods['shape_'+factor]=supplied_physical_test(source,compiled,fitted,max(Q(1),q*Q(factor)),limits,
                                                       rounding_error=rounding)
    return dict(methods=methods,calibration=fitted,**_compilation_metadata(compiled))


def _common(bounds):
    return cycle.DiagonalBounds((max(bounds.covariance),)*3,(max(bounds.bias),)*3,
                                bounds.source+'; common scalar envelope')


def evaluate_cycle(cell,integers,bits,rounding,registry):
    p=cell['parameters']
    policy_data=dict(registry['cycle']['limits'])
    policy_data['precision_bits']=tuple(policy_data['precision_bits'])
    policy=cycle.ref.NumericalPolicy(**policy_data)
    design=cycle.CycleDesign(cell['N'],p['lag'],(cycle.ref.QuarterTurn.PI_OVER_TWO,),Q(1,20),policy)
    compiled=cycle_fast.compile_dyadic(cycle_fast.DyadicRecord(integers,bits),design)
    filters=tuple(tuple((position,Q(value)) for position,value in row) for row in p['filters'])
    bound=physical_models.rotational_family_bounds(cell['N'],p['lag'],filters,Q(registry['cycle']['family_rho_ceiling']))
    if p['condition']=='correlated_error':
        noise=physical_models.rotational_family_bounds(cell['N'],p['lag'],(((0,Q(1)),),)*3,
                                                       Q(registry['cycle']['family_rho_ceiling']))
        variance=Q(p['error_variance'])
        bound=cycle.DiagonalBounds(tuple(a+variance*b for a,b in zip(bound.covariance,noise.covariance)),
                                   tuple(a+variance*b for a,b in zip(bound.bias,noise.bias)),
                                   'Full observed-family envelope including correlated error; source interpretation violates error orthogonality')
    bound=replace(bound,bias=tuple(Q(p['bias_inflation'])*b for b in bound.bias))
    perturbation=cycle.Perturbation((rounding,)*3,'Declared recorder error without saturation')
    methods=dict(diagonal=cycle.evaluate_cycle(compiled,bound,perturbation),
                 common=cycle.evaluate_cycle(compiled,_common(bound),perturbation),
                 combined_frequencies=octant_cycle.evaluate_grid(compiled,bound,perturbation))
    paired={}
    if p.get('gain_and_inflation') and isinstance(compiled,cycle.CompileFailure):
        refusal=cycle.evaluate_cycle(compiled,bound,perturbation)
        for gain in registry['cycle']['gains']:
            paired['gain_'+gain]=dict(diagonal=refusal,common=refusal,inverse_transport=refusal)
        for ck in registry['cycle']['inflation']:
            for cb in registry['cycle']['inflation']:
                paired['inflation_K%s_B%s'%(ck,cb)]=refusal
        for factor in registry['cycle']['underestimation']:
            for component in ('K','B'):
                paired['underestimated_%s_%s'%(component,factor)]=refusal
    if p.get('gain_and_inflation') and isinstance(compiled,cycle.CompiledCycle):
        for gain in registry['cycle']['gains']:
            transported,changed,errors=cycle.transport_gains(compiled,bound,perturbation,(Q(1),Q(1),Q(gain)))
            diagonal=cycle.evaluate_cycle(transported,changed,errors)
            common=cycle.evaluate_cycle(transported,_common(changed),errors) if isinstance(changed,cycle.DiagonalBounds) else diagonal
            inverted,inverse_bounds,inverse_errors=cycle.transport_gains(transported,changed,errors,(Q(1),Q(1),1/Q(gain))) if isinstance(transported,cycle.CompiledCycle) else (transported,changed,errors)
            paired['gain_'+gain]=dict(diagonal=diagonal,common=common,
                                     inverse_transport=cycle.evaluate_cycle(inverted,inverse_bounds,inverse_errors))
        for ck in registry['cycle']['inflation']:
            for cb in registry['cycle']['inflation']:
                inflated=cycle.DiagonalBounds(tuple(Q(ck)*x for x in bound.covariance),
                                              tuple(Q(cb)*x for x in bound.bias),bound.source+'; inflated bounds')
                paired['inflation_K%s_B%s'%(ck,cb)]=cycle.evaluate_cycle(compiled,inflated,perturbation)
        for factor in registry['cycle']['underestimation']:
            for component in ('K','B'):
                wrong=replace(bound,covariance=tuple(Q(factor)*x for x in bound.covariance)) if component=='K' else replace(bound,bias=tuple(Q(factor)*x for x in bound.bias))
                paired['underestimated_%s_%s'%(component,factor)]=cycle.evaluate_cycle(compiled,wrong,perturbation)
    return dict(methods=methods,paired=paired,bounds=bound,perturbation=perturbation,
                compilation=compiled)
