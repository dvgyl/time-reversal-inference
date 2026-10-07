"""Evaluate proved source-envelope conditions with deterministic population inputs."""
from fractions import Fraction as Q
from pathlib import Path
import json
from flint import arb,ctx
from certified_pivot import ball,rational_endpoint


def envelope(n,phi,clo,chi,beta=Q(1,20),qminus=Q(23,1000)):
    if type(n) is not int or any(type(x) is not Q for x in (phi,clo,chi,beta,qminus)):
        raise ValueError('Use an integer n and exact Fraction population parameters.')
    if not 0<=phi<1 or not 0<clo<chi:raise ValueError('Require 0 <= phi < 1 and 0 < c_lo < c_hi.')
    if beta!=Q(1,20) or qminus!=Q(23,1000) or n<129:raise ValueError('The saved q_minus certificate uses beta=.05, q_minus=.023, and n>=129.')
    phi=Q(phi);delta=1-phi;p=ball(phi);d=ball(delta);t=ball(5/beta).log();r=p**n
    dm=(n-1-2*(arb(n-1)*t).sqrt())/n
    dp=(n-1+2*(arb(n-1)*t).sqrt()+2*t)/n
    T=(1+p)/d
    d0=1+2*(T*t/n).sqrt()+2*T*t/n
    # Finite geometric sums, evaluated in enclosing arithmetic.
    sum_phi=p*(1-p**(n-1))/d
    sum_jphi=p*(1-n*p**(n-1)+(n-1)*p**n)/(d*d)
    h=sum_phi-sum_jphi/n
    tau=arb(n-1)/n+arb((n-1)*(2*n-1))/(3*n*n)+r*arb(n*n-1)/(3*n*n)-4*h/n
    g=(T.sqrt()+(arb(n-1)*(1+r)/2).sqrt())**2
    b=tau+2*(g/n*tau*t).sqrt()+2*g*t/n
    A=((2-d)*ball(qminus)).sqrt();B=(b/(n*d)).sqrt();E=(d*d0).sqrt()
    Flo=((2-d)*dm).sqrt();Fhi=((2-d)*dp).sqrt();sl=ball(clo).sqrt();sh=ball(chi).sqrt()
    Jlo=B+sl*E;Jhi=B+sh*E;C=sl*(Flo-E)-B;H=sh*(Fhi-E)-B
    record={'n':n,'phi':str(phi),'clo':str(clo),'chi':str(chi),'beta':'1/20','q_minus':'23/1000',
            'C':str(C),'A_minus_Jhi':str(A-Jhi),'Fminus_minus_E':str(Flo-E),
            'endpoint_trace_over_vn':str(tau),'endpoint_norm_bound_over_vn':str(g/n),'endpoint_upper_b':str(b)}
    if not(C>0 and A>Jhi and Flo>E):
        return dict(record,status='SUFFICIENT_ENVELOPE_CONDITION_FAILED')
    Xhi=H/(A-Jhi);Xlo=C/(sh*Fhi-Jlo)
    gap=H/C*(A-Jlo)/(A-Jhi)
    factor=2/(2-d*Xhi) if d*Xhi<1 else arb(2)
    span=gap*factor
    # Endpoints are rounded outwards; physical truncation is exact.
    ell=max(Q(0),rational_endpoint(1-d*Xhi));upper=min(Q(1),rational_endpoint(1-d*Xlo,True))
    def trans(x):return (1+x)/(1-x)
    bank=[Q(0),Q(1,2),Q(4,5),Q(9,10),Q(97,100),Q(199,200)]
    member_bounds=[{'a':str(a),'S_upper':str(max((trans(upper)/trans(a))**2,(trans(a)/trans(ell))**2))} for a in bank]
    best=min(member_bounds,key=lambda row:Q(row['S_upper']))
    return dict(record,status='SUFFICIENT_ENVELOPE_CONDITION_PASSED',gap_ratio_bound=str(gap),
                span_bound=str(span),ell=str(ell),upper=str(upper),members=member_bounds,best_member=best)


def main():
    ctx.prec=128;rows=[]
    for n in [16384,65536]:
        path=Path(f'../records/cutoffs_001/pivot_n{n}_alpha1_200.json');c=json.loads(path.read_text())
        for phi in [Q(4,5),Q(97,100),Q(199,200)]:
            row=envelope(n,phi,Q(c['lower_cutoff']),Q(c['upper_cutoff']));rows.append(row)
            print(n,phi,row['status'],flush=True)
    Path('../records/population_envelope_001.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()


def complete_power_margin(row,coupling=Q(9,10),phase=Q(1,200),shape=Q(101,100),operator_factor=1):
    """Check an ideal conditional margin; calibration upper bounds are assumptions."""
    if operator_factor not in (1,2) or type(operator_factor) is not int:raise ValueError('Use proved operator factors 1 or 2.')
    if row['status']!='SUFFICIENT_ENVELOPE_CONDITION_PASSED':return {'status':'SOURCE_ENVELOPE_UNAVAILABLE'}
    N=row['n'];m=N-1;M=6;phi=Q(row['phi']);member=row['best_member'];a=Q(member['a']);S=Q(member['S_upper'])
    q=2*ball(S)*ball(shape);D=ball(1+a*a-2*a*phi)
    contrast=ball(coupling*(1-phi*phi)*(1+a*a-a*phi))
    u=arb(200*M).log();sv=arb(200).log();st=arb(80*M).log();sa=arb(200).log()
    den=1-q/N-2*(q*u/N).sqrt()
    out={'bank_member':str(a),'S_upper':str(S),'q_upper':str(q),'D':str(D),'contrast':str(contrast),
         'scale_denominator':str(den),'operator_factor':operator_factor,'calibration_phase_assumption':str(phase),
         'calibration_shape_assumption':str(shape),'scope':'Ideal conditional population margin. Calibration probability and numerical completion are separate conditions.'}
    if not den>0:return dict(out,status='NONPOSITIVE_SCALE_DENOMINATOR')
    Dup=D*(1+2*(q*sv/N).sqrt()+2*q*sv/N);scale=Dup/den
    Fn=ball(Q(m+1,m*m)-Q(2,m**3)-Q(2,m**4)).sqrt()
    def rad(scale,t):
        K=q*scale;fro=K*Fn;mix=operator_factor*(K*N*2*scale).sqrt()/m
        # Both terms bound the same norm. Choose the one with smaller certified upper endpoint.
        term=fro if fro.upper()<mix.upper() else mix
        return 2*term*t.sqrt()+2*operator_factor*K*t/m
    null=2*ball(phase)*scale+rad(scale,st)
    alt=rad(D,sa)
    bias=q*D*arb(2*(4*m-2)).sqrt()/(m*m)
    margin=contrast-null-alt-bias
    return dict(out,status='CONDITIONAL_MARGIN_POSITIVE' if margin>0 else 'CONDITIONAL_MARGIN_NOT_POSITIVE',
                contrast_margin=str(margin),required_contrast=str(null+alt+bias))


def calibration_population_bounds(rows=256,sigma=Q(1,100),error=Q(1,2**21),beta=Q(1,100)):
    """Bound identity-response certificates for an orthogonal two-column design."""
    p=2;nu=rows-p;t=ball(2/beta).log();t0=arb(200).log()
    Cp=p+2*(p*t).sqrt()+2*t;C0=p+2*(p*t0).sqrt()+2*t0
    vlo=nu-2*(nu*t0).sqrt();vup=nu+2*(nu*t).sqrt()+2*t
    e=ball(sigma)*(Cp/rows).sqrt();eta=ball(error)
    rad=(ball(sigma)*vup.sqrt()+2*arb(rows).sqrt()*eta)*(C0/(vlo*rows)).sqrt()+eta
    z=e+eta+rad
    phase=z/(1-2*z);shape=((1+arb(2).sqrt()*z)/(1-z))**2
    return {'rows':rows,'sigma':str(sigma),'recording_error':str(error),'failure_budget':str(beta),
            'coefficient_error':str(e),'radius_upper':str(rad),'total_radius':str(z),
            'phase_upper':str(rational_endpoint(phase,True)),'shape_upper':str(rational_endpoint(shape,True)),
            'scope':'Identity response; X transpose X = rows I; two independent Gaussian calibration coordinates; upper-tail events only.'}


def rounded_envelope(n,phi,clo,chi,error=Q(1,2**21),root_bits=48):
    """Check outward source bounds with the telescoping recorder relaxation."""
    base=envelope(n,phi,clo,chi)
    if base['status']!='SUFFICIENT_ENVELOPE_CONDITION_PASSED':return base
    delta=1-Q(phi);d=ball(delta);p=ball(phi);t=arb(100).log();eta=ball(error)
    dm=(n-1-2*(arb(n-1)*t).sqrt())/n;dp=(n-1+2*(arb(n-1)*t).sqrt()+2*t)/n
    T=(1+p)/d;d0=1+2*(T*t/n).sqrt()+2*T*t/n
    # Recover bounds from exact formulas; serialized displays are not computation inputs.
    r=p**n;h=p*(1-p**(n-1))/d-p*(1-n*p**(n-1)+(n-1)*p**n)/(n*d*d)
    tau=arb(n-1)/n+arb((n-1)*(2*n-1))/(3*n*n)+r*arb(n*n-1)/(3*n*n)-4*h/n
    g=(T.sqrt()+(arb(n-1)*(1+r)/2).sqrt())**2;b=tau+2*(g/n*tau*t).sqrt()+2*g*t/n
    Amin=((2-d)*ball(Q(23,1000))).sqrt();B=(b/(n*d)).sqrt();E=(d*d0).sqrt()
    Flo=((2-d)*dm).sqrt();Fhi=((2-d)*dp).sqrt();Amax=ball(chi).sqrt()*Fhi
    import two_sided_source as source
    arithmetic=source.base.Arithmetic(source.WorkLimits())
    unit=arithmetic.mulq(arithmetic.sqrt(arithmetic.point(n)).upper,error)
    a0=arithmetic.mulq(2*error,arithmetic.sqrt(arithmetic.point(n-1)).upper)
    a1=arithmetic.mulq(Q(n,2),unit)
    eps=ball(Q(1,2**arithmetic.limits.sqrt_bits))
    u0=ball(a0)/(d.sqrt()*n);u1=ball(a1)*d/(d.sqrt()*n)
    CQ=(Amax+B)/d + u0+u1/d+eps/(d.sqrt()*n)
    w=2*ball(unit)/(n*d).sqrt()
    CD=Fhi+max(arb(1),(1/d-1))*E+w+eps/(n*d).sqrt()
    DeltaD=2*CD*w+w*w
    def qerr(x):
        u=u0+u1*x
        return 2*CQ*u+u*u
    def lower_exclusion(x):
        return (B+x*(Amax-B))**2-ball(clo)*(Flo-E+x*E)**2+2*(qerr(x)+ball(clo)*DeltaD)
    def upper_exclusion(x):
        return (B+x*(Amin-B))**2-ball(chi)*(Fhi-E+x*E)**2-2*(qerr(x)+ball(chi)*DeltaD)
    # The lower exclusion polynomial is convex. The upper polynomial is convex
    # and increasing to the right of its positive root.
    loquad=(Amax-B)**2-ball(clo)*E*E+2*u1*u1
    upquad=(Amin-B)**2-ball(chi)*E*E-2*u1*u1
    if not(loquad>0 and upquad>0 and lower_exclusion(arb(0))<0):
        return dict(base,status='ROUNDED_ENVELOPE_CONDITION_FAILED')
    xlo,xhi=Q(0),Q(1)
    for _ in range(64):
        mid=(xlo+xhi)/2
        if lower_exclusion(ball(mid))<0:xlo=mid
        else:xhi=mid
    eta_gap=xlo*Q(999,1000)
    # Find a certified upper exclusion point with positive derivative.
    def derivative(x):
        return 2*(B+x*(Amin-B))*(Amin-B)-2*ball(chi)*(Fhi-E+x*E)*E-4*u1*(CQ+u0+u1*x)
    left,right=Q(1),1/delta
    if not(upper_exclusion(ball(right))>0 and derivative(ball(right))>0):
        L=right
    else:
        for _ in range(64):
            mid=(left+right)/2
            if upper_exclusion(ball(mid))>0 and derivative(ball(mid))>0:right=mid
            else:left=mid
        L=right*Q(1001,1000)
    width=Q(8,2**root_bits)
    ell=max(Q(0),1-delta*L-width);upper=min(Q(1),1-delta*eta_gap+width)
    if upper>=1:return dict(base,status='ROUNDED_ENVELOPE_CONDITION_FAILED')
    def trans(x):return (1+x)/(1-x)
    bank=[Q(0),Q(1,2),Q(4,5),Q(9,10),Q(97,100),Q(199,200)]
    members=[{'a':str(a),'S_upper':str(max((trans(upper)/trans(a))**2,(trans(a)/trans(ell))**2))} for a in bank]
    return dict(base,status='SUFFICIENT_ENVELOPE_CONDITION_PASSED',rounding_error=str(error),
                root_enlargement=str(width),ell=str(ell),upper=str(upper),members=members,
                best_member=min(members,key=lambda r:Q(r['S_upper'])),
                rounding_scope='Requires the prescribed telescoping polynomial relaxation and its outward constants.',
                source_qualification='Includes source coverage and the five-event envelope probability.')


def calibration_algorithm_bounds(error=Q(1,2**21)):
    """Enclose the implemented calibration bound on two Gaussian upper events."""
    import two_sided_source as source
    ar=source.base.Arithmetic(source.WorkLimits())
    sigma=Q(1,100);m=256;p=2;nu=254;t=ar.log(Q(200));eps=Q(1,2**ar.limits.sqrt_bits)
    C=ar.add(ar.add(ar.point(p),ar.mul(ar.point(2),ar.sqrt(ar.mul(ar.point(p),t)))),ar.mul(ar.point(2),t))
    V=ar.add(ar.add(ar.point(nu),ar.mul(ar.point(2),ar.sqrt(ar.mul(ar.point(nu),t)))),ar.mul(ar.point(2),t))
    vlo=ar.sub(ar.point(nu),ar.mul(ar.point(2),ar.sqrt(ar.mul(ar.point(nu),t))))
    e=ar.sqrt(ar.div(ar.mul(ar.point(sigma*sigma),C),ar.point(m))).upper
    rssroot=ar.add(ar.mul(ar.point(sigma),ar.sqrt(V)),ar.point(2*16*error+eps))
    radius=ar.add(ar.sqrt(ar.div(ar.mul(ar.mul(rssroot,rssroot),C),ar.mul(vlo,ar.point(m)))),ar.point(error)).upper
    rho=ar.q(e+error+radius+eps)
    shape,phase=source.base.reference.response_bounds((Q(1),Q(0)),(0,1),rho,ar)
    return {'rows':m,'columns':p,'sigma':'1/100','rounding_error':str(error),'failure_budget':'3/200',
            'residual_norm_lower':str(ar.q(sigma*ar.sqrt(vlo).lower-16*error)),
            'residual_positive_verified':sigma*ar.sqrt(vlo).lower>16*error,
            'phase_upper':str(phase),'shape_upper':str(shape),'coefficient_error_upper':str(e),
            'radius_upper':str(radius),'rho_upper':str(rho),'arithmetic':ar.receipt(),
            'fixed_phase_bound':'33/5000','fixed_shape_bound':'129/125',
            'fixed_bounds_verified':phase<=Q(33,5000) and shape<=Q(129,125)}


def implemented_threshold_population_margin(row,coupling=Q(9,10),phase=Q(33,5000),shape=Q(129,125)):
    """Include the declared recorder correction and threshold interval arithmetic."""
    import importlib.util,hashlib
    import two_sided_source as source
    parent=Path(__file__).resolve().parents[2]/'cycle/calibrated_threshold.py'
    spec=importlib.util.spec_from_file_location('_pre016_threshold_planning',parent)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if row['status']!='SUFFICIENT_ENVELOPE_CONDITION_PASSED':return {'status':'SOURCE_ENVELOPE_UNAVAILABLE'}
    N=row['n'];m=N-1;M=6;phi=Q(row['phi']);a=Q(row['best_member']['a']);S=Q(row['best_member']['S_upper'])
    q=2*S*shape;D=Q(1)+a*a-2*a*phi;contrast=coupling*(1-phi*phi)*(1+a*a-a*phi)
    error=Q(row['rounding_error']);filtered_error=(1+a)*error
    t=arb(200).log();Dp=ball(D)*(1+2*(ball(q)*t/N).sqrt()+2*ball(q)*t/N)
    # Quantized variance, followed by the implementation's own outward correction.
    obsvar=rational_endpoint((Dp.sqrt()+ball(filtered_error))**2,True)
    ar=source.base.Arithmetic(source.WorkLimits())
    idealvar=module.rounding_variance_upper(obsvar,filtered_error,ar).upper
    threshold,details=module.reflection_threshold(q,phase,(idealvar,idealvar),N,M,ar)
    if threshold is None:return {'status':'NONPOSITIVE_SCALE_DENOMINATOR'}
    correction=module.rounding_contrast_upper(4*N*obsvar,4*N*obsvar,m,(filtered_error,filtered_error),ar)
    Fn=ball(Q(m+1,m*m)-Q(2,m**3)-Q(2,m**4)).sqrt();K=ball(q*D)
    fro=K*Fn;mix=(K*N*2*ball(D)).sqrt()/m
    term=fro if fro.upper()<mix.upper() else mix
    alt=2*term*t.sqrt()+2*K*t/m
    bias=K*arb(2*(4*m-2)).sqrt()/(m*m)
    margin=ball(contrast)-ball(threshold.upper)-2*ball(correction.upper)-alt-bias
    return {'status':'ROUNDED_IDEAL_MARGIN_POSITIVE' if margin>0 else 'ROUNDED_IDEAL_MARGIN_NOT_POSITIVE',
            'contrast':str(contrast),'contrast_margin':str(margin),'threshold_upper':str(threshold.upper),
            'rounding_contrast_upper':str(correction.upper),'scale_denominator':str(details['scale_denominator']),
            'calibration_phase_upper':str(phase),'calibration_shape_upper':str(shape),
            'error_allocation':{'source_coverage':'1/200','source_envelope':'1/20','calibration_upper_events':'1/100','calibration_residual_lower':'1/200',
                                'variance_upper_event':'1/100','alternative_tail':'1/100'},
            'ideal_power_lower':'91/100','threshold_arithmetic':ar.receipt(),
            'threshold_code_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
            'scope':'Identity detector responses and unit marginal source variance. Includes bounded recorder errors and outward threshold arithmetic. The probability statement requires completion of source, calibration and member computations; no uniform completion probability is asserted here.'}
