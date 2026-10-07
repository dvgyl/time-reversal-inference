"""Create deterministic cutoff certificates before study evaluation."""
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path
from hashlib import sha256
import argparse,json,time
from flint import ctx
from certified_pivot import quantile_bracket,pivot_cdf


def generate(n,alpha,outdir):
    start=time.time();tail=alpha/2
    low,low_hi,h1=quantile_bracket(n,tail,Q(1,200),Q(1,25),steps=16,tolerance='1e-9')
    high_lo,high,h2=quantile_bracket(n,1-tail,Q(1,2),Q(2),steps=16,tolerance='1e-9')
    lo_check=pivot_cdf(n,low,tolerance='1e-9')
    hi_check=pivot_cdf(n,high,tolerance='1e-9')
    assert lo_check.upper<=tail and hi_check.lower>=1-tail
    data={'n':n,'alpha_bridge':str(alpha),'alpha_lower':str(tail),'alpha_upper':str(tail),
          'lower_cutoff':str(low),'upper_cutoff':str(high),
          'lower_quantile_bracket':[str(low),str(low_hi)],'upper_quantile_bracket':[str(high_lo),str(high)],
          'lower_cutoff_cdf':asdict(lo_check),'upper_cutoff_cdf':asdict(hi_check),
          'cdf_history':[asdict(x) for x in h1+h2],
          'arithmetic':'python-flint Arb, 96 bits','integration_tolerance':'1e-9','truncation':4096,
          'implementation_sha256':sha256(Path('certified_pivot.py').read_bytes()).hexdigest(),
          'seconds':time.time()-start,'scientific_random_draws':0}
    target=outdir/f'pivot_n{n}_alpha{alpha.numerator}_{alpha.denominator}.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(data,default=str,indent=2)+'\n')
    print(n,str(alpha),float(low),float(high),data['seconds'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('n',type=int,nargs='+');args=parser.parse_args()
    ctx.prec=96
    out=Path('../records/cutoffs_001');out.mkdir(exist_ok=True)
    for n in args.n:
        for alpha in [Q(1,200),Q(1,400)]:generate(n,alpha,out)
