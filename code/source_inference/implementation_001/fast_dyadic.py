"""Compile exact dyadic moments with checked integer dot products."""
from fractions import Fraction as Q
import hashlib
import numpy as np
import two_sided_source as s


class Counter:
    def __init__(self,limits):self.limits=limits;self.operations=0;self.bits=0
    def charge(self,n):
        self.operations+=n
        if self.operations>self.limits.max_operations:raise s.base.WorkLimit('max_operations')
    def check(self,x):
        x=int(x);self.bits=max(self.bits,abs(x).bit_length())
        if self.bits>self.limits.max_rational_bits:raise s.base.WorkLimit('max_integer_bits')
        return x
    def dot(self,a,b):
        self.charge(2*len(a));a=np.asarray(a,dtype=np.int64);b=np.asarray(b,dtype=np.int64)
        if not len(a):return 0
        ma=max(abs(int(a.min())),abs(int(a.max())));mb=max(abs(int(b.min())),abs(int(b.max())))
        if not ma or not mb:return 0
        if ma*mb>(1<<61):
            # Signed Euclidean limbs preserve each integer exactly.
            radix=1<<26
            al=a%radix;ah=a//radix;bl=b%radix;bh=b//radix
            return self.check(self.dot(al,bl)+radix*(self.dot(al,bh)+self.dot(ah,bl))+radix*radix*self.dot(ah,bh))
        chunk=max(1,min(len(a),((1<<62)-1)//(ma*mb)))
        total=0
        for start in range(0,len(a),chunk):
            total+=int(np.dot(a[start:start+chunk],b[start:start+chunk]))
        return self.check(total)
    def sum(self,a):return self.dot(a,np.ones(len(a),dtype=np.int64))
    def fraction(self,x):
        x=Q(x);self.check(x.numerator);self.check(x.denominator);return x


def _poly(values):
    minus,zero,plus=values
    return s.Quadratic(zero,(plus-minus)/2,(plus+minus)/2-zero)


def compile_dyadic(integers,exponent,limits=s.WorkLimits()):
    """Compile values integers / 2**exponent, without rounding them.

    The caller supplies an integer N-by-2 array. Quantization and its recorder
    error bound are separate inputs to the procedure. No random generator is used.
    """
    if type(exponent) is not int or not 0<=exponent<=52:raise ValueError('Use an integer exponent from 0 through 52.')
    data=np.asarray(integers)
    if data.ndim!=2 or data.shape[1]!=2 or data.dtype.kind not in 'iu':raise ValueError('Supply an N-by-2 integer array.')
    N=len(data)
    if N<3:raise ValueError('At least three observations are required.')
    if N>limits.max_observations:raise s.base.WorkLimit('max_observations')
    # Conversion to int64 occurs only after both bounds were checked.
    minimum=int(data.min());maximum=int(data.max());m=max(abs(minimum),abs(maximum))
    if max(m.bit_length(),exponent+1)>limits.max_input_bits:raise s.base.WorkLimit('max_input_bits')
    intermediate_bit_bound=2*m.bit_length()+2*exponent+8*N.bit_length()+32
    if intermediate_bit_bound>limits.max_rational_bits:raise s.base.WorkLimit('intermediate_integer_bits')
    if m*max(8,2*N)>=(1<<62):raise s.base.WorkLimit('int64_prefix_range')
    data=np.asarray(data,dtype=np.int64)
    count=Counter(limits);count.charge(2*N);den=1<<(2*exponent)
    count.check(den);n=N-1;nr=N-2
    norms=[[],[]];lags=[];reflections=[];bridges=[]
    for psi in (-1,0,1):
        residual=data[1:]-psi*data[:-1];count.charge(4*n)
        sums=[count.sum(residual[:,i]) for i in range(2)]
        for i in range(2):
            norm=Q(count.dot(residual[:,i],residual[:,i]))-Q(sums[i]*sums[i],n)
            norms[i].append(count.fraction(norm/den))
        e=residual[:,0];mean=Q(sums[0],n)
        lag=Q(count.dot(e[:-1],e[1:]))-mean*(2*sums[0]-int(e[0])-int(e[-1]))+(n-1)*mean*mean
        lags.append(count.fraction(lag/den))
        u=residual[:-1,0]+residual[1:,0];w=residual[:-1,1]-residual[1:,1]
        reflection=Q(count.dot(u,w),nr)-Q(count.sum(u)*count.sum(w),nr*nr)
        reflections.append(count.fraction(reflection/den))
        prefix=np.cumsum(e,dtype=np.int64)[:-1]
        idx=np.arange(1,n,dtype=np.int64)
        raw=Q(count.dot(prefix,prefix))-2*mean*count.dot(prefix,idx)+mean*mean*Q((n-1)*n*(2*n-1),6)
        bridges.append(count.fraction(raw/den))
    digest=hashlib.sha256(b'pre016-dyadic-v1\0'+str(exponent).encode()+b'\0'+data.astype('<i8',copy=False).tobytes()).hexdigest()
    refdigest=hashlib.sha256(b'pre016-dyadic-reference-v1\0'+str(exponent).encode()+b'\0'+data[:,0].astype('<i8').tobytes()).hexdigest()
    receipt={'engine':'dyadic','implementation':'checked_numpy_int64_chunks_and_python_int_totals',
             'common_dyadic_exponent':exponent,'maximum_integer_bits':count.bits,
             'operations':count.operations,'limits':limits.__dict__.copy(),
             'operation_accounting':'scalar integer additions and products charged by vector length; recursive limb products also charged; input checks and digest cost are not arithmetic operations',
             'intermediate_bit_bound':intermediate_bit_bound,'hash_schema':'pre016-dyadic-v1'}
    polynomials=s.base.frozen.RecordingPolynomials(N,tuple(_poly(v) for v in norms),_poly(lags),_poly(reflections),digest,refdigest,receipt)
    identity=s.base.RecordIdentity(N,digest,refdigest,limits,'dyadic',s.base.DEPENDENCIES)
    bank=s.base.BasePolynomials(identity,polynomials,dict(receipt,phase='base_moments'))
    bridge=s.base.BridgePolynomial(identity,_poly(bridges),dict(receipt,phase='bridge_moments'))
    return s.base.CompiledRecord(identity,bank,bridge,dict(receipt,phase='preparation'))
