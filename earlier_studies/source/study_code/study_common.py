from dataclasses import asdict, is_dataclass
from fractions import Fraction as Q
import hashlib
import json
import os
from pathlib import Path

LENGTHS = (4097, 32769)
REPLICATES = 64
MASTER_SEED = 2026100312
BANK = (Q(0), Q(1,2), Q(4,5), Q(9,10))
CELLS = tuple(dict(cell_id=i, phi=phi, response=h, delay=d, correlation=c)
              for i,(phi,h,d,c) in enumerate((phi,h,d,c)
              for phi in (Q(1,5),Q(4,5),Q(97,100))
              for h in ((Q(1),Q(0)),(Q(1),Q(1,10)))
              for d,c in ((0,Q(9,10)),(1,Q(2,5)),(1,Q(9,10)))))
DESIGN = tuple(tuple(1 if (i//(2**j))%2 == 0 else -1 for j in range(2)) for i in range(128))


def encode(value):
    if isinstance(value,Q):
        return dict(numerator_hex=hex(value.numerator),denominator_hex=hex(value.denominator))
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def decode(value):
    if set(value) == {'numerator_hex','denominator_hex'}:
        return Q(int(value['numerator_hex'],16),int(value['denominator_hex'],16))
    return value


def save(path,value):
    data=(json.dumps(value,default=encode,sort_keys=True,indent=2)+'\n').encode()
    path=Path(path)
    pending=path.with_name(path.name+'.pending')
    with pending.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending,path)
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),object_hook=decode)


def array_hash(array):
    return hashlib.sha256(array.astype('<f8',copy=False).tobytes(order='C')).hexdigest()
