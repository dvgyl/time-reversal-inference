"""Generate only through the locked worker entry point."""
import numpy as np
from scipy.signal import lfilter
from study_common import MASTER_SEED, DESIGN, LENGTHS, array_hash


def stationary_from_normals(normals, phi):
    initial=float(normals[0])
    values=np.empty(len(normals),dtype=np.float64)
    values[0]=initial
    values[1:]=lfilter([np.sqrt(1.0-phi*phi)],[1.0,-phi],normals[1:],zi=[phi*initial])[0]
    return values


def assemble(cell, x_normals, z_normals, calibration_normals):
    phi,c=float(cell['phi']),float(cell['correlation'])
    h0,h1=map(float,cell['response'])
    d=cell['delay']
    x=stationary_from_normals(x_normals,phi)
    z=stationary_from_normals(z_normals,phi)
    length=len(x)-2
    if len(z)!=len(x) or len(calibration_normals)!=128:
        raise ValueError('The primitive dimensions do not match.')
    indices=np.arange(length)+2
    mix=np.sqrt(1.0-c*c)
    second0=c*x[indices-d]+mix*z[indices]
    second1=c*x[indices-1-d]+mix*z[indices-1]
    record=np.column_stack((x[indices]+2.0,h0*second0+h1*second1-1.0))
    rows=np.asarray(DESIGN,dtype=np.float64)
    response=h0*rows[:,0]+h1*rows[:,1]+0.01*calibration_normals
    return record,response


def draw_locked_job(cell, replicate, lock_receipt):
    if lock_receipt.get('observation_authorized') is not True:
        raise RuntimeError('A final observation lock is required.')
    primitives=[]
    initial_states=[]
    for stream,size in ((0,LENGTHS[-1]+2),(1,LENGTHS[-1]+2),(2,128)):
        seed=np.random.SeedSequence(MASTER_SEED,spawn_key=(cell['cell_id'],replicate,stream))
        rng=np.random.Generator(np.random.PCG64(seed))
        initial_states.append(rng.bit_generator.state)
        primitives.append(rng.standard_normal(size))
    record,response=assemble(cell,*primitives)
    receipt=dict(master_seed=MASTER_SEED,spawn_keys=[(cell['cell_id'],replicate,s) for s in range(3)],
                 initial_generator_states=initial_states, generator_algorithm='PCG64',
                 primitive_hashes=[array_hash(a) for a in primitives],
                 primitive_lengths=[len(a) for a in primitives],record_hash=array_hash(record),
                 calibration_response_hash=array_hash(response),source_index_range=(-2,LENGTHS[-1]-1))
    return record,response,receipt
