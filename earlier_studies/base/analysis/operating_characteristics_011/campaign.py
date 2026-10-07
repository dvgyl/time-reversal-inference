"""Generate one record for the fixed earlier operating-characteristics design."""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.signal import lfilter
from inputs.reference import CalibrationData, array_sha256, run_test
from factored import evaluate_cached, with_envelope

HERE = Path(__file__).resolve().parent
INPUT_LIMIT = 10*2**30
OUTPUT_LIMIT = 2*2**30
WALL_LIMIT = 16*3600
MEMORY_LIMIT = 2**30


def peak_resident_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    if path.exists():
        raise RuntimeError('Preserve the existing file. Use a new output path.')
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def rng(design_id, replicate_id, stream_id):
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([20261011,11,design_id,replicate_id,stream_id])))


def generate(design, replicate, matrix):
    number, count = design['design_id'], max(design['lengths'])
    raw = {}
    for channel, coefficients in enumerate(design['coefficients']):
        noise = rng(number,replicate,channel).standard_normal(len(matrix))
        raw[f'calibration_response_{channel}'] = matrix@np.asarray(coefficients)+.02*noise
    if design['source'] == 'AR1':
        phi = design['phi']
        paths = []
        for label, initial_stream, innovation_stream in [('x',10,11),('z',20,21)]:
            initial = rng(number,replicate,initial_stream).standard_normal(1)
            innovations = rng(number,replicate,innovation_stream).standard_normal(count+3)
            tail, _ = lfilter([math.sqrt(1-phi**2)], [1,-phi], innovations,
                              zi=[phi*initial[0]])
            paths.append(np.concatenate([initial,tail]))
            raw[label+'_initial'] = initial
            raw[label+'_innovations'] = innovations
        first = paths[0]
        lag = int(design['hypothesis'] == 'alternative')
        second = .4*paths[0][1-lag:len(paths[0])-lag]+math.sqrt(.84)*paths[1][1:]
        # first starts at -4. second starts at -3 in both source laws.
        source_paths = [(first,-4),(second,-3)]
    else:
        initial = rng(number,replicate,10).standard_normal(2)
        innovations = rng(number,replicate,11).standard_normal((count+3,2))
        damping, omega = design['damping'], design['omega']
        coefficient = damping*complex(math.cos(omega),math.sin(omega))
        tail, _ = lfilter([math.sqrt(1-damping**2)], [1,-coefficient],
                          innovations[:,0]+1j*innovations[:,1],
                          zi=[coefficient*complex(*initial)])
        first = np.concatenate([[initial[0]],tail.real])
        raw['rotation_initial'] = initial
        raw['rotation_innovations'] = innovations
        source_paths = [(first,-4),(first,-4)]
    recording = np.empty((count,2),dtype=np.float64)
    for channel, ((path,start), coefficients) in enumerate(zip(source_paths,design['coefficients'])):
        filtered = np.zeros(count)
        for position, coefficient in enumerate(coefficients):
            begin = -position-start
            filtered += coefficient*path[begin:begin+count]
        recording[:,channel] = filtered+[2.,-3.][channel]
    raw['recording'] = recording
    if not all(np.isfinite(value).all() for value in raw.values()):
        raise ArithmeticError('Generated inputs contain a non-finite value.')
    return raw


