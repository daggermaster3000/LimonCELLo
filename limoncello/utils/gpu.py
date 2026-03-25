# utils/gpu.py

import pyclesperanto_prototype as cle


def to_gpu(x):
    return cle.push(x)


def to_cpu(x):
    return cle.pull(x)