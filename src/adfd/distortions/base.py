import hashlib
import inspect

import numpy as np

REGISTRY = {}


def register(name):
    def wrap(function):
        REGISTRY[name] = function
        return function

    return wrap


def get_distortion(name):
    return REGISTRY[name]


def apply_distortion(kind, signal, sr, level, options=None, salt=0):
    function = REGISTRY[kind]
    kwargs = dict(options or {})
    if "salt" in inspect.signature(function).parameters:
        kwargs["salt"] = salt
    return function(signal, sr, level, **kwargs)


def available_distortions():
    return sorted(REGISTRY)


def file_salt(file_id):
    digest = hashlib.blake2b(str(file_id).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def rng_for(level, length, salt=0):
    seed = (int(round(float(level) * 1000)) + length * 2654435761 + int(salt)) % (2 ** 32)
    return np.random.default_rng(seed)
