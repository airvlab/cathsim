from toolz.dicttoolz import itemmap
import numpy as np

CAMERA_MATRICES = {
    "80": np.array(
        [
            [-5.79411255e02, 0.00000000e00, 2.39500000e02, -5.33073376e01],
            [0.00000000e00, 5.79411255e02, 2.39500000e02, -1.08351407e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, -1.50000000e-01],
        ]
    ),
    "480": np.array(
        [
            [-5.79411255e02, 0.00000000e00, 2.39500000e02, -5.33073376e01],
            [0.00000000e00, 5.79411255e02, 2.39500000e02, -1.08351407e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, -1.50000000e-01],
        ],
    ),
}


def flatten_dict(d: dict, parent_key: str = None) -> dict:
    acc = {}
    for k, v in d.items():
        if parent_key:
            k = parent_key + "-" + k
        if isinstance(v, dict):
            acc = acc | flatten_dict(v, k)
        else:
            acc[k] = v
    return acc


def expand_dict(xd: dict, yd: dict) -> dict:
    zd = xd.copy()
    for k, v in yd.items():
        if k not in xd:
            zd[k] = [v]
        elif isinstance(v, dict) and isinstance(xd[k], dict):
            zd[k] = expand_dict(xd[k], v)
        else:
            zd[k] = xd[k] + [v]
    return zd


def map_val(g: callable, d: dict):
    def f(item):
        k, v = item
        if isinstance(v, dict):
            return (k, itemmap(f, v))
        else:
            return (k, g(v))

    return itemmap(f, d)
