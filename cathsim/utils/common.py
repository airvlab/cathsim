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
