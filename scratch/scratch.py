import toolz


def flatten_dict(d: dict) -> dict:
    def flatten_dict(d: dict, acc, parent_key: str = None, sep="-"):
        for k, v in d.items():
            if parent_key:
                k = parent_key + sep + k
            if isinstance(v, dict):
                flatten_dict(v, acc, k, sep)
            else:
                acc[k] = v
        return acc

    return flatten_dict(d, {})


nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)  # {'a': 1, 'b-c': 2, 'b-d-e': 3}
