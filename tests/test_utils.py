import pytest
from cathsim.utils import expand_dict, flatten_dict


expand_dict_data = [
    (
        {"a": [1, 2, 3]},
        {"a": 7},
        {"a": [1, 2, 3, 7]},
    ),  # Test with simple dictionary
    (
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        {"a": 7, "b": 8},
        {"a": [1, 2, 3, 7], "b": [4, 5, 6, 8]},
    ),  # Test with multiple keys
    (
        {"a": [1, 2, 3], "b": {"c": [4, 5, 6]}},
        {"a": 7, "b": {"c": 8}},
        {"a": [1, 2, 3, 7], "b": {"c": [4, 5, 6, 8]}},
    ),  # Test with nested dictionary
    (
        {"a": [1, 2, 3], "b": {"c": [4, 5, 6], "d": [7, 8, 9]}},
        {"a": 10, "b": {"c": 11, "d": 12}},
        {"a": [1, 2, 3, 10], "b": {"c": [4, 5, 6, 11], "d": [7, 8, 9, 12]}},
    ),  # Test with deeper nested dictionary
]


@pytest.mark.parametrize("xd, yd, expected", expand_dict_data)
def test_expand_dict(xd, yd, expected):
    zd = expand_dict(xd, yd)
    assert zd == expected, "expand_dict function doesn't work properly"


flatten_dict_test_data = [
    (
        {"a": {"b": 1, "c": 2}},
        "a-b",
        "a-c",
        {"a-b": 1, "a-c": 2},
    ),  # Test with one nested dictionary
    (
        {"a": {"b": {"c": 1}, "d": 2}},
        "a-b-c",
        "a-d",
        {"a-b-c": 1, "a-d": 2},
    ),  # Test with deeper nested dictionary
    (
        {"a": {"b": {"c": {"d": 1}}}},
        "a-b-c-d",
        None,
        {"a-b-c-d": 1},
    ),  # Test with even deeper nested dictionary
]


@pytest.mark.parametrize("d, key1, key2, expected", flatten_dict_test_data)
def test_flatten_dict(d, key1, key2, expected):
    flat = flatten_dict(d)
    for key in [key1, key2]:
        actual = flat.get(key)
        expected_val = expected.get(key)
        assert flat.get(key1) == expected.get(
            key1
        ), f"expected val != actual: {expected_val} != {actual}"


# Define the test data outside of the test function so that other test functions can access it
mapd_test_data = [
    # Test with a single level dictionary, applying the 'len' function to each value
    ({"a": [1, 2, 3], "b": [4, 5]}, len, None, {"a": 3, "b": 2}),
    # Test with a nested dictionary, applying the 'sum' function to each value
    ({"a": {"b": [1, 2, 3]}, "c": [4, 5]}, sum, None, {"a": {"b": 6}, "c": 9}),
    # Test with a deeply nested dictionary, applying the 'len' function to each value
    (
        {"a": {"b": {"c": [1, 2, 3, 4]}}, "d": [5, 6]},
        len,
        None,
        {"a": {"b": {"c": 4}}, "d": 2},
    ),
]


@pytest.mark.parametrize("d, fn, key, expected", mapd_test_data)
def test_mapd(d, fn, key, expected):
    mapped = mapd(d, fn, key)
    assert mapped == expected, "mapd function doesn't work properly"
