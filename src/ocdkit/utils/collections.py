"""Dictionary and collection utilities."""

import sys


def sort_dict(d, reverse=True):
    """Sort a dict by value."""
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))


def unique_dict(x):
    """Return a ``{value: count}`` dict from an array, sorted by count."""
    import numpy as np
    unique, counts = np.unique(x, return_counts=True)
    return sort_dict({u: c for u, c in zip(unique, counts)})


def reduce_to_diff(strings, delim='_'):
    """Reduce a list of strings to only the tokens that differ across them.

    *delim* can be a single delimiter string or a list of delimiters.
    """
    import re
    if isinstance(delim, list):
        pattern = '|'.join(re.escape(d) for d in delim)
    else:
        pattern = re.escape(delim)

    token_lists = [re.split(pattern, s) for s in strings]
    max_len = max(len(tokens) for tokens in token_lists)

    diff_positions = []
    for i in range(max_len):
        col = [tokens[i] if i < len(tokens) else None for tokens in token_lists]
        if len(set(col)) > 1:
            diff_positions.append(i)

    joiner = delim[0] if isinstance(delim, list) else delim
    return [joiner.join(tokens[i] for i in diff_positions if i < len(tokens))
            for tokens in token_lists]


def num_elements(obj):
    """Recursively count scalar elements in nested lists/tuples/arrays."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.size
    elif isinstance(obj, (list, tuple)):
        return sum(num_elements(sub) for sub in obj)
    return 1


def get_size(obj):
    """Recursively estimate the memory footprint of *obj* in bytes."""
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_size(v) for v in obj.values())
        size += sum(get_size(k) for k in obj.keys())
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(v) for v in obj)
    return size


def map_lists(list1, list2):
    """Map elements of *list1* to their indices in *list2*.

    Returns a list of integers (or ``None`` for missing elements).
    """
    return [list2.index(item) if item in list2 else None for item in list1]


def invert_dict(d):
    """Swap keys and values, collecting into lists for many-to-one mappings.

    If a value in *d* is a list, each element becomes a key in the inverse.
    The result is sorted by key.
    """
    inverse = {}
    for k, values in d.items():
        if isinstance(values, list):
            for v in values:
                inverse.setdefault(v, []).append(k)
        else:
            inverse.setdefault(values, []).append(k)
    return {k: inverse[k] for k in sorted(inverse)}


def analyze_dict_types(d, prefix=""):
    """Print a recursive type summary of a nested dict (for debugging)."""
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}[{k}] is a nested dict:")
            analyze_dict_types(v, prefix=prefix + "    ")
        elif isinstance(v, list):
            print(f"{prefix}[{k}] is a list, first element type: {type(v[0]).__name__ if v else 'Empty'}")
        else:
            print(f"{prefix}[{k}] type: {type(v).__name__}")
