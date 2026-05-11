"""Keyword-argument routing and coercion utilities."""

from __future__ import annotations

import functools
import inspect
from collections import defaultdict
from typing import Sequence, Mapping, Callable, Dict, Any

import numpy as np


def split_kwargs(
    funcs: Sequence[Callable],
    kwargs: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Dict[Callable, Dict[str, Any]]:
    """Split *kwargs* into per-function dictionaries.

    Each output bucket is padded with the function's parameter defaults
    so the caller sees a complete ``{name: value}`` snapshot for every
    function — user-passed keys override the defaults. This matches the
    expectation that "defaults always pass through; deviations are
    explicit."

    Downstream callers that compute a value to substitute for an
    unspecified kwarg should treat ``None`` as "not specified" rather
    than using ``setdefault`` (which can't distinguish a user-passed
    ``None`` from an auto-filled signature default of ``None``)::

        if kwargs.get('intervals') is None:
            kwargs['intervals'] = computed_intervals

    Keys that match *no* explicit parameter are routed to the **first**
    callable that accepts a ``**kwargs`` argument (VAR_KEYWORD). Only when
    *none* of *funcs* can accept arbitrary keywords does ``strict=True``
    raise an error.
    """
    dispatch: Dict[str, set] = {}
    for fn in funcs:
        for p in inspect.signature(fn).parameters.values():
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            dispatch.setdefault(p.name, set()).add(fn)

    catch_all = [
        fn for fn in funcs
        if any(p.kind == inspect.Parameter.VAR_KEYWORD
               for p in inspect.signature(fn).parameters.values())
    ]

    out = defaultdict(dict)
    unknown = {}

    for k, v in kwargs.items():
        fns = dispatch.get(k)

        if fns is None:
            if catch_all:
                out[catch_all[0]][k] = v
            elif strict:
                unknown[k] = v
            continue

        for fn in fns:
            out[fn][k] = v

    if strict and unknown and not catch_all:
        raise TypeError(f"Unknown parameters: {set(unknown)}")

    for fn in funcs:
        out.setdefault(fn, {})

    # Pad each bucket with parameter defaults — caller-supplied keys
    # already populated above take precedence.
    for fn in funcs:
        sig = inspect.signature(fn)
        fn_kwargs = out[fn]
        for name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                               inspect.Parameter.VAR_KEYWORD):
                continue
            if name not in fn_kwargs and param.default is not inspect._empty:
                fn_kwargs[name] = param.default

    return [out[f] for f in funcs] if len(funcs) > 1 else out[funcs[0]]


def base_kwargs(local_vars, *, exclude=None):
    """Extract kwargs from a function's local variables dict."""
    exclude = set(exclude or ())
    base = {k: v for k, v in local_vars.items() if k not in exclude}
    base.update(local_vars.get("kwargs", {}))
    return base


def split_kwargs_for(func, local_vars, *, exclude=None):
    """Split local variables into kwargs matching *func*'s signature."""
    base = base_kwargs(local_vars, exclude=exclude)
    return split_kwargs([func], base, strict=False)


def listify_args(*names):
    """Decorator to auto-convert parameters to lists.

    When used bare (``@listify_args``), converts every parameter whose
    name ends in ``"channels"`` or ``"rounds"`` to a list.

    When used with arguments (``@listify_args('foo', 'bar')``), converts
    only the named parameters.
    """
    if names and callable(names[0]) and len(names) == 1:
        fn = names[0]
        auto = [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.name.endswith(("channels", "rounds"))
        ]
        return listify_args(*auto)(fn)

    def _decorator(fn):
        sig = inspect.signature(fn)
        targets = names or [
            p.name
            for p in sig.parameters.values()
            if p.name.endswith(("channels", "rounds"))
        ]

        def _wrapper(*args,
                     __sig=sig,
                     __targets=tuple(targets),
                     __to_list=_to_list,
                     __fn=fn,
                     **kwargs):
            bound = __sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for n in __targets:
                bound.arguments[n] = __to_list(bound.arguments[n])
            return __fn(*bound.args, **bound.kwargs)

        return functools.wraps(fn)(_wrapper)
    return _decorator


def _to_list(x):
    """Coerce *x* to a list."""
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [x]
