"""Generic Result container for named returns with tuple-like unpacking."""


class Result:
    """Lightweight named-result container.

    Supports attribute access, integer/slice indexing, and tuple unpacking::

        r = Result(masks=m, flows=f)
        r.masks                # attribute access
        r[0]                   # integer index
        r[:-1]                 # slice
        masks, flows = r       # unpacking
    """

    def __init__(self, **kwargs):
        self._fields = tuple(kwargs.keys())
        self._values = tuple(kwargs.values())
        self.__dict__.update(kwargs)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._values[key]
        return getattr(self, key)

    def __len__(self):
        return len(self._fields)

    def __repr__(self):
        items = ', '.join(f'{k}=...' for k in self._fields)
        return f'Result({items})'
