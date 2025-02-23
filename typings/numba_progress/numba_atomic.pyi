__all__ = ['atomic_add', 'atomic_sub', 'atomic_max', 'atomic_min', 'atomic_xchg']

def atomic_add(ary, i, v):
    """
    Atomically, perform `ary[i] += v` and return the previous value of `ary[i]`.

    i must be a simple index for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
def atomic_sub(ary, i, v):
    """
    Atomically, perform `ary[i] -= v` and return the previous value of `ary[i]`.

    i must be a simple index for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
def atomic_max(ary, i, v):
    """
    Atomically, perform `ary[i] = max(ary[i], v)` and return the previous value of `ary[i]`.
    This operation does not support floating-point values.

    i must be a simple index for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
def atomic_min(ary, i, v):
    """
    Atomically, perform `ary[i] = min(ary[i], v)` and return the previous value of `ary[i]`.
    This operation does not support floating-point values.

    i must be a simple index for a single element of ary. Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
def atomic_xchg(ary, i, v): ...
