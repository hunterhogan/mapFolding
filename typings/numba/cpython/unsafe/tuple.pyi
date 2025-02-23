from numba.core import errors as errors, types as types, typing as typing
from numba.core.cgutils import alloca_once as alloca_once
from numba.core.extending import intrinsic as intrinsic

def tuple_setitem(typingctx, tup, idx, val):
    """Return a copy of the tuple with item at *idx* replaced with *val*.

    Operation: ``out = tup[:idx] + (val,) + tup[idx + 1:]

    **Warning**

    - No boundchecking.
    - The dtype of the tuple cannot be changed.
      *val* is always cast to the existing dtype of the tuple.
    """
def build_full_slice_tuple(tyctx, sz):
    """Creates a sz-tuple of full slices."""
def unpack_single_tuple(tyctx, tup):
    """This exists to handle the situation y = (*x,), the interpreter injects a
    call to it in the case of a single value unpack. It's not possible at
    interpreting time to differentiate between an unpack on a variable sized
    container e.g. list and a fixed one, e.g. tuple. This function handles the
    situation should it arise.
    """
