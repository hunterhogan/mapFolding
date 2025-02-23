from numba.core import types as types, typing as typing
from numba.core.cgutils import unpack_tuple as unpack_tuple
from numba.core.errors import RequireLiteralValue as RequireLiteralValue, TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic
from numba.core.imputils import impl_ret_new_ref as impl_ret_new_ref
from numba.cpython.unsafe.tuple import tuple_setitem as tuple_setitem

def empty_inferred(typingctx, shape):
    '''A version of numpy.empty whose dtype is inferred by the type system.

    Expects `shape` to be a int-tuple.

    There is special logic in the type-inferencer to handle the "refine"-ing
    of undefined dtype.
    '''
def to_fixed_tuple(typingctx, array, length):
    """Convert *array* into a tuple of *length*

    Returns ``UniTuple(array.dtype, length)``

    ** Warning **
    - No boundchecking.
      If *length* is longer than *array.size*, the behavior is undefined.
    """
