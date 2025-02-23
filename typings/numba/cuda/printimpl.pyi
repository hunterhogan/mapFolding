from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types
from numba.core.errors import NumbaWarning as NumbaWarning
from numba.core.imputils import Registry as Registry
from numba.cuda import nvvmutils as nvvmutils

registry: Incomplete
lower: Incomplete
voidptr: Incomplete

def print_item(ty, context, builder, val) -> None:
    """
    Handle printing of a single value of the given Numba type.
    A (format string, [list of arguments]) is returned that will allow
    forming the final printf()-like call.
    """
def int_print_impl(ty, context, builder, val): ...
def real_print_impl(ty, context, builder, val): ...
def const_print_impl(ty, context, builder, sigval): ...
def print_varargs(context, builder, sig, args):
    """This function is a generic 'print' wrapper for arbitrary types.
    It dispatches to the appropriate 'print' implementations above
    depending on the detected real types in the signature."""
