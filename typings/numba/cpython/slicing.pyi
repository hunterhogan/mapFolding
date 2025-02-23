from itertools import zip_longest as zip_longest
from numba.core import cgutils as cgutils, types as types, typing as typing, utils as utils
from numba.core.imputils import impl_ret_borrowed as impl_ret_borrowed, impl_ret_new_ref as impl_ret_new_ref, impl_ret_untracked as impl_ret_untracked, iternext_impl as iternext_impl, lower_builtin as lower_builtin, lower_cast as lower_cast, lower_constant as lower_constant, lower_getattr as lower_getattr

def fix_index(builder, idx, size):
    """
    Fix negative index by adding *size* to it.  Positive
    indices are left untouched.
    """
def fix_slice(builder, slice, size) -> None:
    """
    Fix *slice* start and stop to be valid (inclusive and exclusive, resp)
    indexing bounds for a sequence of the given *size*.
    """
def get_slice_length(builder, slicestruct):
    """
    Given a slice, compute the number of indices it spans, i.e. the
    number of iterations that for_range_slice() will execute.

    Pseudo-code:
        assert step != 0
        if step > 0:
            if stop <= start:
                return 0
            else:
                return (stop - start - 1) // step + 1
        else:
            if stop >= start:
                return 0
            else:
                return (stop - start + 1) // step + 1

    (see PySlice_GetIndicesEx() in CPython)
    """
def get_slice_bounds(builder, slicestruct):
    """
    Return the [lower, upper) indexing bounds of a slice.
    """
def fix_stride(builder, slice, stride):
    """
    Fix the given stride for the slice's step.
    """
def guard_invalid_slice(context, builder, typ, slicestruct) -> None:
    """
    Guard against *slicestruct* having a zero step (and raise ValueError).
    """
def get_defaults(context):
    """
    Get the default values for a slice's members:
    (start for positive step, start for negative step,
     stop for positive step, stop for negative step, step)
    """
def slice_constructor_impl(context, builder, sig, args): ...
def slice_start_impl(context, builder, typ, value): ...
def slice_stop_impl(context, builder, typ, value): ...
def slice_step_impl(context, builder, typ, value): ...
def slice_indices(context, builder, sig, args): ...
def make_slice_from_constant(context, builder, ty, pyval): ...
def constant_slice(context, builder, ty, pyval): ...
def cast_from_literal(context, builder, fromty, toty, val): ...
