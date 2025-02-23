from _typeshed import Incomplete
from numba import literal_unroll as literal_unroll, pndindex as pndindex
from numba.core import cgutils as cgutils, config as config, errors as errors, extending as extending, types as types, typing as typing
from numba.core.extending import intrinsic as intrinsic, overload as overload, overload_attribute as overload_attribute, overload_classmethod as overload_classmethod, overload_method as overload_method, register_jitable as register_jitable
from numba.core.imputils import RefType as RefType, impl_ret_borrowed as impl_ret_borrowed, impl_ret_new_ref as impl_ret_new_ref, impl_ret_untracked as impl_ret_untracked, iternext_impl as iternext_impl, lower_builtin as lower_builtin, lower_cast as lower_cast, lower_constant as lower_constant, lower_getattr as lower_getattr, lower_getattr_generic as lower_getattr_generic, lower_setattr_generic as lower_setattr_generic
from numba.core.types import StringLiteral as StringLiteral
from numba.core.typing import signature as signature
from numba.core.typing.npydecl import _choose_concatenation_layout as _choose_concatenation_layout, _parse_nested_sequence as _parse_nested_sequence, _sequence_of_arrays as _sequence_of_arrays
from numba.cpython import slicing as slicing
from numba.cpython.unsafe.tuple import build_full_slice_tuple as build_full_slice_tuple, tuple_setitem as tuple_setitem
from numba.misc import mergesort as mergesort, quicksort as quicksort
from numba.np.numpy_support import as_dtype as as_dtype, carray as carray, check_is_integer as check_is_integer, farray as farray, from_dtype as from_dtype, is_contiguous as is_contiguous, is_fortran as is_fortran, is_nonelike as is_nonelike, lt_complex as lt_complex, lt_floats as lt_floats, numpy_version as numpy_version, type_can_asarray as type_can_asarray, type_is_scalar as type_is_scalar

def set_range_metadata(builder, load, lower_bound, upper_bound) -> None:
    '''
    Set the "range" metadata on a load instruction.
    Note the interval is in the form [lower_bound, upper_bound).
    '''
def mark_positive(builder, load) -> None:
    """
    Mark the result of a load instruction as positive (or zero).
    """
def make_array(array_type):
    """
    Return the Structure representation of the given *array_type*
    (an instance of types.ArrayCompatible).

    Note this does not call __array_wrap__ in case a new array structure
    is being created (rather than populated).
    """
def get_itemsize(context, array_type):
    """
    Return the item size for the given array or buffer type.
    """
def load_item(context, builder, arrayty, ptr):
    """
    Load the item at the given array pointer.
    """
def store_item(context, builder, arrayty, val, ptr):
    """
    Store the item at the given array pointer.
    """
def fix_integer_index(context, builder, idxty, idx, size):
    """
    Fix the integer index' type and value for the given dimension size.
    """
def normalize_index(context, builder, idxty, idx):
    """
    Normalize the index type and value.  0-d arrays are converted to scalars.
    """
def normalize_indices(context, builder, index_types, indices):
    """
    Same as normalize_index(), but operating on sequences of
    index types and values.
    """
def populate_array(array, data, shape, strides, itemsize, meminfo, parent: Incomplete | None = None):
    """
    Helper function for populating array structures.
    This avoids forgetting to set fields.

    *shape* and *strides* can be Python tuples or LLVM arrays.
    """
def update_array_info(aryty, array) -> None:
    """
    Update some auxiliary information in *array* after some of its fields
    were changed.  `itemsize` and `nitems` are updated.
    """
def normalize_axis(func_name, arg_name, ndim, axis) -> None:
    """Constrain axis values to valid positive values."""
def normalize_axis_overloads(func_name, arg_name, ndim, axis): ...
def getiter_array(context, builder, sig, args): ...
def _getitem_array_single_int(context, builder, return_type, aryty, ary, idx):
    """ Evaluate `ary[idx]`, where idx is a single int. """
def iternext_array(context, builder, sig, args, result) -> None: ...
def basic_indexing(context, builder, aryty, ary, index_types, indices, boundscheck: Incomplete | None = None):
    """
    Perform basic indexing on the given array.
    A (data pointer, shapes, strides) tuple is returned describing
    the corresponding view.
    """
def make_view(context, builder, aryty, ary, return_type, data, shapes, strides):
    """
    Build a view over the given array with the given parameters.
    """
def _getitem_array_generic(context, builder, return_type, aryty, ary, index_types, indices):
    """
    Return the result of indexing *ary* with the given *indices*,
    returning either a scalar or a view.
    """
def getitem_arraynd_intp(context, builder, sig, args):
    """
    Basic indexing with an integer or a slice.
    """
def getitem_array_tuple(context, builder, sig, args):
    """
    Basic or advanced indexing with a tuple.
    """
def setitem_array(context, builder, sig, args):
    """
    array[a] = scalar_or_array
    array[a,..,b] = scalar_or_array
    """
def array_len(context, builder, sig, args): ...
def array_item(context, builder, sig, args): ...
def array_itemset(context, builder, sig, args): ...

class Indexer:
    """
    Generic indexer interface, for generating indices over a fancy indexed
    array on a single dimension.
    """
    def prepare(self) -> None:
        """
        Prepare the indexer by initializing any required variables, basic
        blocks...
        """
    def get_size(self) -> None:
        """
        Return this dimension's size as an integer.
        """
    def get_shape(self) -> None:
        """
        Return this dimension's shape as a tuple.
        """
    def get_index_bounds(self) -> None:
        """
        Return a half-open [lower, upper) range of indices this dimension
        is guaranteed not to step out of.
        """
    def loop_head(self) -> None:
        """
        Start indexation loop.  Return a (index, count) tuple.
        *index* is an integer LLVM value representing the index over this
        dimension.
        *count* is either an integer LLVM value representing the current
        iteration count, or None if this dimension should be omitted from
        the indexation result.
        """
    def loop_tail(self) -> None:
        """
        Finish indexation loop.
        """

class EntireIndexer(Indexer):
    """
    Compute indices along an entire array dimension.
    """
    context: Incomplete
    builder: Incomplete
    aryty: Incomplete
    ary: Incomplete
    dim: Incomplete
    ll_intp: Incomplete
    def __init__(self, context, builder, aryty, ary, dim) -> None: ...
    size: Incomplete
    index: Incomplete
    bb_start: Incomplete
    bb_end: Incomplete
    def prepare(self) -> None: ...
    def get_size(self): ...
    def get_shape(self): ...
    def get_index_bounds(self): ...
    def loop_head(self): ...
    def loop_tail(self) -> None: ...

class IntegerIndexer(Indexer):
    """
    Compute indices from a single integer.
    """
    context: Incomplete
    builder: Incomplete
    idx: Incomplete
    ll_intp: Incomplete
    def __init__(self, context, builder, idx) -> None: ...
    def prepare(self) -> None: ...
    def get_size(self): ...
    def get_shape(self): ...
    def get_index_bounds(self): ...
    def loop_head(self): ...
    def loop_tail(self) -> None: ...

class IntegerArrayIndexer(Indexer):
    """
    Compute indices from an array of integer indices.
    """
    context: Incomplete
    builder: Incomplete
    idxty: Incomplete
    idxary: Incomplete
    size: Incomplete
    ll_intp: Incomplete
    def __init__(self, context, builder, idxty, idxary, size) -> None: ...
    idx_size: Incomplete
    idx_index: Incomplete
    bb_start: Incomplete
    bb_end: Incomplete
    def prepare(self) -> None: ...
    def get_size(self): ...
    def get_shape(self): ...
    def get_index_bounds(self): ...
    def loop_head(self): ...
    def loop_tail(self) -> None: ...

class BooleanArrayIndexer(Indexer):
    """
    Compute indices from an array of boolean predicates.
    """
    context: Incomplete
    builder: Incomplete
    idxty: Incomplete
    idxary: Incomplete
    ll_intp: Incomplete
    zero: Incomplete
    def __init__(self, context, builder, idxty, idxary) -> None: ...
    size: Incomplete
    idx_index: Incomplete
    count: Incomplete
    bb_start: Incomplete
    bb_tail: Incomplete
    bb_end: Incomplete
    def prepare(self) -> None: ...
    def get_size(self): ...
    def get_shape(self): ...
    def get_index_bounds(self): ...
    def loop_head(self): ...
    def loop_tail(self) -> None: ...

class SliceIndexer(Indexer):
    """
    Compute indices along a slice.
    """
    context: Incomplete
    builder: Incomplete
    aryty: Incomplete
    ary: Incomplete
    dim: Incomplete
    idxty: Incomplete
    slice: Incomplete
    ll_intp: Incomplete
    zero: Incomplete
    def __init__(self, context, builder, aryty, ary, dim, idxty, slice) -> None: ...
    dim_size: Incomplete
    is_step_negative: Incomplete
    index: Incomplete
    count: Incomplete
    bb_start: Incomplete
    bb_end: Incomplete
    def prepare(self) -> None: ...
    def get_size(self): ...
    def get_shape(self): ...
    def get_index_bounds(self): ...
    def loop_head(self): ...
    def loop_tail(self) -> None: ...

class FancyIndexer:
    """
    Perform fancy indexing on the given array.
    """
    context: Incomplete
    builder: Incomplete
    aryty: Incomplete
    shapes: Incomplete
    strides: Incomplete
    ll_intp: Incomplete
    newaxes: Incomplete
    indexers: Incomplete
    def __init__(self, context, builder, aryty, ary, index_types, indices) -> None: ...
    indexers_shape: Incomplete
    def prepare(self) -> None: ...
    def get_shape(self):
        """
        Get the resulting data shape as Python tuple.
        """
    def get_offset_bounds(self, strides, itemsize):
        """
        Get a half-open [lower, upper) range of byte offsets spanned by
        the indexer with the given strides and itemsize.  The indexer is
        guaranteed to not go past those bounds.
        """
    def begin_loops(self): ...
    def end_loops(self) -> None: ...

def fancy_getitem(context, builder, sig, args, aryty, ary, index_types, indices): ...
def fancy_getitem_array(context, builder, sig, args):
    """
    Advanced or basic indexing with an array.
    """
def offset_bounds_from_strides(context, builder, arrty, arr, shapes, strides):
    """
    Compute a half-open range [lower, upper) of byte offsets from the
    array's data pointer, that bound the in-memory extent of the array.

    This mimics offset_bounds_from_strides() from
    numpy/core/src/private/mem_overlap.c
    """
def compute_memory_extents(context, builder, lower, upper, data):
    """
    Given [lower, upper) byte offsets and a base data pointer,
    compute the memory pointer bounds as pointer-sized integers.
    """
def get_array_memory_extents(context, builder, arrty, arr, shapes, strides, data):
    """
    Compute a half-open range [start, end) of pointer-sized integers
    which fully contain the array data.
    """
def extents_may_overlap(context, builder, a_start, a_end, b_start, b_end):
    """
    Whether two memory extents [a_start, a_end) and [b_start, b_end)
    may overlap.
    """
def maybe_copy_source(context, builder, use_copy, srcty, src, src_shapes, src_strides, src_data): ...
def _bc_adjust_dimension(context, builder, shapes, strides, target_shape):
    """
    Preprocess dimension for broadcasting.
    Returns (shapes, strides) such that the ndim match *target_shape*.
    When expanding to higher ndim, the returning shapes and strides are
    prepended with ones and zeros, respectively.
    When truncating to lower ndim, the shapes are checked (in runtime).
    All extra dimension must have size of 1.
    """
def _bc_adjust_shape_strides(context, builder, shapes, strides, target_shape):
    """
    Broadcast shapes and strides to target_shape given that their ndim already
    matches.  For each location where the shape is 1 and does not match the
    dim for target, it is set to the value at the target and the stride is
    set to zero.
    """
def _broadcast_to_shape(context, builder, arrtype, arr, target_shape):
    """
    Broadcast the given array to the target_shape.
    Returns (array_type, array)
    """
def _numpy_broadcast_to(typingctx, array, shape): ...
def get_readonly_array(typingctx, arr): ...
def _can_broadcast(array, dest_shape) -> None: ...
def _default_broadcast_to_impl(array, shape): ...
def numpy_broadcast_to(array, shape): ...
def numpy_broadcast_shapes_list(r, m, shape) -> None: ...
def ol_numpy_broadcast_shapes(*args): ...
def numpy_broadcast_arrays(*args): ...
def fancy_setslice(context, builder, sig, args, index_types, indices):
    """
    Implement slice assignment for arrays.  This implementation works for
    basic as well as fancy indexing, since there's no functional difference
    between the two for indexed assignment.
    """
def vararg_to_tuple(context, builder, sig, args): ...
def array_transpose(context, builder, sig, args): ...
def permute_arrays(axis, shape, strides) -> None: ...
def array_transpose_tuple(context, builder, sig, args): ...
def array_transpose_vararg(context, builder, sig, args): ...
def numpy_transpose(a, axes: Incomplete | None = None): ...
def array_T(context, builder, typ, value): ...
def numpy_logspace(start, stop, num: int = 50): ...
def numpy_geomspace(start, stop, num: int = 50): ...
def numpy_rot90(m, k: int = 1): ...
def _attempt_nocopy_reshape(context, builder, aryty, ary, newnd, newshape, newstrides):
    """
    Call into Numba_attempt_nocopy_reshape() for the given array type
    and instance, and the specified new shape.

    Return value is non-zero if successful, and the array pointed to
    by *newstrides* will be filled up with the computed results.
    """
def normalize_reshape_value(origsize, shape) -> None: ...
def array_reshape(context, builder, sig, args): ...
def array_reshape_vararg(context, builder, sig, args): ...
def np_reshape(a, newshape): ...
def numpy_resize(a, new_shape): ...
def np_append(arr, values, axis: Incomplete | None = None): ...
def array_ravel(context, builder, sig, args): ...
def np_ravel(context, builder, sig, args): ...
def array_flatten(context, builder, sig, args): ...
def _np_clip_impl(a, a_min, a_max, out): ...
def _np_clip_impl_none(a, b, use_min, out): ...
def np_clip(a, a_min, a_max, out: Incomplete | None = None): ...
def array_clip(a, a_min: Incomplete | None = None, a_max: Incomplete | None = None, out: Incomplete | None = None): ...
def _change_dtype(context, builder, oldty, newty, ary):
    """
    Attempt to fix up *ary* for switching from *oldty* to *newty*.

    See Numpy's array_descr_set()
    (np/core/src/multiarray/getset.c).
    Attempt to fix the array's shape and strides for a new dtype.
    False is returned on failure, True on success.
    """
def np_shape(a): ...
def np_size(a): ...
def np_unique(ar): ...
def np_repeat(a, repeats): ...
def np_repeat_impl_repeats_scaler(a, repeats): ...
def array_repeat(a, repeats): ...
def _intrin_get_itemsize(tyctx, dtype):
    """Computes the itemsize of the dtype"""
def _compatible_view(a, dtype) -> None: ...
def ol_compatible_view(a, dtype):
    """Determines if the array and dtype are compatible for forming a view."""
def array_view(context, builder, sig, args): ...
def array_dtype(context, builder, typ, value): ...
def array_shape(context, builder, typ, value): ...
def array_strides(context, builder, typ, value): ...
def array_ndim(context, builder, typ, value): ...
def array_size(context, builder, typ, value): ...
def array_itemsize(context, builder, typ, value): ...
def array_nbytes(context, builder, typ, value):
    """
    nbytes = size * itemsize
    """
def array_contiguous(context, builder, typ, value): ...
def array_c_contiguous(context, builder, typ, value): ...
def array_f_contiguous(context, builder, typ, value): ...
def array_readonly(context, builder, typ, value): ...
def array_ctypes(context, builder, typ, value): ...
def array_ctypes_data(context, builder, typ, value): ...
def array_ctypes_to_pointer(context, builder, fromty, toty, val): ...
def _call_contiguous_check(checker, context, builder, aryty, ary):
    """Helper to invoke the contiguous checker function on an array

    Args
    ----
    checker :
        ``numba.numpy_supports.is_contiguous``, or
        ``numba.numpy_supports.is_fortran``.
    context : target context
    builder : llvm ir builder
    aryty : numba type
    ary : llvm value
    """
def array_flags(context, builder, typ, value): ...
def array_flags_c_contiguous(context, builder, typ, value): ...
def array_flags_f_contiguous(context, builder, typ, value): ...
def array_real_part(context, builder, typ, value): ...
def array_imag_part(context, builder, typ, value): ...
def array_complex_attr(context, builder, typ, value, attr):
    """
    Given a complex array, it's memory layout is:

        R C R C R C
        ^   ^   ^

    (`R` indicates a float for the real part;
     `C` indicates a float for the imaginary part;
     the `^` indicates the start of each element)

    To get the real part, we can simply change the dtype and itemsize to that
    of the underlying float type.  The new layout is:

        R x R x R x
        ^   ^   ^

    (`x` indicates unused)

    A load operation will use the dtype to determine the number of bytes to
    load.

    To get the imaginary part, we shift the pointer by 1 float offset and
    change the dtype and itemsize.  The new layout is:

        x C x C x C
          ^   ^   ^
    """
def array_conj(arr): ...
def dtype_type(context, builder, dtypety, dtypeval): ...
def static_getitem_number_clazz(context, builder, sig, args):
    '''This handles the "static_getitem" when a Numba type is subscripted e.g:
    var = typed.List.empty_list(float64[::1, :])
    It only allows this on simple numerical types. Compound types, like
    records, are not supported.
    '''
def array_record_getattr(context, builder, typ, value, attr):
    """
    Generic getattr() implementation for record arrays: fetch the given
    record member, i.e. a subarray.
    """
def array_record_getitem(context, builder, sig, args): ...
def record_getattr(context, builder, typ, value, attr):
    """
    Generic getattr() implementation for records: get the given record member.
    """
def record_setattr(context, builder, sig, args, attr) -> None:
    """
    Generic setattr() implementation for records: set the given record member.
    """
def record_static_getitem_str(context, builder, sig, args):
    """
    Record.__getitem__ redirects to getattr()
    """
def record_static_getitem_int(context, builder, sig, args):
    """
    Record.__getitem__ redirects to getattr()
    """
def record_static_setitem_str(context, builder, sig, args):
    """
    Record.__setitem__ redirects to setattr()
    """
def record_static_setitem_int(context, builder, sig, args):
    """
    Record.__setitem__ redirects to setattr()
    """
def constant_array(context, builder, ty, pyval):
    """
    Create a constant array (mechanism is target-dependent).
    """
def constant_record(context, builder, ty, pyval):
    """
    Create a record constant as a stack-allocated array of bytes.
    """
def constant_bytes(context, builder, ty, pyval):
    """
    Create a constant array from bytes (mechanism is target-dependent).
    """
def array_is(context, builder, sig, args): ...
def ol_array_hash(arr): ...
def make_array_flat_cls(flatiterty):
    """
    Return the Structure representation of the given *flatiterty* (an
    instance of types.NumpyFlatType).
    """
def make_array_ndenumerate_cls(nditerty):
    """
    Return the Structure representation of the given *nditerty* (an
    instance of types.NumpyNdEnumerateType).
    """
def _increment_indices(context, builder, ndim, shape, indices, end_flag: Incomplete | None = None, loop_continue: Incomplete | None = None, loop_break: Incomplete | None = None) -> None: ...
def _increment_indices_array(context, builder, arrty, arr, indices, end_flag: Incomplete | None = None) -> None: ...
def make_nditer_cls(nditerty):
    """
    Return the Structure representation of the given *nditerty* (an
    instance of types.NumpyNdIterType).
    """
def make_ndindex_cls(nditerty):
    """
    Return the Structure representation of the given *nditerty* (an
    instance of types.NumpyNdIndexType).
    """
def _make_flattening_iter_cls(flatiterty, kind): ...
def make_array_flatiter(context, builder, arrty, arr): ...
def iternext_numpy_flatiter(context, builder, sig, args, result) -> None: ...
def iternext_numpy_getitem(context, builder, sig, args): ...
def iternext_numpy_getitem_any(context, builder, sig, args): ...
def iternext_numpy_getitem_flat(context, builder, sig, args): ...
def make_array_ndenumerate(context, builder, sig, args): ...
def iternext_numpy_nditer(context, builder, sig, args, result) -> None: ...
def make_array_ndindex(context, builder, sig, args):
    """ndindex(*shape)"""
def make_array_ndindex_tuple(context, builder, sig, args):
    """ndindex(shape)"""
def iternext_numpy_ndindex(context, builder, sig, args, result) -> None: ...
def make_array_nditer(context, builder, sig, args):
    """
    nditer(...)
    """
def iternext_numpy_nditer2(context, builder, sig, args, result) -> None: ...
def dtype_eq_impl(context, builder, sig, args): ...
def _empty_nd_impl(context, builder, arrtype, shapes):
    """Utility function used for allocating a new array during LLVM code
    generation (lowering).  Given a target context, builder, array
    type, and a tuple or list of lowered dimension sizes, returns a
    LLVM value pointing at a Numba runtime allocated array.
    """
def _ol_array_allocate(cls, allocsize, align):
    """Implements a Numba-only default target (cpu) classmethod on the array
    type.
    """
def _call_allocator(arrtype, size, align):
    """Trampoline to call the intrinsic used for allocation
    """
def intrin_alloc(typingctx, allocsize, align):
    """Intrinsic to call into the allocator for Array
    """
def _parse_shape(context, builder, ty, val):
    """
    Parse the shape argument to an array constructor.
    """
def _parse_empty_args(context, builder, sig, args):
    """
    Parse the arguments of a np.empty(), np.zeros() or np.ones() call.
    """
def _parse_empty_like_args(context, builder, sig, args):
    """
    Parse the arguments of a np.empty_like(), np.zeros_like() or
    np.ones_like() call.
    """
def _check_const_str_dtype(fname, dtype) -> None: ...
def numpy_empty_nd(tyctx, ty_shape, ty_dtype, ty_retty_ref): ...
def ol_np_empty(shape, dtype=...): ...
def numpy_empty_like_nd(tyctx, ty_prototype, ty_dtype, ty_retty_ref): ...
def ol_np_empty_like(arr, dtype: Incomplete | None = None): ...
def _zero_fill_array_method(tyctx, self): ...
def ol_array_zero_fill(self):
    """Adds a `._zero_fill` method to zero fill an array using memset."""
def ol_np_zeros(shape, dtype=...): ...
def ol_np_zeros_like(a, dtype: Incomplete | None = None): ...
def ol_np_ones_like(a, dtype: Incomplete | None = None): ...
def impl_np_full(shape, fill_value, dtype: Incomplete | None = None): ...
def impl_np_full_like(a, fill_value, dtype: Incomplete | None = None): ...
def ol_np_ones(shape, dtype: Incomplete | None = None): ...
def impl_np_identity(n, dtype: Incomplete | None = None): ...
def _eye_none_handler(N, M) -> None: ...
def _eye_none_handler_impl(N, M): ...
def numpy_eye(N, M: Incomplete | None = None, k: int = 0, dtype=...): ...
def impl_np_diag(v, k: int = 0): ...
def numpy_indices(dimensions): ...
def numpy_diagflat(v, k: int = 0): ...
def generate_getitem_setitem_with_axis(ndim, kind): ...
def numpy_take(a, indices, axis: Incomplete | None = None): ...
def _arange_dtype(*args): ...
def np_arange(start, /, stop: Incomplete | None = None, step: Incomplete | None = None, dtype: Incomplete | None = None): ...
def numpy_linspace(start, stop, num: int = 50): ...
def _array_copy(context, builder, sig, args):
    """
    Array copy.
    """
def _array_copy_intrinsic(typingctx, a): ...
def array_copy(context, builder, sig, args): ...
def impl_numpy_copy(a): ...
def _as_layout_array(context, builder, sig, args, output_layout):
    """
    Common logic for layout conversion function;
    e.g. ascontiguousarray and asfortranarray
    """
def _as_layout_array_intrinsic(typingctx, a, output_layout): ...
def array_ascontiguousarray(a): ...
def array_asfortranarray(a): ...
def array_astype(context, builder, sig, args): ...
def np_frombuffer(typingctx, buffer, dtype, retty): ...
def impl_np_frombuffer(buffer, dtype=...): ...
def impl_carray(ptr, shape, dtype: Incomplete | None = None): ...
def impl_farray(ptr, shape, dtype: Incomplete | None = None): ...
def get_cfarray_intrinsic(layout, dtype_): ...
def np_cfarray(context, builder, sig, args):
    """
    numba.numpy_support.carray(...) and
    numba.numpy_support.farray(...).
    """
def _get_seq_size(context, builder, seqty, seq): ...
def _get_borrowing_getitem(context, seqty):
    """
    Return a getitem() implementation that doesn't incref its result.
    """
def compute_sequence_shape(context, builder, ndim, seqty, seq):
    """
    Compute the likely shape of a nested sequence (possibly 0d).
    """
def check_sequence_shape(context, builder, seqty, seq, shapes) -> None:
    """
    Check the nested sequence matches the given *shapes*.
    """
def assign_sequence_to_array(context, builder, data, shapes, strides, arrty, seqty, seq) -> None:
    """
    Assign a nested sequence contents to an array.  The shape must match
    the sequence's structure.
    """
def np_array_typer(typingctx, object, dtype): ...
def np_array(typingctx, obj, dtype): ...
def impl_np_array(object, dtype: Incomplete | None = None): ...
def _normalize_axis(context, builder, func_name, ndim, axis): ...
def _insert_axis_in_shape(context, builder, orig_shape, ndim, axis):
    """
    Compute shape with the new axis inserted
    e.g. given original shape (2, 3, 4) and axis=2,
    the returned new shape is (2, 3, 1, 4).
    """
def _insert_axis_in_strides(context, builder, orig_strides, ndim, axis):
    """
    Same as _insert_axis_in_shape(), but with a strides array.
    """
def expand_dims(context, builder, sig, args, axis):
    """
    np.expand_dims() with the given axis.
    """
def np_expand_dims(typingctx, a, axis): ...
def impl_np_expand_dims(a, axis): ...
def _atleast_nd(minimum, axes): ...
def _atleast_nd_transform(min_ndim, axes):
    """
    Return a callback successively inserting 1-sized dimensions at the
    following axes.
    """
def np_atleast_1d(*args): ...
def np_atleast_2d(*args): ...
def np_atleast_3d(*args): ...
def _do_concatenate(context, builder, axis, arrtys, arrs, arr_shapes, arr_strides, retty, ret_shapes):
    """
    Concatenate arrays along the given axis.
    """
def _np_concatenate(context, builder, arrtys, arrs, retty, axis): ...
def _np_stack(context, builder, arrtys, arrs, retty, axis): ...
def np_concatenate_typer(typingctx, arrays, axis): ...
def np_concatenate(typingctx, arrays, axis): ...
def impl_np_concatenate(arrays, axis: int = 0): ...
def _column_stack_dims(context, func_name, arrays): ...
def np_column_stack(typingctx, tup): ...
def impl_column_stack(tup): ...
def _np_stack_common(context, builder, sig, args, axis):
    """
    np.stack() with the given axis value.
    """
def np_stack_common(typingctx, arrays, axis): ...
def impl_np_stack(arrays, axis: int = 0): ...
def NdStack_typer(typingctx, func_name, arrays, ndim_min): ...
def _np_hstack(typingctx, tup): ...
def impl_np_hstack(tup): ...
def _np_vstack(typingctx, tup): ...
def impl_np_vstack(tup): ...
def _np_dstack(typingctx, tup): ...
def impl_np_dstack(tup): ...
def arr_fill(arr, val): ...
def array_dot(arr, other): ...
def np_flip_lr(m): ...
def np_flip_ud(m): ...
def _build_flip_slice_tuple(tyctx, sz):
    """ Creates a tuple of slices for np.flip indexing like
    `(slice(None, None, -1),) * sz` """
def np_flip(m): ...
def np_array_split(ary, indices_or_sections, axis: int = 0): ...
def np_split(ary, indices_or_sections, axis: int = 0): ...
def numpy_vsplit(ary, indices_or_sections): ...
def numpy_hsplit(ary, indices_or_sections): ...
def numpy_dsplit(ary, indices_or_sections): ...

_sorts: Incomplete

def default_lt(a, b):
    """
    Trivial comparison function between two keys.
    """
def get_sort_func(kind, lt_impl, is_argsort: bool = False):
    """
    Get a sort implementation of the given kind.
    """
def lt_implementation(dtype): ...
def array_sort(context, builder, sig, args): ...
def impl_np_sort(a): ...
def array_argsort(context, builder, sig, args): ...
def array_to_array(context, builder, fromty, toty, val): ...
def array0d_to_scalar(context, builder, fromty, toty, val): ...
def array_to_unichrseq(context, builder, fromty, toty, val): ...
def reshape_unchecked(a, shape, strides) -> None:
    """
    An intrinsic returning a derived array with the given shape and strides.
    """
def type_reshape_unchecked(context): ...
def impl_shape_unchecked(context, builder, sig, args): ...
def as_strided(x, shape: Incomplete | None = None, strides: Incomplete | None = None): ...
def sliding_window_view(x, window_shape, axis: Incomplete | None = None): ...
def ol_bool(arr): ...
def numpy_swapaxes(a, axis1, axis2): ...
def _take_along_axis_impl(arr, indices, axis, Ni_orig, Nk_orig, indices_broadcast_shape): ...
def arr_take_along_axis(arr, indices, axis): ...
def nan_to_num_impl(x, copy: bool = True, nan: float = 0.0): ...
