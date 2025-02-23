from _typeshed import Incomplete
from numba.core import config as config, errors as errors, types as types
from numba.core.cgutils import is_nonelike as is_nonelike
from numba.core.errors import TypingError as TypingError
from numba.core.typing.templates import signature as signature
from numba.np import npdatetime_helpers as npdatetime_helpers
from typing import NamedTuple

numpy_version: Incomplete
FROM_DTYPE: Incomplete
re_typestr: Incomplete
re_datetimestr: Incomplete
sizeof_unicode_char: Incomplete

def _from_str_dtype(dtype): ...
def _from_datetime_dtype(dtype): ...
def from_dtype(dtype):
    """
    Return a Numba Type instance corresponding to the given Numpy *dtype*.
    NumbaNotImplementedError is raised on unsupported Numpy dtypes.
    """

_as_dtype_letters: Incomplete

def as_dtype(nbtype):
    """
    Return a numpy dtype instance corresponding to the given Numba type.
    NotImplementedError is if no correspondence is known.
    """
def as_struct_dtype(rec):
    """Convert Numba Record type to NumPy structured dtype
    """
def _check_struct_alignment(rec, fields) -> None:
    """Check alignment compatibility with Numpy"""
def map_arrayscalar_type(val): ...
def is_array(val): ...
def map_layout(val): ...
def select_array_wrapper(inputs):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    select the appropriate input for wrapping the operation output,
    according to each input's __array_priority__.

    An index into *inputs* is returned.
    """
def resolve_output_type(context, inputs, formal_output):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    and the operation's formal output type (a types.Array instance),
    resolve the actual output type using the typing *context*.

    This uses a mechanism compatible with Numpy's __array_priority__ /
    __array_wrap__.
    """
def supported_ufunc_loop(ufunc, loop):
    """Return whether the *loop* for the *ufunc* is supported -in nopython-.

    *loop* should be a UFuncLoopSpec instance, and *ufunc* a numpy ufunc.

    For ufuncs implemented using the ufunc_db, it is supported if the ufunc_db
    contains a lowering definition for 'loop' in the 'ufunc' entry.

    For other ufuncs, it is type based. The loop will be considered valid if it
    only contains the following letter types: '?bBhHiIlLqQfd'. Note this is
    legacy and when implementing new ufuncs the ufunc_db should be preferred,
    as it allows for a more fine-grained incremental support.
    """

class UFuncLoopSpec(NamedTuple('_UFuncLoopSpec', [('inputs', Incomplete), ('outputs', Incomplete), ('ufunc_sig', Incomplete)])):
    '''
    An object describing a ufunc loop\'s inner types.  Properties:
    - inputs: the inputs\' Numba types
    - outputs: the outputs\' Numba types
    - ufunc_sig: the string representing the ufunc\'s type signature, in
      Numpy format (e.g. "ii->i")
    '''
    __slots__: Incomplete
    @property
    def numpy_inputs(self): ...
    @property
    def numpy_outputs(self): ...

def _ufunc_loop_sig(out_tys, in_tys): ...
def ufunc_can_cast(from_, to, has_mixed_inputs, casting: str = 'safe'):
    """
    A variant of np.can_cast() that can allow casting any integer to
    any real or complex type, in case the operation has mixed-kind
    inputs.

    For example we want `np.power(float32, int32)` to be computed using
    SP arithmetic and return `float32`.
    However, `np.sqrt(int32)` should use DP arithmetic and return `float64`.
    """
def ufunc_find_matching_loop(ufunc, arg_types):
    """Find the appropriate loop to be used for a ufunc based on the types
    of the operands

    ufunc        - The ufunc we want to check
    arg_types    - The tuple of arguments to the ufunc, including any
                   explicit output(s).
    return value - A UFuncLoopSpec identifying the loop, or None
                   if no matching loop is found.
    """
def _is_aligned_struct(struct): ...
def from_struct_dtype(dtype):
    """Convert a NumPy structured dtype to Numba Record type
    """
def _get_bytes_buffer(ptr, nbytes):
    """
    Get a ctypes array of *nbytes* starting at *ptr*.
    """
def _get_array_from_ptr(ptr, nbytes, dtype): ...
def carray(ptr, shape, dtype: Incomplete | None = None):
    """
    Return a Numpy array view over the data pointed to by *ptr* with the
    given *shape*, in C order.  If *dtype* is given, it is used as the
    array's dtype, otherwise the array's dtype is inferred from *ptr*'s type.
    """
def farray(ptr, shape, dtype: Incomplete | None = None):
    """
    Return a Numpy array view over the data pointed to by *ptr* with the
    given *shape*, in Fortran order.  If *dtype* is given, it is used as the
    array's dtype, otherwise the array's dtype is inferred from *ptr*'s type.
    """
def is_contiguous(dims, strides, itemsize):
    """Is the given shape, strides, and itemsize of C layout?

    Note: The code is usable as a numba-compiled function
    """
def is_fortran(dims, strides, itemsize):
    """Is the given shape, strides, and itemsize of F layout?

    Note: The code is usable as a numba-compiled function
    """
def type_can_asarray(arr):
    """ Returns True if the type of 'arr' is supported by the Numba `np.asarray`
    implementation, False otherwise.
    """
def type_is_scalar(typ):
    """ Returns True if the type of 'typ' is a scalar type, according to
    NumPy rules. False otherwise.
    https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
    """
def check_is_integer(v, name) -> None:
    """Raises TypingError if the value is not an integer."""
def lt_floats(a, b): ...
def lt_complex(a, b): ...
