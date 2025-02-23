from _typeshed import Incomplete
from numba import pndindex as pndindex
from numba.core import config as config, types as types, utils as utils
from numba.core.errors import NumbaAssertionError as NumbaAssertionError, NumbaPerformanceWarning as NumbaPerformanceWarning, NumbaTypeError as NumbaTypeError, TypingError as TypingError
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, AttributeTemplate as AttributeTemplate, CallableTemplate as CallableTemplate, Registry as Registry, signature as signature
from numba.np.numpy_support import _ufunc_loop_sig as _ufunc_loop_sig, as_dtype as as_dtype, carray as carray, farray as farray, from_dtype as from_dtype, resolve_output_type as resolve_output_type, supported_ufunc_loop as supported_ufunc_loop, ufunc_find_matching_loop as ufunc_find_matching_loop

registry: Incomplete
infer: Incomplete
infer_global: Incomplete
infer_getattr: Incomplete

class Numpy_rules_ufunc(AbstractTemplate):
    @classmethod
    def _handle_inputs(cls, ufunc, args, kws):
        """
        Process argument types to a given *ufunc*.
        Returns a (base types, explicit outputs, ndims, layout) tuple where:
        - `base types` is a tuple of scalar types for each input
        - `explicit outputs` is a tuple of explicit output types (arrays)
        - `ndims` is the number of dimensions of the loop and also of
          any outputs, explicit or implicit
        - `layout` is the layout for any implicit output to be allocated
        """
    @property
    def ufunc(self): ...
    def generic(self, args, kws): ...

class NumpyRulesArrayOperator(Numpy_rules_ufunc):
    _op_map: Incomplete
    @property
    def ufunc(self): ...
    @classmethod
    def install_operations(cls) -> None: ...
    def generic(self, args, kws):
        """Overloads and calls base class generic() method, returning
        None if a TypingError occurred.

        Returning None for operators is important since operators are
        heavily overloaded, and by suppressing type errors, we allow
        type inference to check other possibilities before giving up
        (particularly user-defined operators).
        """

_binop_map: Incomplete

class NumpyRulesInplaceArrayOperator(NumpyRulesArrayOperator):
    _op_map: Incomplete
    def generic(self, args, kws): ...

class NumpyRulesUnaryArrayOperator(NumpyRulesArrayOperator):
    _op_map: Incomplete
    def generic(self, args, kws): ...

math_operations: Incomplete
trigonometric_functions: Incomplete
bit_twiddling_functions: Incomplete
comparison_functions: Incomplete
floating_functions: Incomplete
logic_functions: Incomplete
_unsupported: Incomplete

def register_numpy_ufunc(name, register_global=...) -> None: ...

all_ufuncs: Incomplete
supported_ufuncs: Incomplete
supported_array_operators: Incomplete

class Numpy_method_redirection(AbstractTemplate):
    """
    A template redirecting a Numpy global function (e.g. np.sum) to an
    array method of the same name (e.g. ndarray.sum).
    """
    prefer_literal: bool
    def generic(self, args, kws): ...

def _numpy_redirect(fname) -> None: ...

np_types: Incomplete

def register_number_classes(register_global) -> None: ...
def parse_shape(shape):
    """
    Given a shape, return the number of dimensions.
    """
def parse_dtype(dtype):
    """
    Return the dtype of a type, if it is either a DtypeSpec (used for most
    dtypes) or a TypeRef (used for record types).
    """
def _parse_nested_sequence(context, typ):
    """
    Parse a (possibly 0d) nested sequence type.
    A (ndim, dtype) tuple is returned.  Note the sequence may still be
    heterogeneous, as long as it converts to the given dtype.
    """
def _infer_dtype_from_inputs(inputs): ...
def _homogeneous_dims(context, func_name, arrays): ...
def _sequence_of_arrays(context, func_name, arrays, dim_chooser=...): ...
def _choose_concatenation_layout(arrays): ...

class MatMulTyperMixin:
    def matmul_typer(self, a, b, out: Incomplete | None = None):
        """
        Typer function for Numpy matrix multiplication.
        """

def _check_linalg_matrix(a, func_name) -> None: ...

class NdEnumerate(AbstractTemplate):
    def generic(self, args, kws): ...

class NdIter(AbstractTemplate):
    def generic(self, args, kws): ...

class NdIndex(AbstractTemplate):
    def generic(self, args, kws): ...

class DtypeEq(AbstractTemplate):
    def generic(self, args, kws): ...
