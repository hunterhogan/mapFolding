from _typeshed import Incomplete
from numba.core import types as types
from numba.core.errors import TypingError as TypingError
from numba.core.typing import templates as templates
from numba.np import numpy_support as numpy_support

ffi: Incomplete
SUPPORTED: Incomplete
_ool_func_types: Incomplete
_ool_func_ptr: Incomplete
_ffi_instances: Incomplete

def is_ffi_instance(obj): ...
def is_cffi_func(obj):
    """Check whether the obj is a CFFI function"""
def get_pointer(cffi_func):
    """
    Get a pointer to the underlying function for a CFFI function as an
    integer.
    """

_cached_type_map: Incomplete

def _type_map():
    """
    Lazily compute type map, as calling ffi.typeof() involves costly
    parsing of C code...
    """
def map_type(cffi_type, use_record_dtype: bool = False):
    """
    Map CFFI type to numba type.

    Parameters
    ----------
    cffi_type:
        The CFFI type to be converted.
    use_record_dtype: bool (default: False)
        When True, struct types are mapped to a NumPy Record dtype.

    """
def map_struct_to_record_dtype(cffi_type):
    """Convert a cffi type into a NumPy Record dtype
    """
def make_function_type(cffi_func, use_record_dtype: bool = False):
    """
    Return a Numba type for the given CFFI function pointer.
    """

registry: Incomplete

class FFI_from_buffer(templates.AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class FFIAttribute(templates.AttributeTemplate):
    key: Incomplete
    def resolve_from_buffer(self, ffi): ...

def register_module(mod) -> None:
    """
    Add typing for all functions in an out-of-line CFFI module to the typemap
    """
def register_type(cffi_type, numba_type) -> None:
    """
    Add typing for a given CFFI type to the typemap
    """
