from _typeshed import Incomplete
from llvmlite import binding as binding
from numba.core import cgutils as cgutils, config as config, types as types

_word_type: Incomplete
_pointer_type: Incomplete
_meminfo_struct_type: Incomplete
incref_decref_ty: Incomplete
meminfo_data_ty: Incomplete

def _define_nrt_meminfo_data(module) -> None:
    """
    Implement NRT_MemInfo_data_fast in the module.  This allows LLVM
    to inline lookup of the data pointer.
    """
def _define_nrt_incref(module, atomic_incr) -> None:
    """
    Implement NRT_incref in the module
    """
def _define_nrt_decref(module, atomic_decr) -> None:
    """
    Implement NRT_decref in the module
    """

_disable_atomicity: int

def _define_atomic_inc_dec(module, op, ordering):
    '''Define a llvm function for atomic increment/decrement to the given module
    Argument ``op`` is the operation "add"/"sub".  Argument ``ordering`` is
    the memory ordering.  The generated function returns the new value.
    '''
def _define_atomic_cas(module, ordering):
    """Define a llvm function for atomic compare-and-swap.
    The generated function is a direct wrapper of the LLVM cmpxchg with the
    difference that the a int indicate success (1) or failure (0) is returned
    and the last argument is a output pointer for storing the old value.

    Note
    ----
    On failure, the generated function behaves like an atomic load.  The loaded
    value is stored to the last argument.
    """
def _define_nrt_unresolved_abort(ctx, module):
    """
    Defines an abort function due to unresolved symbol.

    The function takes no args and will always raise an exception.
    It should be safe to call this function with incorrect number of arguments.
    """
def create_nrt_module(ctx):
    """
    Create an IR module defining the LLVM NRT functions.
    A (IR module, library) tuple is returned.
    """
def compile_nrt_functions(ctx):
    """
    Compile all LLVM NRT functions and return a library containing them.
    The library is created using the given target context.
    """
