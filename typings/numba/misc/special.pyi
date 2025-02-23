from numba.core.typing.asnumbatype import as_numba_type as as_numba_type
from numba.core.typing.typeof import typeof as typeof

__all__ = ['typeof', 'as_numba_type', 'prange', 'pndindex', 'gdb', 'gdb_breakpoint', 'gdb_init', 'literally', 'literal_unroll']

def pndindex(*args):
    """ Provides an n-dimensional parallel iterator that generates index tuples
    for each iteration point. Sequentially, pndindex is identical to np.ndindex.
    """

class prange:
    """ Provides a 1D parallel iterator that generates a sequence of integers.
    In non-parallel contexts, prange is identical to range.
    """
    def __new__(cls, *args): ...

def gdb(*args) -> None:
    """
    Calling this function will invoke gdb and attach it to the current process
    at the call site. Arguments are strings in the gdb command language syntax
    which will be executed by gdb once initialisation has occurred.
    """
def gdb_breakpoint() -> None:
    """
    Calling this function will inject a breakpoint at the call site that is
    recognised by both `gdb` and `gdb_init`, this is to allow breaking at
    multiple points. gdb will stop in the user defined code just after the frame
    employed by the breakpoint returns.
    """
def gdb_init(*args) -> None:
    """
    Calling this function will invoke gdb and attach it to the current process
    at the call site, then continue executing the process under gdb's control.
    Arguments are strings in the gdb command language syntax which will be
    executed by gdb once initialisation has occurred.
    """
def literally(obj):
    """Forces Numba to interpret *obj* as an Literal value.

    *obj* must be either a literal or an argument of the caller function, where
    the argument must be bound to a literal. The literal requirement
    propagates up the call stack.

    This function is intercepted by the compiler to alter the compilation
    behavior to wrap the corresponding function parameters as ``Literal``.
    It has **no effect** outside of nopython-mode (interpreter, and objectmode).

    The current implementation detects literal arguments in two ways:

    1. Scans for uses of ``literally`` via a compiler pass.
    2. ``literally`` is overloaded to raise ``numba.errors.ForceLiteralArg``
       to signal the dispatcher to treat the corresponding parameter
       differently. This mode is to support indirect use (via a function call).

    The execution semantic of this function is equivalent to an identity
    function.

    See :ghfile:`numba/tests/test_literal_dispatch.py` for examples.
    """
def literal_unroll(container): ...
