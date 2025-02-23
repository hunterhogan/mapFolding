from _typeshed import Incomplete
from numba import _helperlib as _helperlib
from numba.core import intrinsics as intrinsics, utils as utils

def _add_missing_symbol(symbol, addr) -> None:
    """Add missing symbol into LLVM internal symtab
    """
def _get_msvcrt_symbol(symbol):
    """
    Under Windows, look up a symbol inside the C runtime
    and return the raw pointer value as an integer.
    """
def compile_multi3(context):
    """
    Compile the multi3() helper function used by LLVM
    for 128-bit multiplication on 32-bit platforms.
    """

class _Installer:
    _installed: bool
    def install(self, context) -> None:
        """
        Install the functions into LLVM.  This only needs to be done once,
        as the mappings are persistent during the process lifetime.
        """

class _ExternalMathFunctions(_Installer):
    """
    Map the math functions from the C runtime library into the LLVM
    execution environment.
    """
    _multi3_lib: Incomplete
    def _do_install(self, context) -> None: ...

c_math_functions: Incomplete
