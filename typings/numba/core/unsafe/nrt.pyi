from numba.core import types as types
from numba.core.extending import intrinsic as intrinsic
from numba.core.typing import signature as signature

def NRT_get_api(tyctx):
    """NRT_get_api()

    Calls NRT_get_api() from the NRT C API
    Returns LLVM Type i8* (void pointer)
    """
