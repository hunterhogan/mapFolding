from _typeshed import Incomplete
from numba.core.typing.templates import signature as signature
from numba.types import Tuple as Tuple, float32 as float32, float64 as float64, int16 as int16, int32 as int32, int64 as int64, void as void
from typing import NamedTuple

class arg(NamedTuple):
    name: Incomplete
    ty: Incomplete
    is_ptr: Incomplete

functions: Incomplete

def create_signature(retty, args):
    """
    Given the return type and arguments for a libdevice function, return the
    signature of the stub function used to call it from CUDA Python.
    """

docstring_template: str
param_template: str

def generate_stubs(): ...
