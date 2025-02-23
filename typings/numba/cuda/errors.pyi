from _typeshed import Incomplete
from numba.core.errors import LoweringError as LoweringError

class KernelRuntimeError(RuntimeError):
    tid: Incomplete
    ctaid: Incomplete
    msg: Incomplete
    def __init__(self, msg, tid: Incomplete | None = None, ctaid: Incomplete | None = None) -> None: ...

class CudaLoweringError(LoweringError): ...

_launch_help_url: str
missing_launch_config_msg: Incomplete

def normalize_kernel_dimensions(griddim, blockdim):
    """
    Normalize and validate the user-supplied kernel dimensions.
    """
