from _typeshed import Incomplete
from numba.core import config as config
from numba.cuda.cudadrv import enums as enums
from numba.cuda.cudadrv.driver import ERROR_MAP as ERROR_MAP, make_logger as make_logger
from numba.cuda.cudadrv.error import CudaRuntimeError as CudaRuntimeError, CudaSupportError as CudaSupportError
from numba.cuda.cudadrv.libs import open_cudalib as open_cudalib
from numba.cuda.cudadrv.rtapi import API_PROTOTYPES as API_PROTOTYPES

class CudaRuntimeAPIError(CudaRuntimeError):
    """
    Raised when there is an error accessing a C API from the CUDA Runtime.
    """
    code: Incomplete
    msg: Incomplete
    def __init__(self, code, msg) -> None: ...
    def __str__(self) -> str: ...

class Runtime:
    """
    Runtime object that lazily binds runtime API functions.
    """
    is_initialized: bool
    def __init__(self) -> None: ...
    lib: Incomplete
    def _initialize(self) -> None: ...
    def __getattr__(self, fname): ...
    def _wrap_api_call(self, fname, libfn): ...
    def _check_error(self, fname, retcode) -> None: ...
    def _find_api(self, fname): ...
    def get_version(self):
        """
        Returns the CUDA Runtime version as a tuple (major, minor).
        """
    def is_supported_version(self):
        """
        Returns True if the CUDA Runtime is a supported version.
        """
    @property
    def supported_versions(self):
        """A tuple of all supported CUDA toolkit versions. Versions are given in
        the form ``(major_version, minor_version)``."""

runtime: Incomplete

def get_version():
    """
    Return the runtime version as a tuple of (major, minor)
    """
