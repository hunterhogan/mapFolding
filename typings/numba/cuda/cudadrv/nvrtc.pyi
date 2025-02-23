from _typeshed import Incomplete
from ctypes import c_int, c_void_p
from enum import IntEnum
from numba.core import config as config
from numba.cuda.cudadrv.error import NvrtcCompilationError as NvrtcCompilationError, NvrtcError as NvrtcError, NvrtcSupportError as NvrtcSupportError

nvrtc_program = c_void_p
nvrtc_result = c_int

class NvrtcResult(IntEnum):
    NVRTC_SUCCESS = 0
    NVRTC_ERROR_OUT_OF_MEMORY = 1
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
    NVRTC_ERROR_INVALID_INPUT = 3
    NVRTC_ERROR_INVALID_PROGRAM = 4
    NVRTC_ERROR_INVALID_OPTION = 5
    NVRTC_ERROR_COMPILATION = 6
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
    NVRTC_ERROR_INTERNAL_ERROR = 11

_nvrtc_lock: Incomplete

class NvrtcProgram:
    """
    A class for managing the lifetime of nvrtcProgram instances. Instances of
    the class own an nvrtcProgram; when an instance is deleted, the underlying
    nvrtcProgram is destroyed using the appropriate NVRTC API.
    """
    _nvrtc: Incomplete
    _handle: Incomplete
    def __init__(self, nvrtc, handle) -> None: ...
    @property
    def handle(self): ...
    def __del__(self) -> None: ...

class NVRTC:
    """
    Provides a Pythonic interface to the NVRTC APIs, abstracting away the C API
    calls.

    The sole instance of this class is a process-wide singleton, similar to the
    NVVM interface. Initialization is protected by a lock and uses the standard
    (for Numba) open_cudalib function to load the NVRTC library.
    """
    _PROTOTYPES: Incomplete
    __INSTANCE: Incomplete
    def __new__(cls): ...
    def get_version(self):
        """
        Get the NVRTC version as a tuple (major, minor).
        """
    def create_program(self, src, name):
        """
        Create an NVRTC program with managed lifetime.
        """
    def compile_program(self, program, options):
        """
        Compile an NVRTC program. Compilation may fail due to a user error in
        the source; this function returns ``True`` if there is a compilation
        error and ``False`` on success.
        """
    def destroy_program(self, program) -> None:
        """
        Destroy an NVRTC program.
        """
    def get_compile_log(self, program):
        """
        Get the compile log as a Python string.
        """
    def get_ptx(self, program):
        """
        Get the compiled PTX as a Python string.
        """

def compile(src, name, cc):
    """
    Compile a CUDA C/C++ source to PTX for a given compute capability.

    :param src: The source code to compile
    :type src: str
    :param name: The filename of the source (for information only)
    :type name: str
    :param cc: A tuple ``(major, minor)`` of the compute capability
    :type cc: tuple
    :return: The compiled PTX and compilation log
    :rtype: tuple
    """
