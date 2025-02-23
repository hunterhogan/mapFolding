from .error import NvvmError as NvvmError, NvvmSupportError as NvvmSupportError, NvvmWarning as NvvmWarning
from .libs import get_libdevice as get_libdevice, open_cudalib as open_cudalib, open_libdevice as open_libdevice
from _typeshed import Incomplete
from ctypes import c_int, c_void_p
from numba.core import cgutils as cgutils, config as config

logger: Incomplete
ADDRSPACE_GENERIC: int
ADDRSPACE_GLOBAL: int
ADDRSPACE_SHARED: int
ADDRSPACE_CONSTANT: int
ADDRSPACE_LOCAL: int
nvvm_program = c_void_p
nvvm_result = c_int
RESULT_CODE_NAMES: Incomplete
_datalayout_original: str
_datalayout_i128: str

def is_available():
    """
    Return if libNVVM is available
    """

_nvvm_lock: Incomplete

class NVVM:
    """Process-wide singleton.
    """
    _PROTOTYPES: Incomplete
    __INSTANCE: Incomplete
    def __new__(cls): ...
    _majorIR: Incomplete
    _minorIR: Incomplete
    _majorDbg: Incomplete
    _minorDbg: Incomplete
    _supported_ccs: Incomplete
    def __init__(self) -> None: ...
    @property
    def data_layout(self): ...
    @property
    def supported_ccs(self): ...
    def get_version(self): ...
    def get_ir_version(self): ...
    def check_error(self, error, msg, exit: bool = False) -> None: ...

class CompilationUnit:
    driver: Incomplete
    _handle: Incomplete
    def __init__(self) -> None: ...
    def __del__(self) -> None: ...
    def add_module(self, buffer) -> None:
        """
         Add a module level NVVM IR to a compilation unit.
         - The buffer should contain an NVVM module IR either in the bitcode
           representation (LLVM3.0) or in the text representation.
        """
    def lazy_add_module(self, buffer) -> None:
        """
        Lazily add an NVVM IR module to a compilation unit.
        The buffer should contain NVVM module IR either in the bitcode
        representation or in the text representation.
        """
    log: Incomplete
    def compile(self, **options):
        '''Perform Compilation.

        Compilation options are accepted as keyword arguments, with the
        following considerations:

        - Underscores (`_`) in option names are converted to dashes (`-`), to
          match NVVM\'s option name format.
        - Options that take a value will be emitted in the form
          "-<name>=<value>".
        - Booleans passed as option values will be converted to integers.
        - Options which take no value (such as `-gen-lto`) should have a value
          of `None` passed in and will be emitted in the form "-<name>".

        For documentation on NVVM compilation options, see the CUDA Toolkit
        Documentation:

        https://docs.nvidia.com/cuda/libnvvm-api/index.html#_CPPv418nvvmCompileProgram11nvvmProgramiPPKc
        '''
    def _try_error(self, err, msg) -> None: ...
    def get_log(self): ...

COMPUTE_CAPABILITIES: Incomplete
CTK_SUPPORTED: Incomplete

def ccs_supported_by_ctk(ctk_version): ...
def get_supported_ccs(): ...
def find_closest_arch(mycc):
    """
    Given a compute capability, return the closest compute capability supported
    by the CUDA toolkit.

    :param mycc: Compute capability as a tuple ``(MAJOR, MINOR)``
    :return: Closest supported CC as a tuple ``(MAJOR, MINOR)``
    """
def get_arch_option(major, minor):
    """Matches with the closest architecture option
    """

MISSING_LIBDEVICE_FILE_MSG: str

class LibDevice:
    _cache_: Incomplete
    bc: Incomplete
    def __init__(self) -> None: ...
    def get(self): ...

cas_nvvm: str
ir_numba_atomic_binary_template: str
ir_numba_atomic_inc_template: str
ir_numba_atomic_dec_template: str
ir_numba_atomic_minmax_template: str

def ir_cas(Ti): ...
def ir_numba_atomic_binary(T, Ti, OP, FUNC): ...
def ir_numba_atomic_minmax(T, Ti, NAN, OP, PTR_OR_VAL, FUNC): ...
def ir_numba_atomic_inc(T, Tu): ...
def ir_numba_atomic_dec(T, Tu): ...
def llvm_replace(llvmir): ...
def compile_ir(llvmir, **opts): ...

re_attributes_def: Incomplete

def llvm140_to_70_ir(ir):
    """
    Convert LLVM 14.0 IR for LLVM 7.0.
    """
def set_cuda_kernel(function) -> None:
    """
    Mark a function as a CUDA kernel. Kernels have the following requirements:

    - Metadata that marks them as a kernel.
    - Addition to the @llvm.used list, so that they will not be discarded.
    - The noinline attribute is not permitted, because this causes NVVM to emit
      a warning, which counts as failing IR verification.

    Presently it is assumed that there is one kernel per module, which holds
    for Numba-jitted functions. If this changes in future or this function is
    to be used externally, this function may need modification to add to the
    @llvm.used list rather than creating it.
    """
def add_ir_version(mod) -> None:
    """Add NVVM IR version to module"""
