from .cudadrv import devices as devices, driver as driver, nvvm as nvvm, runtime as runtime
from _typeshed import Incomplete
from numba.core import config as config, serialize as serialize
from numba.core.codegen import CodeLibrary as CodeLibrary, Codegen as Codegen
from numba.cuda.cudadrv.libs import get_cudalib as get_cudalib

CUDA_TRIPLE: str

def run_nvdisasm(cubin, flags): ...
def disassemble_cubin(cubin): ...
def disassemble_cubin_for_cfg(cubin): ...

class CUDACodeLibrary(serialize.ReduceMixin, CodeLibrary):
    """
    The CUDACodeLibrary generates PTX, SASS, cubins for multiple different
    compute capabilities. It also loads cubins to multiple devices (via
    get_cufunc), which may be of different compute capabilities.
    """
    _module: Incomplete
    _linking_libraries: Incomplete
    _linking_files: Incomplete
    needs_cudadevrt: bool
    _llvm_strs: Incomplete
    _ptx_cache: Incomplete
    _ltoir_cache: Incomplete
    _cubin_cache: Incomplete
    _linkerinfo_cache: Incomplete
    _cufunc_cache: Incomplete
    _max_registers: Incomplete
    _nvvm_options: Incomplete
    _entry_name: Incomplete
    def __init__(self, codegen, name, entry_name: Incomplete | None = None, max_registers: Incomplete | None = None, nvvm_options: Incomplete | None = None) -> None:
        """
        codegen:
            Codegen object.
        name:
            Name of the function in the source.
        entry_name:
            Name of the kernel function in the binary, if this is a global
            kernel and not a device function.
        max_registers:
            The maximum register usage to aim for when linking.
        nvvm_options:
                Dict of options to pass to NVVM.
        """
    @property
    def llvm_strs(self): ...
    def get_llvm_str(self): ...
    def _ensure_cc(self, cc): ...
    def get_asm_str(self, cc: Incomplete | None = None): ...
    def get_ltoir(self, cc: Incomplete | None = None): ...
    def get_cubin(self, cc: Incomplete | None = None): ...
    def get_cufunc(self): ...
    def get_linkerinfo(self, cc): ...
    def get_sass(self, cc: Incomplete | None = None): ...
    def get_sass_cfg(self, cc: Incomplete | None = None): ...
    def add_ir_module(self, mod) -> None: ...
    def add_linking_library(self, library) -> None: ...
    def add_linking_file(self, filepath) -> None: ...
    def get_function(self, name): ...
    @property
    def modules(self): ...
    @property
    def linking_libraries(self): ...
    _finalized: bool
    def finalize(self) -> None: ...
    def _reduce_states(self):
        """
        Reduce the instance for serialization. We retain the PTX and cubins,
        but loaded functions are discarded. They are recreated when needed
        after deserialization.
        """
    @classmethod
    def _rebuild(cls, codegen, name, entry_name, llvm_strs, ptx_cache, cubin_cache, linkerinfo_cache, max_registers, nvvm_options, needs_cudadevrt):
        """
        Rebuild an instance.
        """

class JITCUDACodegen(Codegen):
    """
    This codegen implementation for CUDA only generates optimized LLVM IR.
    Generation of PTX code is done separately (see numba.cuda.compiler).
    """
    _library_class = CUDACodeLibrary
    def __init__(self, module_name) -> None: ...
    def _create_empty_module(self, name): ...
    def _add_module(self, module) -> None: ...
    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
