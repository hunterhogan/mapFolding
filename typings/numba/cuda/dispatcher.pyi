from _typeshed import Incomplete
from numba import _dispatcher as _dispatcher, cuda as cuda
from numba.core import config as config, serialize as serialize, sigutils as sigutils, types as types, typing as typing, utils as utils
from numba.core.caching import Cache as Cache, CacheImpl as CacheImpl
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.core.dispatcher import Dispatcher as Dispatcher
from numba.core.errors import NumbaPerformanceWarning as NumbaPerformanceWarning
from numba.core.typing.typeof import Purpose as Purpose, typeof as typeof
from numba.cuda.api import get_current_device as get_current_device
from numba.cuda.args import wrap_arg as wrap_arg
from numba.cuda.compiler import CUDACompiler as CUDACompiler, compile_cuda as compile_cuda
from numba.cuda.cudadrv import driver as driver
from numba.cuda.cudadrv.devices import get_context as get_context
from numba.cuda.descriptor import cuda_target as cuda_target
from numba.cuda.errors import missing_launch_config_msg as missing_launch_config_msg, normalize_kernel_dimensions as normalize_kernel_dimensions

cuda_fp16_math_funcs: Incomplete

class _Kernel(serialize.ReduceMixin):
    """
    CUDA Kernel specialized for a given set of argument types. When called, this
    object launches the kernel on the device.
    """
    objectmode: bool
    entry_point: Incomplete
    py_func: Incomplete
    argtypes: Incomplete
    debug: Incomplete
    lineinfo: Incomplete
    extensions: Incomplete
    cooperative: Incomplete
    entry_name: Incomplete
    signature: Incomplete
    _type_annotation: Incomplete
    _codelibrary: Incomplete
    call_helper: Incomplete
    target_context: Incomplete
    fndesc: Incomplete
    environment: Incomplete
    _referenced_environments: Incomplete
    lifted: Incomplete
    reload_init: Incomplete
    def __init__(self, py_func, argtypes, link: Incomplete | None = None, debug: bool = False, lineinfo: bool = False, inline: bool = False, fastmath: bool = False, extensions: Incomplete | None = None, max_registers: Incomplete | None = None, opt: bool = True, device: bool = False) -> None: ...
    @property
    def library(self): ...
    @property
    def type_annotation(self): ...
    def _find_referenced_environments(self): ...
    @property
    def codegen(self): ...
    @property
    def argument_types(self): ...
    @classmethod
    def _rebuild(cls, cooperative, name, signature, codelibrary, debug, lineinfo, call_helper, extensions):
        """
        Rebuild an instance.
        """
    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are serialized in PTX form.
        Type annotation are discarded.
        Thread, block and shared memory configuration are serialized.
        Stream information is discarded.
        """
    def bind(self) -> None:
        """
        Force binding to current CUDA context
        """
    @property
    def regs_per_thread(self):
        """
        The number of registers used by each thread for this kernel.
        """
    @property
    def const_mem_size(self):
        """
        The amount of constant memory used by this kernel.
        """
    @property
    def shared_mem_per_block(self):
        """
        The amount of shared memory used per block for this kernel.
        """
    @property
    def max_threads_per_block(self):
        """
        The maximum allowable threads per block.
        """
    @property
    def local_mem_per_thread(self):
        """
        The amount of local memory used per thread for this kernel.
        """
    def inspect_llvm(self):
        """
        Returns the LLVM IR for this kernel.
        """
    def inspect_asm(self, cc):
        """
        Returns the PTX code for this kernel.
        """
    def inspect_sass_cfg(self):
        """
        Returns the CFG of the SASS for this kernel.

        Requires nvdisasm to be available on the PATH.
        """
    def inspect_sass(self):
        """
        Returns the SASS code for this kernel.

        Requires nvdisasm to be available on the PATH.
        """
    def inspect_types(self, file: Incomplete | None = None) -> None:
        """
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        """
    def max_cooperative_grid_blocks(self, blockdim, dynsmemsize: int = 0):
        """
        Calculates the maximum number of blocks that can be launched for this
        kernel in a cooperative grid in the current context, for the given block
        and dynamic shared memory sizes.

        :param blockdim: Block dimensions, either as a scalar for a 1D block, or
                         a tuple for 2D or 3D blocks.
        :param dynsmemsize: Dynamic shared memory size in bytes.
        :return: The maximum number of blocks in the grid.
        """
    def launch(self, args, griddim, blockdim, stream: int = 0, sharedmem: int = 0): ...
    def _prepare_args(self, ty, val, stream, retr, kernelargs) -> None:
        """
        Convert arguments to ctypes and append to kernelargs
        """

class ForAll:
    dispatcher: Incomplete
    ntasks: Incomplete
    thread_per_block: Incomplete
    stream: Incomplete
    sharedmem: Incomplete
    def __init__(self, dispatcher, ntasks, tpb, stream, sharedmem) -> None: ...
    def __call__(self, *args): ...
    def _compute_thread_per_block(self, dispatcher): ...

class _LaunchConfiguration:
    dispatcher: Incomplete
    griddim: Incomplete
    blockdim: Incomplete
    stream: Incomplete
    sharedmem: Incomplete
    def __init__(self, dispatcher, griddim, blockdim, stream, sharedmem) -> None: ...
    def __call__(self, *args): ...

class CUDACacheImpl(CacheImpl):
    def reduce(self, kernel): ...
    def rebuild(self, target_context, payload): ...
    def check_cachable(self, cres): ...

class CUDACache(Cache):
    """
    Implements a cache that saves and loads CUDA kernels and compile results.
    """
    _impl_class = CUDACacheImpl
    def load_overload(self, sig, target_context): ...

class CUDADispatcher(Dispatcher, serialize.ReduceMixin):
    """
    CUDA Dispatcher object. When configured and called, the dispatcher will
    specialize itself for the given arguments (if no suitable specialized
    version already exists) & compute capability, and launch on the device
    associated with the current context.

    Dispatcher objects are not to be constructed by the user, but instead are
    created using the :func:`numba.cuda.jit` decorator.
    """
    _fold_args: bool
    targetdescr = cuda_target
    _specialized: bool
    specializations: Incomplete
    def __init__(self, py_func, targetoptions, pipeline_class=...) -> None: ...
    @property
    def _numba_type_(self): ...
    _cache: Incomplete
    def enable_caching(self) -> None: ...
    def configure(self, griddim, blockdim, stream: int = 0, sharedmem: int = 0): ...
    def __getitem__(self, args): ...
    def forall(self, ntasks, tpb: int = 0, stream: int = 0, sharedmem: int = 0):
        """Returns a 1D-configured dispatcher for a given number of tasks.

        This assumes that:

        - the kernel maps the Global Thread ID ``cuda.grid(1)`` to tasks on a
          1-1 basis.
        - the kernel checks that the Global Thread ID is upper-bounded by
          ``ntasks``, and does nothing if it is not.

        :param ntasks: The number of tasks.
        :param tpb: The size of a block. An appropriate value is chosen if this
                    parameter is not supplied.
        :param stream: The stream on which the configured dispatcher will be
                       launched.
        :param sharedmem: The number of bytes of dynamic shared memory required
                          by the kernel.
        :return: A configured dispatcher, ready to launch on a set of
                 arguments."""
    @property
    def extensions(self):
        """
        A list of objects that must have a `prepare_args` function. When a
        specialized kernel is called, each argument will be passed through
        to the `prepare_args` (from the last object in this list to the
        first). The arguments to `prepare_args` are:

        - `ty` the numba type of the argument
        - `val` the argument value itself
        - `stream` the CUDA stream used for the current call to the kernel
        - `retr` a list of zero-arg functions that you may want to append
          post-call cleanup work to.

        The `prepare_args` function must return a tuple `(ty, val)`, which
        will be passed in turn to the next right-most `extension`. After all
        the extensions have been called, the resulting `(ty, val)` will be
        passed into Numba's default argument marshalling logic.
        """
    def __call__(self, *args, **kwargs) -> None: ...
    def call(self, args, griddim, blockdim, stream, sharedmem) -> None:
        """
        Compile if necessary and invoke this kernel with *args*.
        """
    def _compile_for_args(self, *args, **kws): ...
    def typeof_pyval(self, val): ...
    def specialize(self, *args):
        """
        Create a new instance of this dispatcher specialized for the given
        *args*.
        """
    @property
    def specialized(self):
        """
        True if the Dispatcher has been specialized.
        """
    def get_regs_per_thread(self, signature: Incomplete | None = None):
        """
        Returns the number of registers used by each thread in this kernel for
        the device in the current context.

        :param signature: The signature of the compiled kernel to get register
                          usage for. This may be omitted for a specialized
                          kernel.
        :return: The number of registers used by the compiled variant of the
                 kernel for the given signature and current device.
        """
    def get_const_mem_size(self, signature: Incomplete | None = None):
        """
        Returns the size in bytes of constant memory used by this kernel for
        the device in the current context.

        :param signature: The signature of the compiled kernel to get constant
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The size in bytes of constant memory allocated by the
                 compiled variant of the kernel for the given signature and
                 current device.
        """
    def get_shared_mem_per_block(self, signature: Incomplete | None = None):
        """
        Returns the size in bytes of statically allocated shared memory
        for this kernel.

        :param signature: The signature of the compiled kernel to get shared
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The amount of shared memory allocated by the compiled variant
                 of the kernel for the given signature and current device.
        """
    def get_max_threads_per_block(self, signature: Incomplete | None = None):
        """
        Returns the maximum allowable number of threads per block
        for this kernel. Exceeding this threshold will result in
        the kernel failing to launch.

        :param signature: The signature of the compiled kernel to get the max
                          threads per block for. This may be omitted for a
                          specialized kernel.
        :return: The maximum allowable threads per block for the compiled
                 variant of the kernel for the given signature and current
                 device.
        """
    def get_local_mem_per_thread(self, signature: Incomplete | None = None):
        """
        Returns the size in bytes of local memory per thread
        for this kernel.

        :param signature: The signature of the compiled kernel to get local
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The amount of local memory allocated by the compiled variant
                 of the kernel for the given signature and current device.
        """
    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows resolution of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
    def compile_device(self, args, return_type: Incomplete | None = None):
        """Compile the device function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.

        Returns the `CompileResult`.
        """
    def add_overload(self, kernel, argtypes) -> None: ...
    def compile(self, sig):
        """
        Compile and bind to the current context a version of this kernel
        specialized for the given signature.
        """
    def inspect_llvm(self, signature: Incomplete | None = None):
        """
        Return the LLVM IR for this kernel.

        :param signature: A tuple of argument types.
        :return: The LLVM IR for the given signature, or a dict of LLVM IR
                 for all previously-encountered signatures.

        """
    def inspect_asm(self, signature: Incomplete | None = None):
        """
        Return this kernel's PTX assembly code for for the device in the
        current context.

        :param signature: A tuple of argument types.
        :return: The PTX code for the given signature, or a dict of PTX codes
                 for all previously-encountered signatures.
        """
    def inspect_sass_cfg(self, signature: Incomplete | None = None):
        """
        Return this kernel's CFG for the device in the current context.

        :param signature: A tuple of argument types.
        :return: The CFG for the given signature, or a dict of CFGs
                 for all previously-encountered signatures.

        The CFG for the device in the current context is returned.

        Requires nvdisasm to be available on the PATH.
        """
    def inspect_sass(self, signature: Incomplete | None = None):
        """
        Return this kernel's SASS assembly code for for the device in the
        current context.

        :param signature: A tuple of argument types.
        :return: The SASS code for the given signature, or a dict of SASS codes
                 for all previously-encountered signatures.

        SASS for the device in the current context is returned.

        Requires nvdisasm to be available on the PATH.
        """
    def inspect_types(self, file: Incomplete | None = None) -> None:
        """
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        """
    @classmethod
    def _rebuild(cls, py_func, targetoptions):
        """
        Rebuild an instance.
        """
    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are discarded.
        """
