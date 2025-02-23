from _typeshed import Incomplete
from numba.core import config as config, sigutils as sigutils, types as types
from numba.core.errors import DeprecationError as DeprecationError, NumbaInvalidConfigWarning as NumbaInvalidConfigWarning
from numba.cuda.compiler import declare_device_function as declare_device_function
from numba.cuda.dispatcher import CUDADispatcher as CUDADispatcher
from numba.cuda.simulator.kernel import FakeCUDAKernel as FakeCUDAKernel

_msg_deprecated_signature_arg: str

def jit(func_or_sig: Incomplete | None = None, device: bool = False, inline: bool = False, link=[], debug: Incomplete | None = None, opt: bool = True, lineinfo: bool = False, cache: bool = False, **kws):
    """
    JIT compile a Python function for CUDA GPUs.

    :param func_or_sig: A function to JIT compile, or *signatures* of a
       function to compile. If a function is supplied, then a
       :class:`Dispatcher <numba.cuda.dispatcher.CUDADispatcher>` is returned.
       Otherwise, ``func_or_sig`` may be a signature or a list of signatures,
       and a function is returned. The returned function accepts another
       function, which it will compile and then return a :class:`Dispatcher
       <numba.cuda.dispatcher.CUDADispatcher>`. See :ref:`jit-decorator` for
       more information about passing signatures.

       .. note:: A kernel cannot have any return value.
    :param device: Indicates whether this is a device function.
    :type device: bool
    :param link: A list of files containing PTX or CUDA C/C++ source to link
       with the function
    :type link: list
    :param debug: If True, check for exceptions thrown when executing the
       kernel. Since this degrades performance, this should only be used for
       debugging purposes. If set to True, then ``opt`` should be set to False.
       Defaults to False.  (The default value can be overridden by setting
       environment variable ``NUMBA_CUDA_DEBUGINFO=1``.)
    :param fastmath: When True, enables fastmath optimizations as outlined in
       the :ref:`CUDA Fast Math documentation <cuda-fast-math>`.
    :param max_registers: Request that the kernel is limited to using at most
       this number of registers per thread. The limit may not be respected if
       the ABI requires a greater number of registers than that requested.
       Useful for increasing occupancy.
    :param opt: Whether to compile from LLVM IR to PTX with optimization
                enabled. When ``True``, ``-opt=3`` is passed to NVVM. When
                ``False``, ``-opt=0`` is passed to NVVM. Defaults to ``True``.
    :type opt: bool
    :param lineinfo: If True, generate a line mapping between source code and
       assembly code. This enables inspection of the source code in NVIDIA
       profiling tools and correlation with program counter sampling.
    :type lineinfo: bool
    :param cache: If True, enables the file-based cache for this function.
    :type cache: bool
    """
def declare_device(name, sig):
    """
    Declare the signature of a foreign function. Returns a descriptor that can
    be used to call the function from a Python kernel.

    :param name: The name of the foreign function.
    :type name: str
    :param sig: The Numba signature of the function.
    """
