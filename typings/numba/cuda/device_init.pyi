from .api import *
from .api import _auto_device as _auto_device
from .args import In as In, InOut as InOut, Out as Out
from .cudadrv import nvvm as nvvm
from .cudadrv.error import CudaSupportError as CudaSupportError
from .decorators import declare_device as declare_device, jit as jit
from .errors import KernelRuntimeError as KernelRuntimeError
from .intrinsic_wrapper import all_sync as all_sync, any_sync as any_sync, ballot_sync as ballot_sync, eq_sync as eq_sync, shfl_down_sync as shfl_down_sync, shfl_sync as shfl_sync, shfl_up_sync as shfl_up_sync, shfl_xor_sync as shfl_xor_sync
from .intrinsics import grid as grid, gridsize as gridsize, syncthreads as syncthreads, syncthreads_and as syncthreads_and, syncthreads_count as syncthreads_count, syncthreads_or as syncthreads_or
from .kernels import reduction as reduction
from .stubs import _vector_type_stubs as _vector_type_stubs, activemask as activemask, atomic as atomic, blockDim as blockDim, blockIdx as blockIdx, brev as brev, cbrt as cbrt, clz as clz, const as const, ffs as ffs, fma as fma, fp16 as fp16, gridDim as gridDim, laneid as laneid, lanemask_lt as lanemask_lt, local as local, match_all_sync as match_all_sync, match_any_sync as match_any_sync, nanosleep as nanosleep, popc as popc, selp as selp, shared as shared, shfl_sync_intrinsic as shfl_sync_intrinsic, syncwarp as syncwarp, threadIdx as threadIdx, threadfence as threadfence, threadfence_block as threadfence_block, threadfence_system as threadfence_system, vote_sync_intrinsic as vote_sync_intrinsic, warpsize as warpsize
from _typeshed import Incomplete
from numba.cuda import cg as cg, initialize as initialize
from numba.cuda.cudadrv.driver import BaseCUDAMemoryManager as BaseCUDAMemoryManager, GetIpcHandleMixin as GetIpcHandleMixin, HostOnlyCUDAMemoryManager as HostOnlyCUDAMemoryManager, IpcHandle as IpcHandle, MappedMemory as MappedMemory, MemoryInfo as MemoryInfo, MemoryPointer as MemoryPointer, PinnedMemory as PinnedMemory, set_memory_manager as set_memory_manager
from numba.cuda.cudadrv.runtime import runtime as runtime

reduce: Incomplete
Reduce: Incomplete

def is_available():
    """Returns a boolean to indicate the availability of a CUDA GPU.

    This will initialize the driver if it hasn't been initialized.
    """
def is_supported_version():
    """Returns True if the CUDA Runtime is a supported version.

    Unsupported versions (e.g. newer versions than those known to Numba)
    may still work; this function provides a facility to check whether the
    current Numba version is tested and known to work with the current
    runtime version. If the current version is unsupported, the caller can
    decide how to act. Options include:

    - Continuing silently,
    - Emitting a warning,
    - Generating an error or otherwise preventing the use of CUDA.
    """
def cuda_error():
    """Returns None if there was no error initializing the CUDA driver.
    If there was an error initializing the driver, a string describing the
    error is returned.
    """
