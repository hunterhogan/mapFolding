import threading
from ..args import ArgHint as ArgHint, wrap_arg as wrap_arg
from ..errors import normalize_kernel_dimensions as normalize_kernel_dimensions
from .cudadrv.devicearray import FakeCUDAArray as FakeCUDAArray, FakeWithinKernelCUDAArray as FakeWithinKernelCUDAArray
from .kernelapi import Dim3 as Dim3, FakeCUDAModule as FakeCUDAModule, swapped_cuda_module as swapped_cuda_module
from _typeshed import Incomplete
from collections.abc import Generator

_kernel_context: Incomplete

def _push_kernel_context(mod) -> Generator[None]:
    """
    Push the current kernel context.
    """
def _get_kernel_context():
    """
    Get the current kernel context. This is usually done by a device function.
    """

class FakeOverload:
    """
    Used only to provide the max_cooperative_grid_blocks method
    """
    def max_cooperative_grid_blocks(self, blockdim): ...

class FakeOverloadDict(dict):
    def __getitem__(self, key): ...

class FakeCUDAKernel:
    """
    Wraps a @cuda.jit-ed function.
    """
    fn: Incomplete
    _device: Incomplete
    _fastmath: Incomplete
    _debug: Incomplete
    extensions: Incomplete
    grid_dim: Incomplete
    block_dim: Incomplete
    stream: int
    dynshared_size: int
    def __init__(self, fn, device, fastmath: bool = False, extensions=[], debug: bool = False) -> None: ...
    def __call__(self, *args): ...
    def __getitem__(self, configuration): ...
    def bind(self) -> None: ...
    def specialize(self, *args): ...
    def forall(self, ntasks, tpb: int = 0, stream: int = 0, sharedmem: int = 0): ...
    @property
    def overloads(self): ...
    @property
    def py_func(self): ...

class BlockThread(threading.Thread):
    """
    Manages the execution of a function for a single CUDA thread.
    """
    syncthreads_event: Incomplete
    syncthreads_blocked: bool
    _manager: Incomplete
    blockIdx: Incomplete
    threadIdx: Incomplete
    exception: Incomplete
    daemon: bool
    abort: bool
    debug: Incomplete
    thread_id: Incomplete
    def __init__(self, f, manager, blockIdx, threadIdx, debug) -> None: ...
    def run(self) -> None: ...
    def syncthreads(self) -> None: ...
    def syncthreads_count(self, value): ...
    def syncthreads_and(self, value): ...
    def syncthreads_or(self, value): ...
    def __str__(self) -> str: ...

class BlockManager:
    """
    Manages the execution of a thread block.

    When run() is called, all threads are started. Each thread executes until it
    hits syncthreads(), at which point it sets its own syncthreads_blocked to
    True so that the BlockManager knows it is blocked. It then waits on its
    syncthreads_event.

    The BlockManager polls threads to determine if they are blocked in
    syncthreads(). If it finds a blocked thread, it adds it to the set of
    blocked threads. When all threads are blocked, it unblocks all the threads.
    The thread are unblocked by setting their syncthreads_blocked back to False
    and setting their syncthreads_event.

    The polling continues until no threads are alive, when execution is
    complete.
    """
    _grid_dim: Incomplete
    _block_dim: Incomplete
    _f: Incomplete
    _debug: Incomplete
    block_state: Incomplete
    def __init__(self, f, grid_dim, block_dim, debug) -> None: ...
    def run(self, grid_point, *args) -> None: ...
