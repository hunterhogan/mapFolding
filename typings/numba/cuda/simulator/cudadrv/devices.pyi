from _typeshed import Incomplete
from typing import NamedTuple

class _MemoryInfo(NamedTuple):
    free: Incomplete
    total: Incomplete

_SIMULATOR_CC: Incomplete

class FakeCUDADevice:
    uuid: str
    def __init__(self) -> None: ...
    @property
    def compute_capability(self): ...

class FakeCUDAContext:
    """
    This stub implements functionality only for simulating a single GPU
    at the moment.
    """
    _device_id: Incomplete
    _device: Incomplete
    def __init__(self, device_id) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def __str__(self) -> str: ...
    @property
    def id(self): ...
    @property
    def device(self): ...
    @property
    def compute_capability(self): ...
    def reset(self) -> None: ...
    def get_memory_info(self):
        """
        Cross-platform free / total host memory is hard without external
        dependencies, e.g. `psutil` - so return infinite memory to maintain API
        type compatibility
        """
    def memalloc(self, sz):
        """
        Allocates memory on the simulated device
        At present, there is no division between simulated
        host memory and simulated device memory.
        """
    def memhostalloc(self, sz, mapped: bool = False, portable: bool = False, wc: bool = False):
        """Allocates memory on the host"""

class FakeDeviceList:
    """
    This stub implements a device list containing a single GPU. It also
    keeps track of the GPU status, i.e. whether the context is closed or not,
    which may have been set by the user calling reset()
    """
    lst: Incomplete
    closed: bool
    def __init__(self) -> None: ...
    def __getitem__(self, devnum): ...
    def __str__(self) -> str: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    @property
    def current(self): ...

gpus: Incomplete

def reset() -> None: ...
def get_context(devnum: int = 0): ...
def require_context(func):
    '''
    In the simulator, a context is always "available", so this is a no-op.
    '''
