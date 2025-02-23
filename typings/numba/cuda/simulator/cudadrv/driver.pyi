from _typeshed import Incomplete

def device_memset(dst, val, size, stream: int = 0) -> None: ...
def host_to_device(dst, src, size, stream: int = 0) -> None: ...
def device_to_host(dst, src, size, stream: int = 0) -> None: ...
def device_memory_size(obj): ...
def device_to_device(dst, src, size, stream: int = 0) -> None: ...

class FakeDriver:
    def get_device_count(self): ...

driver: Incomplete

class Linker:
    @classmethod
    def new(cls, max_registers: int = 0, lineinfo: bool = False, cc: Incomplete | None = None): ...
    @property
    def lto(self): ...

class LinkerError(RuntimeError): ...
class NvrtcError(RuntimeError): ...
class CudaAPIError(RuntimeError): ...

def launch_kernel(*args, **kwargs) -> None: ...

USE_NV_BINDING: bool
