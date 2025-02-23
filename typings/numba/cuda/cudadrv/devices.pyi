from .driver import USE_NV_BINDING as USE_NV_BINDING, driver as driver
from _typeshed import Incomplete
from collections.abc import Generator

class _DeviceList:
    lst: Incomplete
    def __getattr__(self, attr): ...
    def __getitem__(self, devnum):
        """
        Returns the context manager for device *devnum*.
        """
    def __str__(self) -> str: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    @property
    def current(self):
        """Returns the active device or None if there's no active device
        """

class _DeviceContextManager:
    """
    Provides a context manager for executing in the context of the chosen
    device. The normal use of instances of this type is from
    ``numba.cuda.gpus``. For example, to execute on device 2::

       with numba.cuda.gpus[2]:
           d_a = numba.cuda.to_device(a)

    to copy the array *a* onto device 2, referred to by *d_a*.
    """
    _device: Incomplete
    def __init__(self, device) -> None: ...
    def __getattr__(self, item): ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def __str__(self) -> str: ...

class _Runtime:
    """Emulate the CUDA runtime context management.

    It owns all Devices and Contexts.
    Keeps at most one Context per Device
    """
    gpus: Incomplete
    _tls: Incomplete
    _mainthread: Incomplete
    _lock: Incomplete
    def __init__(self) -> None: ...
    def ensure_context(self) -> Generator[None]:
        """Ensure a CUDA context is available inside the context.

        On entrance, queries the CUDA driver for an active CUDA context and
        attaches it in TLS for subsequent calls so they do not need to query
        the CUDA driver again.  On exit, detach the CUDA context from the TLS.

        This will allow us to pickup thirdparty activated CUDA context in
        any top-level Numba CUDA API.
        """
    def get_or_create_context(self, devnum):
        """Returns the primary context and push+create it if needed
        for *devnum*.  If *devnum* is None, use the active CUDA context (must
        be primary) or create a new one with ``devnum=0``.
        """
    def _get_or_create_context_uncached(self, devnum):
        """See also ``get_or_create_context(devnum)``.
        This version does not read the cache.
        """
    def _activate_context_for(self, devnum): ...
    def _get_attached_context(self): ...
    def _set_attached_context(self, ctx) -> None: ...
    def reset(self) -> None:
        """Clear all contexts in the thread.  Destroy the context if and only
        if we are in the main thread.
        """
    def _destroy_all_contexts(self) -> None: ...

_runtime: Incomplete
gpus: Incomplete

def get_context(devnum: Incomplete | None = None):
    """Get the current device or use a device by device number, and
    return the CUDA context.
    """
def require_context(fn):
    """
    A decorator that ensures a CUDA context is available when *fn* is executed.

    Note: The function *fn* cannot switch CUDA-context.
    """
def reset() -> None:
    """Reset the CUDA subsystem for the current thread.

    In the main thread:
    This removes all CUDA contexts.  Only use this at shutdown or for
    cleaning up between tests.

    In non-main threads:
    This clear the CUDA context stack only.

    """
