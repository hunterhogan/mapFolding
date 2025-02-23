import abc
import asyncio
from .drvapi import API_PROTOTYPES as API_PROTOTYPES, cu_occupancy_b2d_size as cu_occupancy_b2d_size, cu_stream_callback_pyobj as cu_stream_callback_pyobj, cu_uuid as cu_uuid
from .error import CudaDriverError as CudaDriverError, CudaSupportError as CudaSupportError
from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from numba import mviewbuf as mviewbuf
from numba.core import config as config, serialize as serialize, utils as utils
from numba.cuda.cudadrv import _extras as _extras, drvapi as drvapi, enums as enums, nvrtc as nvrtc
from typing import NamedTuple

USE_NV_BINDING: Incomplete
CU_STREAM_DEFAULT: int
MIN_REQUIRED_CC: Incomplete
SUPPORTS_IPC: Incomplete
_py_decref: Incomplete
_py_incref: Incomplete

def make_logger(): ...

class DeadMemoryError(RuntimeError): ...
class LinkerError(RuntimeError): ...

class CudaAPIError(CudaDriverError):
    code: Incomplete
    msg: Incomplete
    def __init__(self, code, msg) -> None: ...
    def __str__(self) -> str: ...

def locate_driver_and_loader(): ...
def load_driver(dlloader, candidates): ...
def find_driver(): ...

DRIVER_NOT_FOUND_MSG: str
DRIVER_LOAD_ERROR_MSG: str

def _raise_driver_not_found() -> None: ...
def _raise_driver_error(e) -> None: ...
def _build_reverse_error_map(): ...
def _getpid(): ...

ERROR_MAP: Incomplete

class Driver:
    """
    Driver API functions are lazily bound.
    """
    _singleton: Incomplete
    def __new__(cls): ...
    devices: Incomplete
    is_initialized: bool
    initialization_error: Incomplete
    pid: Incomplete
    lib: Incomplete
    def __init__(self) -> None: ...
    def ensure_initialized(self) -> None: ...
    cuIpcOpenMemHandle: Incomplete
    def _initialize_extras(self) -> None: ...
    @property
    def is_available(self): ...
    def __getattr__(self, fname): ...
    def _ctypes_wrap_fn(self, fname, libfn: Incomplete | None = None): ...
    def _cuda_python_wrap_fn(self, fname): ...
    def _find_api(self, fname): ...
    def _detect_fork(self) -> None: ...
    def _check_ctypes_error(self, fname, retcode) -> None: ...
    def _check_cuda_python_error(self, fname, returned): ...
    def get_device(self, devnum: int = 0): ...
    def get_device_count(self): ...
    def list_devices(self):
        """Returns a list of active devices
        """
    def reset(self) -> None:
        """Reset all devices
        """
    def pop_active_context(self):
        """Pop the active CUDA context and return the handle.
        If no CUDA context is active, return None.
        """
    def get_active_context(self):
        """Returns an instance of ``_ActiveContext``.
        """
    def get_version(self):
        """
        Returns the CUDA Runtime version as a tuple (major, minor).
        """

class _ActiveContext:
    """An contextmanager object to cache active context to reduce dependency
    on querying the CUDA driver API.

    Once entering the context, it is assumed that the active CUDA context is
    not changed until the context is exited.
    """
    _tls_cache: Incomplete
    _is_top: Incomplete
    context_handle: Incomplete
    devnum: Incomplete
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def __bool__(self) -> bool:
        """Returns True is there's a valid and active CUDA context.
        """
    __nonzero__ = __bool__

driver: Incomplete

def _build_reverse_device_attrs(): ...

DEVICE_ATTRIBUTES: Incomplete

class Device:
    """
    The device object owns the CUDA contexts.  This is owned by the driver
    object.  User should not construct devices directly.
    """
    @classmethod
    def from_identity(self, identity):
        """Create Device object from device identity created by
        ``Device.get_device_identity()``.
        """
    id: Incomplete
    attributes: Incomplete
    compute_capability: Incomplete
    name: Incomplete
    uuid: Incomplete
    primary_context: Incomplete
    def __init__(self, devnum) -> None: ...
    def get_device_identity(self): ...
    def __repr__(self) -> str: ...
    def __getattr__(self, attr):
        """Read attributes lazily
        """
    def __hash__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def get_primary_context(self):
        """
        Returns the primary context for the device.
        Note: it is not pushed to the CPU thread.
        """
    def release_primary_context(self) -> None:
        """
        Release reference to primary context if it has been retained.
        """
    def reset(self) -> None: ...
    @property
    def supports_float16(self): ...

def met_requirement_for_device(device) -> None: ...

class BaseCUDAMemoryManager(metaclass=ABCMeta):
    """Abstract base class for External Memory Management (EMM) Plugins."""
    context: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @abstractmethod
    def memalloc(self, size):
        """
        Allocate on-device memory in the current context.

        :param size: Size of allocation in bytes
        :type size: int
        :return: A memory pointer instance that owns the allocated memory
        :rtype: :class:`MemoryPointer`
        """
    @abstractmethod
    def memhostalloc(self, size, mapped, portable, wc):
        """
        Allocate pinned host memory.

        :param size: Size of the allocation in bytes
        :type size: int
        :param mapped: Whether the allocated memory should be mapped into the
                       CUDA address space.
        :type mapped: bool
        :param portable: Whether the memory will be considered pinned by all
                         contexts, and not just the calling context.
        :type portable: bool
        :param wc: Whether to allocate the memory as write-combined.
        :type wc: bool
        :return: A memory pointer instance that owns the allocated memory. The
                 return type depends on whether the region was mapped into
                 device memory.
        :rtype: :class:`MappedMemory` or :class:`PinnedMemory`
        """
    @abstractmethod
    def mempin(self, owner, pointer, size, mapped):
        """
        Pin a region of host memory that is already allocated.

        :param owner: The object that owns the memory.
        :param pointer: The pointer to the beginning of the region to pin.
        :type pointer: int
        :param size: The size of the region in bytes.
        :type size: int
        :param mapped: Whether the region should also be mapped into device
                       memory.
        :type mapped: bool
        :return: A memory pointer instance that refers to the allocated
                 memory.
        :rtype: :class:`MappedMemory` or :class:`PinnedMemory`
        """
    @abstractmethod
    def initialize(self):
        """
        Perform any initialization required for the EMM plugin instance to be
        ready to use.

        :return: None
        """
    @abstractmethod
    def get_ipc_handle(self, memory):
        """
        Return an IPC handle from a GPU allocation.

        :param memory: Memory for which the IPC handle should be created.
        :type memory: :class:`MemoryPointer`
        :return: IPC handle for the allocation
        :rtype: :class:`IpcHandle`
        """
    @abstractmethod
    def get_memory_info(self):
        """
        Returns ``(free, total)`` memory in bytes in the context. May raise
        :class:`NotImplementedError`, if returning such information is not
        practical (e.g. for a pool allocator).

        :return: Memory info
        :rtype: :class:`MemoryInfo`
        """
    @abstractmethod
    def reset(self):
        """
        Clears up all memory allocated in this context.

        :return: None
        """
    @abstractmethod
    def defer_cleanup(self):
        """
        Returns a context manager that ensures the implementation of deferred
        cleanup whilst it is active.

        :return: Context manager
        """
    @property
    @abstractmethod
    def interface_version(self):
        """
        Returns an integer specifying the version of the EMM Plugin interface
        supported by the plugin implementation. Should always return 1 for
        implementations of this version of the specification.
        """

class HostOnlyCUDAMemoryManager(BaseCUDAMemoryManager, metaclass=abc.ABCMeta):
    """Base class for External Memory Management (EMM) Plugins that only
    implement on-device allocation. A subclass need not implement the
    ``memhostalloc`` and ``mempin`` methods.

    This class also implements ``reset`` and ``defer_cleanup`` (see
    :class:`numba.cuda.BaseCUDAMemoryManager`) for its own internal state
    management. If an EMM Plugin based on this class also implements these
    methods, then its implementations of these must also call the method from
    ``super()`` to give ``HostOnlyCUDAMemoryManager`` an opportunity to do the
    necessary work for the host allocations it is managing.

    This class does not implement ``interface_version``, as it will always be
    consistent with the version of Numba in which it is implemented. An EMM
    Plugin subclassing this class should implement ``interface_version``
    instead.
    """
    allocations: Incomplete
    deallocations: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _attempt_allocation(self, allocator):
        """
        Attempt allocation by calling *allocator*.  If an out-of-memory error
        is raised, the pending deallocations are flushed and the allocation
        is retried.  If it fails in the second attempt, the error is reraised.
        """
    def memhostalloc(self, size, mapped: bool = False, portable: bool = False, wc: bool = False):
        """Implements the allocation of pinned host memory.

        It is recommended that this method is not overridden by EMM Plugin
        implementations - instead, use the :class:`BaseCUDAMemoryManager`.
        """
    def mempin(self, owner, pointer, size, mapped: bool = False):
        """Implements the pinning of host memory.

        It is recommended that this method is not overridden by EMM Plugin
        implementations - instead, use the :class:`BaseCUDAMemoryManager`.
        """
    def memallocmanaged(self, size, attach_global): ...
    def reset(self) -> None:
        """Clears up all host memory (mapped and/or pinned) in the current
        context.

        EMM Plugins that override this method must call ``super().reset()`` to
        ensure that host allocations are also cleaned up."""
    def defer_cleanup(self) -> Generator[None]:
        """Returns a context manager that disables cleanup of mapped or pinned
        host memory in the current context whilst it is active.

        EMM Plugins that override this method must obtain the context manager
        from this method before yielding to ensure that cleanup of host
        allocations is also deferred."""

class GetIpcHandleMixin:
    """A class that provides a default implementation of ``get_ipc_handle()``.
    """
    def get_ipc_handle(self, memory):
        """Open an IPC memory handle by using ``cuMemGetAddressRange`` to
        determine the base pointer of the allocation. An IPC handle of type
        ``cu_ipc_mem_handle`` is constructed and initialized with
        ``cuIpcGetMemHandle``. A :class:`numba.cuda.IpcHandle` is returned,
        populated with the underlying ``ipc_mem_handle``.
        """

class NumbaCUDAMemoryManager(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
    """Internal on-device memory management for Numba. This is implemented using
    the EMM Plugin interface, but is not part of the public API."""
    def initialize(self) -> None: ...
    def memalloc(self, size): ...
    def get_memory_info(self): ...
    @property
    def interface_version(self): ...

_SUPPORTED_EMM_INTERFACE_VERSION: int
_memory_manager: Incomplete

def _ensure_memory_manager() -> None: ...
def set_memory_manager(mm_plugin) -> None:
    """Configure Numba to use an External Memory Management (EMM) Plugin. If
    the EMM Plugin version does not match one supported by this version of
    Numba, a RuntimeError will be raised.

    :param mm_plugin: The class implementing the EMM Plugin.
    :type mm_plugin: BaseCUDAMemoryManager
    :return: None
    """

class _SizeNotSet(int):
    """
    Dummy object for _PendingDeallocs when *size* is not set.
    """
    def __new__(cls, *args, **kwargs): ...
    def __str__(self) -> str: ...

class _PendingDeallocs:
    """
    Pending deallocations of a context (or device since we are using the primary
    context). The capacity defaults to being unset (_SizeNotSet) but can be
    modified later once the driver is initialized and the total memory capacity
    known.
    """
    _cons: Incomplete
    _disable_count: int
    _size: int
    memory_capacity: Incomplete
    def __init__(self, capacity=...) -> None: ...
    @property
    def _max_pending_bytes(self): ...
    def add_item(self, dtor, handle, size=...) -> None:
        """
        Add a pending deallocation.

        The *dtor* arg is the destructor function that takes an argument,
        *handle*.  It is used as ``dtor(handle)``.  The *size* arg is the
        byte size of the resource added.  It is an optional argument.  Some
        resources (e.g. CUModule) has an unknown memory footprint on the device.
        """
    def clear(self) -> None:
        """
        Flush any pending deallocations unless it is disabled.
        Do nothing if disabled.
        """
    def disable(self) -> Generator[None]:
        """
        Context manager to temporarily disable flushing pending deallocation.
        This can be nested.
        """
    @property
    def is_disabled(self): ...
    def __len__(self) -> int:
        """
        Returns number of pending deallocations.
        """

class MemoryInfo(NamedTuple):
    free: Incomplete
    total: Incomplete

class Context:
    """
    This object wraps a CUDA Context resource.

    Contexts should not be constructed directly by user code.
    """
    device: Incomplete
    handle: Incomplete
    allocations: Incomplete
    deallocations: Incomplete
    memory_manager: Incomplete
    modules: Incomplete
    extras: Incomplete
    def __init__(self, device, handle) -> None: ...
    def reset(self) -> None:
        """
        Clean up all owned resources in this context.
        """
    def get_memory_info(self):
        """Returns (free, total) memory in bytes in the context.
        """
    def get_active_blocks_per_multiprocessor(self, func, blocksize, memsize, flags: Incomplete | None = None):
        """Return occupancy of a function.
        :param func: kernel for which occupancy is calculated
        :param blocksize: block size the kernel is intended to be launched with
        :param memsize: per-block dynamic shared memory usage intended, in bytes
        """
    def _cuda_python_active_blocks_per_multiprocessor(self, func, blocksize, memsize, flags): ...
    def _ctypes_active_blocks_per_multiprocessor(self, func, blocksize, memsize, flags): ...
    def get_max_potential_block_size(self, func, b2d_func, memsize, blocksizelimit, flags: Incomplete | None = None):
        """Suggest a launch configuration with reasonable occupancy.
        :param func: kernel for which occupancy is calculated
        :param b2d_func: function that calculates how much per-block dynamic
                         shared memory 'func' uses based on the block size.
                         Can also be the address of a C function.
                         Use `0` to pass `NULL` to the underlying CUDA API.
        :param memsize: per-block dynamic shared memory usage intended, in bytes
        :param blocksizelimit: maximum block size the kernel is designed to
                               handle
        """
    def _ctypes_max_potential_block_size(self, func, b2d_func, memsize, blocksizelimit, flags): ...
    def _cuda_python_max_potential_block_size(self, func, b2d_func, memsize, blocksizelimit, flags): ...
    def prepare_for_use(self) -> None:
        """Initialize the context for use.
        It's safe to be called multiple times.
        """
    def push(self) -> None:
        """
        Pushes this context on the current CPU Thread.
        """
    def pop(self) -> None:
        """
        Pops this context off the current CPU thread. Note that this context
        must be at the top of the context stack, otherwise an error will occur.
        """
    def memalloc(self, bytesize): ...
    def memallocmanaged(self, bytesize, attach_global: bool = True): ...
    def memhostalloc(self, bytesize, mapped: bool = False, portable: bool = False, wc: bool = False): ...
    def mempin(self, owner, pointer, size, mapped: bool = False): ...
    def get_ipc_handle(self, memory):
        """
        Returns an *IpcHandle* from a GPU allocation.
        """
    def open_ipc_handle(self, handle, size): ...
    def enable_peer_access(self, peer_context, flags: int = 0) -> None:
        """Enable peer access between the current context and the peer context
        """
    def can_access_peer(self, peer_device):
        """Returns a bool indicating whether the peer access between the
        current and peer device is possible.
        """
    def create_module_ptx(self, ptx): ...
    def create_module_image(self, image): ...
    def unload_module(self, module) -> None: ...
    def get_default_stream(self): ...
    def get_legacy_default_stream(self): ...
    def get_per_thread_default_stream(self): ...
    def create_stream(self): ...
    def create_external_stream(self, ptr): ...
    def create_event(self, timing: bool = True): ...
    def synchronize(self) -> None: ...
    def defer_cleanup(self) -> Generator[None]: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...

def load_module_image(context, image):
    """
    image must be a pointer
    """
def load_module_image_ctypes(context, image): ...
def load_module_image_cuda_python(context, image):
    """
    image must be a pointer
    """
def _alloc_finalizer(memory_manager, ptr, alloc_key, size): ...
def _hostalloc_finalizer(memory_manager, ptr, alloc_key, size, mapped):
    """
    Finalize page-locked host memory allocated by `context.memhostalloc`.

    This memory is managed by CUDA, and finalization entails deallocation. The
    issues noted in `_pin_finalizer` are not relevant in this case, and the
    finalization is placed in the `context.deallocations` queue along with
    finalization of device objects.

    """
def _pin_finalizer(memory_manager, ptr, alloc_key, mapped):
    """
    Finalize temporary page-locking of host memory by `context.mempin`.

    This applies to memory not otherwise managed by CUDA. Page-locking can
    be requested multiple times on the same memory, and must therefore be
    lifted as soon as finalization is requested, otherwise subsequent calls to
    `mempin` may fail with `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`, leading
    to unexpected behavior for the context managers `cuda.{pinned,mapped}`.
    This function therefore carries out finalization immediately, bypassing the
    `context.deallocations` queue.

    """
def _event_finalizer(deallocs, handle): ...
def _stream_finalizer(deallocs, handle): ...
def _module_finalizer(context, handle): ...

class _CudaIpcImpl:
    """Implementation of GPU IPC using CUDA driver API.
    This requires the devices to be peer accessible.
    """
    base: Incomplete
    handle: Incomplete
    size: Incomplete
    offset: Incomplete
    _opened_mem: Incomplete
    def __init__(self, parent) -> None: ...
    def open(self, context):
        """
        Import the IPC memory and returns a raw CUDA memory pointer object
        """
    def close(self) -> None: ...

class _StagedIpcImpl:
    """Implementation of GPU IPC using custom staging logic to workaround
    CUDA IPC limitation on peer accessibility between devices.
    """
    parent: Incomplete
    base: Incomplete
    handle: Incomplete
    size: Incomplete
    source_info: Incomplete
    def __init__(self, parent, source_info) -> None: ...
    def open(self, context): ...
    def close(self) -> None: ...

class IpcHandle:
    """
    CUDA IPC handle. Serialization of the CUDA IPC handle object is implemented
    here.

    :param base: A reference to the original allocation to keep it alive
    :type base: MemoryPointer
    :param handle: The CUDA IPC handle, as a ctypes array of bytes.
    :param size: Size of the original allocation
    :type size: int
    :param source_info: The identity of the device on which the IPC handle was
                        opened.
    :type source_info: dict
    :param offset: The offset into the underlying allocation of the memory
                   referred to by this IPC handle.
    :type offset: int
    """
    base: Incomplete
    handle: Incomplete
    size: Incomplete
    source_info: Incomplete
    _impl: Incomplete
    offset: Incomplete
    def __init__(self, base, handle, size, source_info: Incomplete | None = None, offset: int = 0) -> None: ...
    def _sentry_source_info(self) -> None: ...
    def can_access_peer(self, context):
        """Returns a bool indicating whether the active context can peer
        access the IPC handle
        """
    def open_staged(self, context):
        """Open the IPC by allowing staging on the host memory first.
        """
    def open_direct(self, context):
        """
        Import the IPC memory and returns a raw CUDA memory pointer object
        """
    def open(self, context):
        """Open the IPC handle and import the memory for usage in the given
        context.  Returns a raw CUDA memory pointer object.

        This is enhanced over CUDA IPC that it will work regardless of whether
        the source device is peer-accessible by the destination device.
        If the devices are peer-accessible, it uses .open_direct().
        If the devices are not peer-accessible, it uses .open_staged().
        """
    def open_array(self, context, shape, dtype, strides: Incomplete | None = None):
        """
        Similar to `.open()` but returns an device array.
        """
    def close(self) -> None: ...
    def __reduce__(self): ...
    @classmethod
    def _rebuild(cls, handle_ary, size, source_info, offset): ...

class MemoryPointer:
    """A memory pointer that owns a buffer, with an optional finalizer. Memory
    pointers provide reference counting, and instances are initialized with a
    reference count of 1.

    The base ``MemoryPointer`` class does not use the
    reference count for managing the buffer lifetime. Instead, the buffer
    lifetime is tied to the memory pointer instance's lifetime:

    - When the instance is deleted, the finalizer will be called.
    - When the reference count drops to 0, no action is taken.

    Subclasses of ``MemoryPointer`` may modify these semantics, for example to
    tie the buffer lifetime to the reference count, so that the buffer is freed
    when there are no more references.

    :param context: The context in which the pointer was allocated.
    :type context: Context
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the allocation in bytes.
    :type size: int
    :param owner: The owner is sometimes set by the internals of this class, or
                  used for Numba's internal memory management. It should not be
                  provided by an external user of the ``MemoryPointer`` class
                  (e.g. from within an EMM Plugin); the default of `None`
                  should always suffice.
    :type owner: NoneType
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """
    __cuda_memory__: bool
    context: Incomplete
    device_pointer: Incomplete
    size: Incomplete
    _cuda_memsize_: Incomplete
    is_managed: Incomplete
    refct: int
    handle: Incomplete
    _owner: Incomplete
    _finalizer: Incomplete
    def __init__(self, context, pointer, size, owner: Incomplete | None = None, finalizer: Incomplete | None = None) -> None: ...
    @property
    def owner(self): ...
    def own(self): ...
    def free(self) -> None:
        """
        Forces the device memory to the trash.
        """
    def memset(self, byte, count: Incomplete | None = None, stream: int = 0) -> None: ...
    def view(self, start, stop: Incomplete | None = None): ...
    @property
    def device_ctypes_pointer(self): ...
    @property
    def device_pointer_value(self): ...

class AutoFreePointer(MemoryPointer):
    """Modifies the ownership semantic of the MemoryPointer so that the
    instance lifetime is directly tied to the number of references.

    When the reference count reaches zero, the finalizer is invoked.

    Constructor arguments are the same as for :class:`MemoryPointer`.
    """
    def __init__(self, *args, **kwargs) -> None: ...

class MappedMemory(AutoFreePointer):
    """A memory pointer that refers to a buffer on the host that is mapped into
    device memory.

    :param context: The context in which the pointer was mapped.
    :type context: Context
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the buffer in bytes.
    :type size: int
    :param owner: The owner is sometimes set by the internals of this class, or
                  used for Numba's internal memory management. It should not be
                  provided by an external user of the ``MappedMemory`` class
                  (e.g. from within an EMM Plugin); the default of `None`
                  should always suffice.
    :type owner: NoneType
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """
    __cuda_memory__: bool
    owned: Incomplete
    host_pointer: Incomplete
    _bufptr_: Incomplete
    device_pointer: Incomplete
    handle: Incomplete
    _buflen_: Incomplete
    def __init__(self, context, pointer, size, owner: Incomplete | None = None, finalizer: Incomplete | None = None) -> None: ...
    def own(self): ...

class PinnedMemory(mviewbuf.MemAlloc):
    """A pointer to a pinned buffer on the host.

    :param context: The context in which the pointer was mapped.
    :type context: Context
    :param owner: The object owning the memory. For EMM plugin implementation,
                  this ca
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the buffer in bytes.
    :type size: int
    :param owner: An object owning the buffer that has been pinned. For EMM
                  plugin implementation, the default of ``None`` suffices for
                  memory allocated in ``memhostalloc`` - for ``mempin``, it
                  should be the owner passed in to the ``mempin`` method.
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """
    context: Incomplete
    owned: Incomplete
    size: Incomplete
    host_pointer: Incomplete
    is_managed: Incomplete
    handle: Incomplete
    _buflen_: Incomplete
    _bufptr_: Incomplete
    def __init__(self, context, pointer, size, owner: Incomplete | None = None, finalizer: Incomplete | None = None) -> None: ...
    def own(self): ...

class ManagedMemory(AutoFreePointer):
    """A memory pointer that refers to a managed memory buffer (can be accessed
    on both host and device).

    :param context: The context in which the pointer was mapped.
    :type context: Context
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the buffer in bytes.
    :type size: int
    :param owner: The owner is sometimes set by the internals of this class, or
                  used for Numba's internal memory management. It should not be
                  provided by an external user of the ``ManagedMemory`` class
                  (e.g. from within an EMM Plugin); the default of `None`
                  should always suffice.
    :type owner: NoneType
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """
    __cuda_memory__: bool
    owned: Incomplete
    _buflen_: Incomplete
    _bufptr_: Incomplete
    def __init__(self, context, pointer, size, owner: Incomplete | None = None, finalizer: Incomplete | None = None) -> None: ...
    def own(self): ...

class OwnedPointer:
    _mem: Incomplete
    _view: Incomplete
    def __init__(self, memptr, view: Incomplete | None = None) -> None: ...
    def __getattr__(self, fname):
        """Proxy MemoryPointer methods
        """

class MappedOwnedPointer(OwnedPointer, mviewbuf.MemAlloc): ...
class ManagedOwnedPointer(OwnedPointer, mviewbuf.MemAlloc): ...

class Stream:
    context: Incomplete
    handle: Incomplete
    external: Incomplete
    def __init__(self, context, handle, finalizer, external: bool = False) -> None: ...
    def __int__(self) -> int: ...
    def __repr__(self) -> str: ...
    def synchronize(self) -> None:
        """
        Wait for all commands in this stream to execute. This will commit any
        pending memory transfers.
        """
    def auto_synchronize(self) -> Generator[Incomplete]:
        """
        A context manager that waits for all commands in this stream to execute
        and commits any pending memory transfers upon exiting the context.
        """
    def add_callback(self, callback, arg: Incomplete | None = None) -> None:
        """
        Add a callback to a compute stream.
        The user provided function is called from a driver thread once all
        preceding stream operations are complete.

        Callback functions are called from a CUDA driver thread, not from
        the thread that invoked `add_callback`. No CUDA API functions may
        be called from within the callback function.

        The duration of a callback function should be kept short, as the
        callback will block later work in the stream and may block other
        callbacks from being executed.

        Note: The driver function underlying this method is marked for
        eventual deprecation and may be replaced in a future CUDA release.

        :param callback: Callback function with arguments (stream, status, arg).
        :param arg: Optional user data to be passed to the callback function.
        """
    @staticmethod
    def _stream_callback(handle, status, data) -> None: ...
    def async_done(self) -> asyncio.futures.Future:
        """
        Return an awaitable that resolves once all preceding stream operations
        are complete. The result of the awaitable is the current stream.
        """

class Event:
    context: Incomplete
    handle: Incomplete
    def __init__(self, context, handle, finalizer: Incomplete | None = None) -> None: ...
    def query(self):
        """
        Returns True if all work before the most recent record has completed;
        otherwise, returns False.
        """
    def record(self, stream: int = 0) -> None:
        """
        Set the record point of the event to the current point in the given
        stream.

        The event will be considered to have occurred when all work that was
        queued in the stream at the time of the call to ``record()`` has been
        completed.
        """
    def synchronize(self) -> None:
        """
        Synchronize the host thread for the completion of the event.
        """
    def wait(self, stream: int = 0) -> None:
        """
        All future works submitted to stream will wait util the event completes.
        """
    def elapsed_time(self, evtend): ...

def event_elapsed_time(evtstart, evtend):
    """
    Compute the elapsed time between two events in milliseconds.
    """

class Module(metaclass=ABCMeta):
    """Abstract base class for modules"""
    context: Incomplete
    handle: Incomplete
    info_log: Incomplete
    _finalizer: Incomplete
    def __init__(self, context, handle, info_log, finalizer: Incomplete | None = None) -> None: ...
    def unload(self) -> None:
        """Unload this module from the context"""
    @abstractmethod
    def get_function(self, name):
        """Returns a Function object encapsulating the named function"""
    @abstractmethod
    def get_global_symbol(self, name):
        """Return a MemoryPointer referring to the named symbol"""

class CtypesModule(Module):
    def get_function(self, name): ...
    def get_global_symbol(self, name): ...

class CudaPythonModule(Module):
    def get_function(self, name): ...
    def get_global_symbol(self, name): ...

class FuncAttr(NamedTuple):
    regs: Incomplete
    shared: Incomplete
    local: Incomplete
    const: Incomplete
    maxthreads: Incomplete

class Function(metaclass=ABCMeta):
    griddim: Incomplete
    blockdim: Incomplete
    stream: int
    sharedmem: int
    module: Incomplete
    handle: Incomplete
    name: Incomplete
    attrs: Incomplete
    def __init__(self, module, handle, name) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def device(self): ...
    @abstractmethod
    def cache_config(self, prefer_equal: bool = False, prefer_cache: bool = False, prefer_shared: bool = False):
        """Set the cache configuration for this function."""
    @abstractmethod
    def read_func_attr(self, attrid):
        """Return the value of the attribute with given ID."""
    @abstractmethod
    def read_func_attr_all(self):
        """Return a FuncAttr object with the values of various function
        attributes."""

class CtypesFunction(Function):
    def cache_config(self, prefer_equal: bool = False, prefer_cache: bool = False, prefer_shared: bool = False) -> None: ...
    def read_func_attr(self, attrid): ...
    def read_func_attr_all(self): ...

class CudaPythonFunction(Function):
    def cache_config(self, prefer_equal: bool = False, prefer_cache: bool = False, prefer_shared: bool = False) -> None: ...
    def read_func_attr(self, attrid): ...
    def read_func_attr_all(self): ...

def launch_kernel(cufunc_handle, gx, gy, gz, bx, by, bz, sharedmem, hstream, args, cooperative: bool = False) -> None: ...

jitty: Incomplete
FILE_EXTENSION_MAP: Incomplete

class Linker(metaclass=ABCMeta):
    """Abstract base class for linkers"""
    @classmethod
    def new(cls, max_registers: int = 0, lineinfo: bool = False, cc: Incomplete | None = None): ...
    lto: bool
    @abstractmethod
    def __init__(self, max_registers, lineinfo, cc): ...
    @property
    @abstractmethod
    def info_log(self):
        """Return the info log from the linker invocation"""
    @property
    @abstractmethod
    def error_log(self):
        """Return the error log from the linker invocation"""
    @abstractmethod
    def add_ptx(self, ptx, name):
        """Add PTX source in a string to the link"""
    def add_cu(self, cu, name) -> None:
        """Add CUDA source in a string to the link. The name of the source
        file should be specified in `name`."""
    @abstractmethod
    def add_file(self, path, kind):
        """Add code from a file to the link"""
    def add_cu_file(self, path) -> None: ...
    def add_file_guess_ext(self, path) -> None:
        """Add a file to the link, guessing its type from its extension."""
    @abstractmethod
    def complete(self):
        """Complete the link. Returns (cubin, size)

        cubin is a pointer to a internal buffer of cubin owned by the linker;
        thus, it should be loaded before the linker is destroyed.
        """

_MVC_ERROR_MESSAGE: str

class MVCLinker(Linker):
    """
    Linker supporting Minor Version Compatibility, backed by the cubinlinker
    package.
    """
    ptx_compile_options: Incomplete
    _linker: Incomplete
    def __init__(self, max_registers: Incomplete | None = None, lineinfo: bool = False, cc: Incomplete | None = None) -> None: ...
    @property
    def info_log(self): ...
    @property
    def error_log(self): ...
    def add_ptx(self, ptx, name: str = '<cudapy-ptx>') -> None: ...
    def add_file(self, path, kind): ...
    def complete(self): ...

class CtypesLinker(Linker):
    """
    Links for current device if no CC given
    """
    handle: Incomplete
    linker_info_buf: Incomplete
    linker_errors_buf: Incomplete
    _keep_alive: Incomplete
    def __init__(self, max_registers: int = 0, lineinfo: bool = False, cc: Incomplete | None = None) -> None: ...
    @property
    def info_log(self): ...
    @property
    def error_log(self): ...
    def add_ptx(self, ptx, name: str = '<cudapy-ptx>') -> None: ...
    def add_file(self, path, kind) -> None: ...
    def complete(self): ...

class CudaPythonLinker(Linker):
    """
    Links for current device if no CC given
    """
    handle: Incomplete
    linker_info_buf: Incomplete
    linker_errors_buf: Incomplete
    _keep_alive: Incomplete
    def __init__(self, max_registers: int = 0, lineinfo: bool = False, cc: Incomplete | None = None) -> None: ...
    @property
    def info_log(self): ...
    @property
    def error_log(self): ...
    def add_ptx(self, ptx, name: str = '<cudapy-ptx>') -> None: ...
    def add_file(self, path, kind) -> None: ...
    def complete(self): ...

def get_devptr_for_active_ctx(ptr):
    """Query the device pointer usable in the current context from an arbitrary
    pointer.
    """
def device_extents(devmem):
    """Find the extents (half open begin and end pointer) of the underlying
    device memory allocation.

    NOTE: it always returns the extents of the allocation but the extents
    of the device memory view that can be a subsection of the entire allocation.
    """
def device_memory_size(devmem):
    """Check the memory size of the device memory.
    The result is cached in the device memory object.
    It may query the driver for the memory size of the device memory allocation.
    """
def _is_datetime_dtype(obj):
    """Returns True if the obj.dtype is datetime64 or timedelta64
    """
def _workaround_for_datetime(obj):
    """Workaround for numpy#4983: buffer protocol doesn't support
    datetime64 or timedelta64.
    """
def host_pointer(obj, readonly: bool = False):
    """Get host pointer from an obj.

    If `readonly` is False, the buffer must be writable.

    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
def host_memory_extents(obj):
    """Returns (start, end) the start and end pointer of the array (half open)."""
def memory_size_from_info(shape, strides, itemsize):
    """Get the byte size of a contiguous memory buffer given the shape, strides
    and itemsize.
    """
def host_memory_size(obj):
    """Get the size of the memory"""
def device_pointer(obj):
    """Get the device pointer as an integer"""
def device_ctypes_pointer(obj):
    """Get the ctypes object for the device pointer"""
def is_device_memory(obj):
    '''All CUDA memory object is recognized as an instance with the attribute
    "__cuda_memory__" defined and its value evaluated to True.

    All CUDA memory object should also define an attribute named
    "device_pointer" which value is an int object carrying the pointer
    value of the device memory address.  This is not tested in this method.
    '''
def require_device_memory(obj) -> None:
    """A sentry for methods that accept CUDA memory object.
    """
def device_memory_depends(devmem, *objs) -> None:
    """Add dependencies to the device memory.

    Mainly used for creating structures that points to other device memory,
    so that the referees are not GC and released.
    """
def host_to_device(dst, src, size, stream: int = 0) -> None:
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
def device_to_host(dst, src, size, stream: int = 0) -> None:
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
def device_to_device(dst, src, size, stream: int = 0) -> None:
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
def device_memset(dst, val, size, stream: int = 0) -> None:
    """Memset on the device.
    If stream is not zero, asynchronous mode is used.

    dst: device memory
    val: byte value to be written
    size: number of byte to be written
    stream: a CUDA stream
    """
def profile_start() -> None:
    """
    Enable profile collection in the current context.
    """
def profile_stop() -> None:
    """
    Disable profile collection in the current context.
    """
def profiling() -> Generator[None]:
    """
    Context manager that enables profiling on entry and disables profiling on
    exit.
    """
def get_version():
    """
    Return the driver version as a tuple of (major, minor)
    """
