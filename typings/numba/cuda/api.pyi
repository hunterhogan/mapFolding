from .cudadrv import devicearray as devicearray, devices as devices, driver as driver
from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import config as config
from numba.cuda.api_util import prepare_shape_strides_dtype as prepare_shape_strides_dtype

require_context: Incomplete
current_context: Incomplete
gpus: Incomplete

def from_cuda_array_interface(desc, owner: Incomplete | None = None, sync: bool = True):
    """Create a DeviceNDArray from a cuda-array-interface description.
    The ``owner`` is the owner of the underlying memory.
    The resulting DeviceNDArray will acquire a reference from it.

    If ``sync`` is ``True``, then the imported stream (if present) will be
    synchronized.
    """
def as_cuda_array(obj, sync: bool = True):
    """Create a DeviceNDArray from any object that implements
    the :ref:`cuda array interface <cuda-array-interface>`.

    A view of the underlying GPU buffer is created.  No copying of the data
    is done.  The resulting DeviceNDArray will acquire a reference from `obj`.

    If ``sync`` is ``True``, then the imported stream (if present) will be
    synchronized.
    """
def is_cuda_array(obj):
    """Test if the object has defined the `__cuda_array_interface__` attribute.

    Does not verify the validity of the interface.
    """
def is_float16_supported():
    """Whether 16-bit floats are supported.

    float16 is always supported in current versions of Numba - returns True.
    """
def to_device(obj, stream: int = 0, copy: bool = True, to: Incomplete | None = None):
    """to_device(obj, stream=0, copy=True, to=None)

    Allocate and transfer a numpy ndarray or structured scalar to the device.

    To copy host->device a numpy array::

        ary = np.arange(10)
        d_ary = cuda.to_device(ary)

    To enqueue the transfer to a stream::

        stream = cuda.stream()
        d_ary = cuda.to_device(ary, stream=stream)

    The resulting ``d_ary`` is a ``DeviceNDArray``.

    To copy device->host::

        hary = d_ary.copy_to_host()

    To copy device->host to an existing array::

        ary = np.empty(shape=d_ary.shape, dtype=d_ary.dtype)
        d_ary.copy_to_host(ary)

    To enqueue the transfer to a stream::

        hary = d_ary.copy_to_host(stream=stream)
    """
def device_array(shape, dtype=..., strides: Incomplete | None = None, order: str = 'C', stream: int = 0):
    """device_array(shape, dtype=np.float64, strides=None, order='C', stream=0)

    Allocate an empty device ndarray. Similar to :meth:`numpy.empty`.
    """
def managed_array(shape, dtype=..., strides: Incomplete | None = None, order: str = 'C', stream: int = 0, attach_global: bool = True):
    """managed_array(shape, dtype=np.float64, strides=None, order='C', stream=0,
                     attach_global=True)

    Allocate a np.ndarray with a buffer that is managed.
    Similar to np.empty().

    Managed memory is supported on Linux / x86 and PowerPC, and is considered
    experimental on Windows and Linux / AArch64.

    :param attach_global: A flag indicating whether to attach globally. Global
                          attachment implies that the memory is accessible from
                          any stream on any device. If ``False``, attachment is
                          *host*, and memory is only accessible by devices
                          with Compute Capability 6.0 and later.
    """
def pinned_array(shape, dtype=..., strides: Incomplete | None = None, order: str = 'C'):
    """pinned_array(shape, dtype=np.float64, strides=None, order='C')

    Allocate an :class:`ndarray <numpy.ndarray>` with a buffer that is pinned
    (pagelocked).  Similar to :func:`np.empty() <numpy.empty>`.
    """
def mapped_array(shape, dtype=..., strides: Incomplete | None = None, order: str = 'C', stream: int = 0, portable: bool = False, wc: bool = False):
    """mapped_array(shape, dtype=np.float64, strides=None, order='C', stream=0,
                    portable=False, wc=False)

    Allocate a mapped ndarray with a buffer that is pinned and mapped on
    to the device. Similar to np.empty()

    :param portable: a boolean flag to allow the allocated device memory to be
              usable in multiple devices.
    :param wc: a boolean flag to enable writecombined allocation which is faster
        to write by the host and to read by the device, but slower to
        write by the host and slower to write by the device.
    """
def open_ipc_array(handle, shape, dtype, strides: Incomplete | None = None, offset: int = 0) -> Generator[Incomplete]:
    """
    A context manager that opens a IPC *handle* (*CUipcMemHandle*) that is
    represented as a sequence of bytes (e.g. *bytes*, tuple of int)
    and represent it as an array of the given *shape*, *strides* and *dtype*.
    The *strides* can be omitted.  In that case, it is assumed to be a 1D
    C contiguous array.

    Yields a device array.

    The IPC handle is closed automatically when context manager exits.
    """
def synchronize():
    """Synchronize the current context."""
def _contiguous_strides_like_array(ary):
    """
    Given an array, compute strides for a new contiguous array of the same
    shape.
    """
def _order_like_array(ary): ...
def device_array_like(ary, stream: int = 0):
    """
    Call :func:`device_array() <numba.cuda.device_array>` with information from
    the array.
    """
def mapped_array_like(ary, stream: int = 0, portable: bool = False, wc: bool = False):
    """
    Call :func:`mapped_array() <numba.cuda.mapped_array>` with the information
    from the array.
    """
def pinned_array_like(ary):
    """
    Call :func:`pinned_array() <numba.cuda.pinned_array>` with the information
    from the array.
    """
def stream():
    """
    Create a CUDA stream that represents a command queue for the device.
    """
def default_stream():
    """
    Get the default CUDA stream. CUDA semantics in general are that the default
    stream is either the legacy default stream or the per-thread default stream
    depending on which CUDA APIs are in use. In Numba, the APIs for the legacy
    default stream are always the ones in use, but an option to use APIs for
    the per-thread default stream may be provided in future.
    """
def legacy_default_stream():
    """
    Get the legacy default CUDA stream.
    """
def per_thread_default_stream():
    """
    Get the per-thread default CUDA stream.
    """
def external_stream(ptr):
    """Create a Numba stream object for a stream allocated outside Numba.

    :param ptr: Pointer to the external stream to wrap in a Numba Stream
    :type ptr: int
    """
def pinned(*arylist) -> Generator[None]:
    """A context manager for temporary pinning a sequence of host ndarrays.
    """
def mapped(*arylist, **kws) -> Generator[Incomplete]:
    """A context manager for temporarily mapping a sequence of host ndarrays.
    """
def event(timing: bool = True):
    """
    Create a CUDA event. Timing data is only recorded by the event if it is
    created with ``timing=True``.
    """

event_elapsed_time: Incomplete

def select_device(device_id):
    """
    Make the context associated with device *device_id* the current context.

    Returns a Device instance.

    Raises exception on error.
    """
def get_current_device():
    """Get current device associated with the current thread"""
def list_devices():
    """Return a list of all detected devices"""
def close() -> None:
    """
    Explicitly clears all contexts in the current thread, and destroys all
    contexts if the current thread is the main thread.
    """
def _auto_device(ary, stream: int = 0, copy: bool = True): ...
def detect():
    """
    Detect supported CUDA hardware and print a summary of the detected hardware.

    Returns a boolean indicating whether any supported devices were detected.
    """
def defer_cleanup() -> Generator[None]:
    """
    Temporarily disable memory deallocation.
    Use this to prevent resource deallocation breaking asynchronous execution.

    For example::

        with defer_cleanup():
            # all cleanup is deferred in here
            do_speed_critical_code()
        # cleanup can occur here

    Note: this context manager can be nested.
    """

profiling: Incomplete
profile_start: Incomplete
profile_stop: Incomplete
