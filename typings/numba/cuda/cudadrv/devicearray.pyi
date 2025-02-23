import numpy as np
from _typeshed import Incomplete
from collections.abc import Generator
from numba import _devicearray as _devicearray
from numba.core import config as config, types as types
from numba.core.errors import NumbaPerformanceWarning as NumbaPerformanceWarning
from numba.cuda.api_util import prepare_shape_strides_dtype as prepare_shape_strides_dtype
from numba.cuda.cudadrv import devices as devices, dummyarray as dummyarray
from numba.np import numpy_support as numpy_support
from numba.np.numpy_support import numpy_version as numpy_version
from numba.np.unsafe.ndarray import to_fixed_tuple as to_fixed_tuple

lru_cache: Incomplete

def is_cuda_ndarray(obj):
    """Check if an object is a CUDA ndarray"""
def verify_cuda_ndarray_interface(obj) -> None:
    """Verify the CUDA ndarray interface for an obj"""
def require_cuda_ndarray(obj) -> None:
    """Raises ValueError is is_cuda_ndarray(obj) evaluates False"""

class DeviceNDArrayBase(_devicearray.DeviceArray):
    """A on GPU NDArray representation
    """
    __cuda_memory__: bool
    __cuda_ndarray__: bool
    ndim: Incomplete
    _dummy: Incomplete
    shape: Incomplete
    strides: Incomplete
    dtype: Incomplete
    size: Incomplete
    alloc_size: Incomplete
    gpu_data: Incomplete
    stream: Incomplete
    def __init__(self, shape, strides, dtype, stream: int = 0, gpu_data: Incomplete | None = None) -> None:
        """
        Args
        ----

        shape
            array shape.
        strides
            array strides.
        dtype
            data type as np.dtype coercible object.
        stream
            cuda stream.
        gpu_data
            user provided device memory for the ndarray data buffer
        """
    @property
    def __cuda_array_interface__(self): ...
    def bind(self, stream: int = 0):
        """Bind a CUDA stream to this object so that all subsequent operation
        on this array defaults to the given stream.
        """
    @property
    def T(self): ...
    def transpose(self, axes: Incomplete | None = None): ...
    def _default_stream(self, stream): ...
    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
    @property
    def device_ctypes_pointer(self):
        """Returns the ctypes pointer to the GPU data buffer
        """
    def copy_to_device(self, ary, stream: int = 0) -> None:
        """Copy `ary` to `self`.

        If `ary` is a CUDA memory, perform a device-to-device transfer.
        Otherwise, perform a a host-to-device transfer.
        """
    def copy_to_host(self, ary: Incomplete | None = None, stream: int = 0):
        """Copy ``self`` to ``ary`` or create a new Numpy ndarray
        if ``ary`` is ``None``.

        If a CUDA ``stream`` is given, then the transfer will be made
        asynchronously as part as the given stream.  Otherwise, the transfer is
        synchronous: the function returns after the copy is finished.

        Always returns the host array.

        Example::

            import numpy as np
            from numba import cuda

            arr = np.arange(1000)
            d_arr = cuda.to_device(arr)

            my_kernel[100, 100](d_arr)

            result_array = d_arr.copy_to_host()
        """
    def split(self, section, stream: int = 0) -> Generator[Incomplete]:
        """Split the array into equal partition of the `section` size.
        If the array cannot be equally divided, the last section will be
        smaller.
        """
    def as_cuda_arg(self):
        """Returns a device memory object that is used as the argument.
        """
    def get_ipc_handle(self):
        """
        Returns a *IpcArrayHandle* object that is safe to serialize and transfer
        to another process to share the local allocation.

        Note: this feature is only available on Linux.
        """
    def squeeze(self, axis: Incomplete | None = None, stream: int = 0):
        """
        Remove axes of size one from the array shape.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Subset of dimensions to remove. A `ValueError` is raised if an axis
            with size greater than one is selected. If `None`, all axes with
            size one are removed.
        stream : cuda stream or 0, optional
            Default stream for the returned view of the array.

        Returns
        -------
        DeviceNDArray
            Squeezed view into the array.

        """
    def view(self, dtype):
        """Returns a new object by reinterpretting the dtype without making a
        copy of the data.
        """
    @property
    def nbytes(self): ...

class DeviceRecord(DeviceNDArrayBase):
    """
    An on-GPU record type
    """
    def __init__(self, dtype, stream: int = 0, gpu_data: Incomplete | None = None) -> None: ...
    @property
    def flags(self):
        """
        For `numpy.ndarray` compatibility. Ideally this would return a
        `np.core.multiarray.flagsobj`, but that needs to be constructed
        with an existing `numpy.ndarray` (as the C- and F- contiguous flags
        aren't writeable).
        """
    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
    def __getitem__(self, item): ...
    def getitem(self, item, stream: int = 0):
        """Do `__getitem__(item)` with CUDA stream
        """
    def _do_getitem(self, item, stream: int = 0): ...
    def __setitem__(self, key, value) -> None: ...
    def setitem(self, key, value, stream: int = 0):
        """Do `__setitem__(key, value)` with CUDA stream
        """
    def _do_setitem(self, key, value, stream: int = 0) -> None: ...

def _assign_kernel(ndim):
    """
    A separate method so we don't need to compile code every assignment (!).

    :param ndim: We need to have static array sizes for cuda.local.array, so
        bake in the number of dimensions into the kernel
    """

class DeviceNDArray(DeviceNDArrayBase):
    """
    An on-GPU array type
    """
    def is_f_contiguous(self):
        """
        Return true if the array is Fortran-contiguous.
        """
    @property
    def flags(self):
        """
        For `numpy.ndarray` compatibility. Ideally this would return a
        `np.core.multiarray.flagsobj`, but that needs to be constructed
        with an existing `numpy.ndarray` (as the C- and F- contiguous flags
        aren't writeable).
        """
    def is_c_contiguous(self):
        """
        Return true if the array is C-contiguous.
        """
    def __array__(self, dtype: Incomplete | None = None):
        """
        :return: an `numpy.ndarray`, so copies to the host.
        """
    def __len__(self) -> int: ...
    def reshape(self, *newshape, **kws):
        """
        Reshape the array without changing its contents, similarly to
        :meth:`numpy.ndarray.reshape`. Example::

            d_arr = d_arr.reshape(20, 50, order='F')
        """
    def ravel(self, order: str = 'C', stream: int = 0):
        """
        Flattens a contiguous array without changing its contents, similar to
        :meth:`numpy.ndarray.ravel`. If the array is not contiguous, raises an
        exception.
        """
    def __getitem__(self, item): ...
    def getitem(self, item, stream: int = 0):
        """Do `__getitem__(item)` with CUDA stream
        """
    def _do_getitem(self, item, stream: int = 0): ...
    def __setitem__(self, key, value) -> None: ...
    def setitem(self, key, value, stream: int = 0):
        """Do `__setitem__(key, value)` with CUDA stream
        """
    def _do_setitem(self, key, value, stream: int = 0) -> None: ...

class IpcArrayHandle:
    """
    An IPC array handle that can be serialized and transfer to another process
    in the same machine for share a GPU allocation.

    On the destination process, use the *.open()* method to creates a new
    *DeviceNDArray* object that shares the allocation from the original process.
    To release the resources, call the *.close()* method.  After that, the
    destination can no longer use the shared array object.  (Note: the
    underlying weakref to the resource is now dead.)

    This object implements the context-manager interface that calls the
    *.open()* and *.close()* method automatically::

        with the_ipc_array_handle as ipc_array:
            # use ipc_array here as a normal gpu array object
            some_code(ipc_array)
        # ipc_array is dead at this point
    """
    _array_desc: Incomplete
    _ipc_handle: Incomplete
    def __init__(self, ipc_handle, array_desc) -> None: ...
    def open(self):
        """
        Returns a new *DeviceNDArray* that shares the allocation from the
        original process.  Must not be used on the original process.
        """
    def close(self) -> None:
        """
        Closes the IPC handle to the array.
        """
    def __enter__(self): ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class MappedNDArray(DeviceNDArrayBase, np.ndarray):
    """
    A host array that uses CUDA mapped memory.
    """
    gpu_data: Incomplete
    stream: Incomplete
    def device_setup(self, gpu_data, stream: int = 0) -> None: ...

class ManagedNDArray(DeviceNDArrayBase, np.ndarray):
    """
    A host array that uses CUDA managed memory.
    """
    gpu_data: Incomplete
    stream: Incomplete
    def device_setup(self, gpu_data, stream: int = 0) -> None: ...

def from_array_like(ary, stream: int = 0, gpu_data: Incomplete | None = None):
    """Create a DeviceNDArray object that is like ary."""
def from_record_like(rec, stream: int = 0, gpu_data: Incomplete | None = None):
    """Create a DeviceRecord object that is like rec."""
def array_core(ary):
    """
    Extract the repeated core of a broadcast array.

    Broadcast arrays are by definition non-contiguous due to repeated
    dimensions, i.e., dimensions with stride 0. In order to ascertain memory
    contiguity and copy the underlying data from such arrays, we must create
    a view without the repeated dimensions.

    """
def is_contiguous(ary):
    """
    Returns True iff `ary` is C-style contiguous while ignoring
    broadcasted and 1-sized dimensions.
    As opposed to array_core(), it does not call require_context(),
    which can be quite expensive.
    """

errmsg_contiguous_buffer: str

def sentry_contiguous(ary) -> None: ...
def auto_device(obj, stream: int = 0, copy: bool = True, user_explicit: bool = False):
    """
    Create a DeviceRecord or DeviceArray like obj and optionally copy data from
    host to device. If obj already represents device memory, it is returned and
    no copy is made.
    """
def check_array_compatibility(ary1, ary2) -> None: ...
