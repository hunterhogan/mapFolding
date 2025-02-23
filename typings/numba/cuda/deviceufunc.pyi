from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from numba.core import sigutils as sigutils, types as types
from numba.core.typing import signature as signature
from numba.np.ufunc.sigparse import parse_signature as parse_signature
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder as _BaseUFuncBuilder, parse_identity as parse_identity

def _broadcast_axis(a, b):
    """
    Raises
    ------
    ValueError if broadcast fails
    """
def _pairwise_broadcast(shape1, shape2):
    """
    Raises
    ------
    ValueError if broadcast fails
    """
def _multi_broadcast(*shapelist):
    """
    Raises
    ------
    ValueError if broadcast fails
    """

class UFuncMechanism:
    """
    Prepare ufunc arguments for vectorize.
    """
    DEFAULT_STREAM: Incomplete
    SUPPORT_DEVICE_SLICING: bool
    typemap: Incomplete
    args: Incomplete
    argtypes: Incomplete
    scalarpos: Incomplete
    signature: Incomplete
    arrays: Incomplete
    def __init__(self, typemap, args) -> None:
        """Never used directly by user. Invoke by UFuncMechanism.call().
        """
    def _fill_arrays(self) -> None:
        """
        Get all arguments in array form
        """
    def _fill_argtypes(self) -> None:
        """
        Get dtypes
        """
    def _resolve_signature(self) -> None:
        """Resolve signature.
        May have ambiguous case.
        """
    def _get_actual_args(self):
        """Return the actual arguments
        Casts scalar arguments to np.array.
        """
    def _broadcast(self, arys):
        """Perform numpy ufunc broadcasting
        """
    def get_arguments(self):
        """Prepare and return the arguments for the ufunc.
        Does not call to_device().
        """
    def get_function(self):
        """Returns (result_dtype, function)
        """
    def is_device_array(self, obj):
        """Is the `obj` a device array?
        Override in subclass
        """
    def as_device_array(self, obj):
        """Convert the `obj` to a device array
        Override in subclass

        Default implementation is an identity function
        """
    def broadcast_device(self, ary, shape) -> None:
        """Handles ondevice broadcasting

        Override in subclass to add support.
        """
    def force_array_layout(self, ary):
        """Ensures array layout met device requirement.

        Override in sublcass
        """
    @classmethod
    def call(cls, typemap, args, kws):
        """Perform the entire ufunc call mechanism.
        """
    def to_device(self, hostary, stream) -> None:
        """Implement to device transfer
        Override in subclass
        """
    def to_host(self, devary, stream) -> None:
        """Implement to host transfer
        Override in subclass
        """
    def allocate_device_array(self, shape, dtype, stream) -> None:
        """Implements device allocation
        Override in subclass
        """
    def launch(self, func, count, stream, args) -> None:
        """Implements device function invocation
        Override in subclass
        """

def to_dtype(ty): ...

class DeviceVectorize(_BaseUFuncBuilder):
    py_func: Incomplete
    identity: Incomplete
    kernelmap: Incomplete
    def __init__(self, func, identity: Incomplete | None = None, cache: bool = False, targetoptions={}) -> None: ...
    @property
    def pyfunc(self): ...
    def add(self, sig: Incomplete | None = None) -> None: ...
    def build_ufunc(self) -> None: ...
    def _get_kernel_source(self, template, sig, funcname): ...
    def _compile_core(self, sig) -> None: ...
    def _get_globals(self, corefn) -> None: ...
    def _compile_kernel(self, fnobj, sig) -> None: ...

class DeviceGUFuncVectorize(_BaseUFuncBuilder):
    py_func: Incomplete
    identity: Incomplete
    signature: Incomplete
    kernelmap: Incomplete
    def __init__(self, func, sig, identity: Incomplete | None = None, cache: bool = False, targetoptions={}, writable_args=()) -> None: ...
    @property
    def pyfunc(self): ...
    def add(self, sig: Incomplete | None = None) -> None: ...
    def _compile_kernel(self, fnobj, sig) -> None: ...
    def _get_globals(self, sig) -> None: ...

def _determine_gufunc_outer_types(argtys, dims) -> Generator[Incomplete]: ...
def expand_gufunc_template(template, indims, outdims, funcname, argtypes):
    """Expand gufunc source template
    """
def _gen_src_for_indexing(aref, adims, atype): ...
def _gen_src_index(adims, atype): ...

class GUFuncEngine:
    """Determine how to broadcast and execute a gufunc
    base on input shape and signature
    """
    @classmethod
    def from_signature(cls, signature): ...
    sin: Incomplete
    sout: Incomplete
    nin: Incomplete
    nout: Incomplete
    def __init__(self, inputsig, outputsig) -> None: ...
    def schedule(self, ishapes): ...

class GUFuncSchedule:
    parent: Incomplete
    ishapes: Incomplete
    oshapes: Incomplete
    loopdims: Incomplete
    loopn: Incomplete
    pinned: Incomplete
    output_shapes: Incomplete
    def __init__(self, parent, ishapes, oshapes, loopdims, pinned) -> None: ...
    def __str__(self) -> str: ...

class GeneralizedUFunc:
    kernelmap: Incomplete
    engine: Incomplete
    max_blocksize: Incomplete
    def __init__(self, kernelmap, engine) -> None: ...
    def __call__(self, *args, **kws): ...
    def _schedule(self, inputs, outs): ...
    def _search_matching_signature(self, idtypes):
        """
        Given the input types in `idtypes`, return a compatible sequence of
        types that is defined in `kernelmap`.

        Note: Ordering is guaranteed by `kernelmap` being a OrderedDict
        """
    def _broadcast(self, schedule, params, retvals): ...
    def _broadcast_array(self, ary, newdim, innerdim): ...
    def _broadcast_add_axis(self, ary, newshape) -> None: ...
    def _broadcast_scalar_input(self, ary, shape) -> None: ...

class GUFuncCallSteps(metaclass=ABCMeta):
    """
    Implements memory management and kernel launch operations for GUFunc calls.

    One instance of this class is instantiated for each call, and the instance
    is specific to the arguments given to the GUFunc call.

    The base class implements the overall logic; subclasses provide
    target-specific implementations of individual functions.
    """
    __slots__: Incomplete
    @abstractmethod
    def launch_kernel(self, kernel, nelem, args):
        """Implement the kernel launch"""
    @abstractmethod
    def is_device_array(self, obj):
        """
        Return True if `obj` is a device array for this target, False
        otherwise.
        """
    @abstractmethod
    def as_device_array(self, obj):
        """
        Return `obj` as a device array on this target.

        May return `obj` directly if it is already on the target.
        """
    @abstractmethod
    def to_device(self, hostary):
        """
        Copy `hostary` to the device and return the device array.
        """
    @abstractmethod
    def allocate_device_array(self, shape, dtype):
        """
        Allocate a new uninitialized device array with the given shape and
        dtype.
        """
    outputs: Incomplete
    _copy_result_to_host: Incomplete
    inputs: Incomplete
    def __init__(self, nin, nout, args, kwargs) -> None: ...
    def adjust_input_types(self, indtypes) -> None:
        """
        Attempt to cast the inputs to the required types if necessary
        and if they are not device arrays.

        Side effect: Only affects the elements of `inputs` that require
        a type cast.
        """
    def prepare_outputs(self, schedule, outdtypes):
        """
        Returns a list of output parameters that all reside on the target
        device.

        Outputs that were passed-in to the GUFunc are used if they reside on the
        device; other outputs are allocated as necessary.
        """
    def prepare_inputs(self):
        """
        Returns a list of input parameters that all reside on the target device.
        """
    def post_process_outputs(self, outputs):
        """
        Moves the given output(s) to the host if necessary.

        Returns a single value (e.g. an array) if there was one output, or a
        tuple of arrays if there were multiple. Although this feels a little
        jarring, it is consistent with the behavior of GUFuncs in general.
        """
