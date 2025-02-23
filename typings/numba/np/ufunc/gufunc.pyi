from _typeshed import Incomplete
from numba import typeof as typeof
from numba.core import errors as errors, serialize as serialize, types as types
from numba.core.typing import npydecl as npydecl
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, signature as signature
from numba.np.numpy_support import ufunc_find_matching_loop as ufunc_find_matching_loop
from numba.np.ufunc.sigparse import parse_signature as parse_signature
from numba.np.ufunc.ufunc_base import UfuncBase as UfuncBase, UfuncLowererBase as UfuncLowererBase
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder as GUFuncBuilder

def make_gufunc_kernel(_dufunc): ...

class GUFuncLowerer(UfuncLowererBase):
    """Callable class responsible for lowering calls to a specific gufunc.
    """
    def __init__(self, gufunc) -> None: ...

class GUFunc(serialize.ReduceMixin, UfuncBase):
    """
    Dynamic generalized universal function (GUFunc)
    intended to act like a normal Numpy gufunc, but capable
    of call-time (just-in-time) compilation of fast loops
    specialized to inputs.
    """
    ufunc: Incomplete
    _frozen: bool
    _is_dynamic: Incomplete
    _identity: Incomplete
    gufunc_builder: Incomplete
    __name__: Incomplete
    __doc__: Incomplete
    _dispatcher: Incomplete
    def __init__(self, py_func, signature, identity: Incomplete | None = None, cache: Incomplete | None = None, is_dynamic: bool = False, targetoptions={}, writable_args=()) -> None: ...
    _lower_me: Incomplete
    def _initialize(self, dispatcher) -> None: ...
    def _reduce_states(self): ...
    @classmethod
    def _rebuild(cls, py_func, signature, identity, cache, is_dynamic, targetoptions, writable_args, typesigs, frozen): ...
    def __repr__(self) -> str: ...
    def _install_type(self, typingctx: Incomplete | None = None) -> None:
        """Constructs and installs a typing class for a gufunc object in the
        input typing context.  If no typing context is given, then
        _install_type() installs into the typing context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
    def add(self, fty) -> None: ...
    def build_ufunc(self): ...
    def expected_ndims(self): ...
    def _type_me(self, argtys, kws):
        """
        Implement AbstractTemplate.generic() for the typing class
        built by gufunc._install_type().

        Return the call-site signature after either validating the
        element-wise signature or compiling for it.
        """
    def _compile_for_argtys(self, argtys, return_type: Incomplete | None = None) -> None: ...
    def match_signature(self, ewise_types, sig): ...
    @property
    def is_dynamic(self): ...
    def _get_ewise_dtypes(self, args): ...
    def _num_args_match(self, *args): ...
    def _get_function_type(self, *args): ...
    def __call__(self, *args, **kwargs): ...

def _is_array_wrapper(obj):
    """Return True if obj wraps around numpy or another numpy-like library
    and is likely going to apply the ufunc to the wrapped array; False
    otherwise.

    At the moment, this returns True for

    - dask.array.Array
    - dask.dataframe.DataFrame
    - dask.dataframe.Series
    - xarray.DataArray
    - xarray.Dataset
    - xarray.Variable
    - pint.Quantity
    - other potential wrappers around dask array or dask dataframe

    We may need to add other libraries that pickle ufuncs from their
    __array_ufunc__ method in the future.

    Note that the below test is a lot more naive than
    `dask.base.is_dask_collection`
    (https://github.com/dask/dask/blob/5949e54bc04158d215814586a44d51e0eb4a964d/dask/base.py#L209-L249),  # noqa: E501
    because it doesn't need to find out if we're actually dealing with
    a dask collection, only that we're dealing with a wrapper.
    Namely, it will return True for a pint.Quantity wrapping around a plain float, a
    numpy.ndarray, or a dask.array.Array, and it's OK because in all cases
    Quantity.__array_ufunc__ is going to forward the ufunc call inwards.
    """
