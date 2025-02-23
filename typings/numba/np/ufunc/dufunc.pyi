from _typeshed import Incomplete
from numba import jit as jit, typeof as typeof
from numba.core import cgutils as cgutils, errors as errors, serialize as serialize, sigutils as sigutils, types as types
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.core.extending import intrinsic as intrinsic, is_jitted as is_jitted, overload_attribute as overload_attribute, overload_method as overload_method, register_jitable as register_jitable
from numba.core.typing import npydecl as npydecl
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, signature as signature
from numba.cpython.unsafe.tuple import tuple_setitem as tuple_setitem
from numba.np import numpy_support as numpy_support
from numba.np.ufunc import _internal as _internal, ufuncbuilder as ufuncbuilder
from numba.np.ufunc.ufunc_base import UfuncBase as UfuncBase, UfuncLowererBase as UfuncLowererBase
from numba.parfors import array_analysis as array_analysis

class UfuncAtIterator:
    ufunc: Incomplete
    a: Incomplete
    a_ty: Incomplete
    indices: Incomplete
    indices_ty: Incomplete
    b: Incomplete
    b_ty: Incomplete
    def __init__(self, ufunc, a, a_ty, indices, indices_ty, b: Incomplete | None = None, b_ty: Incomplete | None = None) -> None: ...
    def run(self, context, builder) -> None: ...
    def need_advanced_indexing(self): ...
    b_indice: Incomplete
    indexer: Incomplete
    cres: Incomplete
    def _prepare(self, context, builder) -> None: ...
    def _load_val(self, context, builder, loop_indices, array, array_ty): ...
    def _load_flat(self, context, builder, indices, array, array_ty): ...
    def _store_val(self, context, builder, array, array_ty, ptr, val) -> None: ...
    def _compile_ufunc(self, context, builder): ...
    def _call_ufunc(self, context, builder, loop_indices) -> None: ...

def make_dufunc_kernel(_dufunc): ...

class DUFuncLowerer(UfuncLowererBase):
    """Callable class responsible for lowering calls to a specific DUFunc.
    """
    def __init__(self, dufunc) -> None: ...

class DUFunc(serialize.ReduceMixin, _internal._DUFunc, UfuncBase):
    """
    Dynamic universal function (DUFunc) intended to act like a normal
    Numpy ufunc, but capable of call-time (just-in-time) compilation
    of fast loops specialized to inputs.
    """
    __base_kwargs: Incomplete
    def __init__(self, py_func, identity: Incomplete | None = None, cache: bool = False, targetoptions={}) -> None: ...
    reorderable: Incomplete
    __name__: Incomplete
    __doc__: Incomplete
    _lower_me: Incomplete
    def _initialize(self, dispatcher, identity) -> None: ...
    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
    @classmethod
    def _rebuild(cls, dispatcher, identity, frozen, siglist):
        """
        NOTE: part of ReduceMixin protocol
        """
    def build_ufunc(self):
        """
        For compatibility with the various *UFuncBuilder classes.
        """
    @property
    def targetoptions(self): ...
    @property
    def nin(self): ...
    @property
    def nout(self): ...
    @property
    def nargs(self): ...
    @property
    def ntypes(self): ...
    @property
    def types(self): ...
    @property
    def identity(self): ...
    @property
    def signature(self): ...
    _frozen: bool
    def disable_compile(self) -> None:
        """
        Disable the compilation of new signatures at call time.
        """
    def add(self, sig):
        """
        Compile the DUFunc for the given signature.
        """
    def __call__(self, *args, **kws):
        """
        Allow any argument that has overridden __array_ufunc__ (NEP-18)
        to take control of DUFunc.__call__.
        """
    def _compile_for_args(self, *args, **kws): ...
    def _compile_for_argtys(self, argtys, return_type: Incomplete | None = None):
        """
        Given a tuple of argument types (these should be the array
        dtypes, and not the array types themselves), compile the
        element-wise function for those inputs, generate a UFunc loop
        wrapper, and register the loop with the Numpy ufunc object for
        this DUFunc.
        """
    def match_signature(self, ewise_types, sig): ...
    def _install_ufunc_attributes(self, template) -> None: ...
    def _install_ufunc_methods(self, template) -> None: ...
    def _install_ufunc_at(self, template) -> None: ...
    def _install_ufunc_reduce(self, template) -> None: ...
    def at(self, a, indices, b: Incomplete | None = None): ...
    def _install_type(self, typingctx: Incomplete | None = None) -> None:
        """Constructs and installs a typing class for a DUFunc object in the
        input typing context.  If no typing context is given, then
        _install_type() installs into the typing context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
    def find_ewise_function(self, ewise_types):
        """
        Given a tuple of element-wise argument types, find a matching
        signature in the dispatcher.

        Return a 2-tuple containing the matching signature, and
        compilation result.  Will return two None's if no matching
        signature was found.
        """
    def _type_me(self, argtys, kwtys):
        """
        Implement AbstractTemplate.generic() for the typing class
        built by DUFunc._install_type().

        Return the call-site signature after either validating the
        element-wise signature or compiling for it.
        """
