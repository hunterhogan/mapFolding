from _typeshed import Incomplete
from numba.core import types as types
from numba.np import numpy_support as numpy_support

class UfuncLowererBase:
    """Callable class responsible for lowering calls to a specific gufunc.
    """
    ufunc: Incomplete
    make_ufunc_kernel_fn: Incomplete
    kernel: Incomplete
    libs: Incomplete
    def __init__(self, ufunc, make_kernel_fn, make_ufunc_kernel_fn) -> None: ...
    def __call__(self, context, builder, sig, args): ...

class UfuncBase:
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
    @property
    def accumulate(self): ...
    @property
    def at(self): ...
    @property
    def outer(self): ...
    @property
    def reduce(self): ...
    @property
    def reduceat(self): ...
    _frozen: bool
    def disable_compile(self) -> None:
        """
        Disable the compilation of new signatures at call time.
        """
    def _install_cg(self, targetctx: Incomplete | None = None) -> None:
        """
        Install an implementation function for a GUFunc/DUFunc object in the
        given target context.  If no target context is given, then
        _install_cg() installs into the target context of the
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
