from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import compiler as compiler, config as config, serialize as serialize, sigutils as sigutils, targetconfig as targetconfig, types as types, utils as utils
from numba.core.caching import FunctionCache as FunctionCache, NullCache as NullCache
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.core.decorators import jit as jit
from numba.core.descriptors import TargetDescriptor as TargetDescriptor
from numba.core.errors import NumbaDeprecationWarning as NumbaDeprecationWarning
from numba.core.extending import is_jitted as is_jitted
from numba.core.options import TargetOptions as TargetOptions, include_default_options as include_default_options
from numba.core.registry import cpu_target as cpu_target
from numba.core.target_extension import dispatcher_registry as dispatcher_registry, target_registry as target_registry
from numba.np.numpy_support import as_dtype as as_dtype
from numba.np.ufunc import _internal as _internal
from numba.np.ufunc.sigparse import parse_signature as parse_signature
from numba.np.ufunc.wrappers import build_gufunc_wrapper as build_gufunc_wrapper, build_ufunc_wrapper as build_ufunc_wrapper

_options_mixin: Incomplete

class UFuncTargetOptions(_options_mixin, TargetOptions):
    def finalize(self, flags, options) -> None: ...

class UFuncTarget(TargetDescriptor):
    options = UFuncTargetOptions
    def __init__(self) -> None: ...
    @property
    def typing_context(self): ...
    @property
    def target_context(self): ...

ufunc_target: Incomplete

class UFuncDispatcher(serialize.ReduceMixin):
    """
    An object handling compilation of various signatures for a ufunc.
    """
    targetdescr = ufunc_target
    py_func: Incomplete
    overloads: Incomplete
    targetoptions: Incomplete
    locals: Incomplete
    cache: Incomplete
    def __init__(self, py_func, locals={}, targetoptions={}) -> None: ...
    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
    @classmethod
    def _rebuild(cls, pyfunc, locals, targetoptions):
        """
        NOTE: part of ReduceMixin protocol
        """
    def enable_caching(self) -> None: ...
    def compile(self, sig, locals={}, **targetoptions): ...
    def _compile_core(self, sig, flags, locals):
        """
        Trigger the compiler on the core function or load a previously
        compiled version from the cache.  Returns the CompileResult.
        """

def _compile_element_wise_function(nb_func, targetoptions, sig): ...
def _finalize_ufunc_signature(cres, args, return_type):
    """Given a compilation result, argument types, and a return type,
    build a valid Numba signature after validating that it doesn't
    violate the constraints for the compilation mode.
    """
def _build_element_wise_ufunc_wrapper(cres, signature):
    """Build a wrapper for the ufunc loop entry point given by the
    compilation result object, using the element-wise signature.
    """

_identities: Incomplete

def parse_identity(identity):
    """
    Parse an identity value and return the corresponding low-level value
    for Numpy.
    """
def _suppress_deprecation_warning_nopython_not_supplied() -> Generator[None]:
    '''This suppresses the NumbaDeprecationWarning that occurs through the use
    of `jit` without the `nopython` kwarg. This use of `jit` occurs in a few
    places in the `{g,}ufunc` mechanism in Numba, predominantly to wrap the
    "kernel" function.'''

class _BaseUFuncBuilder:
    def add(self, sig: Incomplete | None = None): ...
    def disable_compile(self) -> None:
        """
        Disable the compilation of new signatures at call time.
        """

class UFuncBuilder(_BaseUFuncBuilder):
    py_func: Incomplete
    identity: Incomplete
    nb_func: Incomplete
    _sigs: Incomplete
    _cres: Incomplete
    def __init__(self, py_func, identity: Incomplete | None = None, cache: bool = False, targetoptions={}) -> None: ...
    def _finalize_signature(self, cres, args, return_type):
        """Slated for deprecation, use ufuncbuilder._finalize_ufunc_signature()
        instead.
        """
    def build_ufunc(self): ...
    def build(self, cres, signature):
        """Slated for deprecation, use
        ufuncbuilder._build_element_wise_ufunc_wrapper().
        """

class GUFuncBuilder(_BaseUFuncBuilder):
    py_func: Incomplete
    identity: Incomplete
    nb_func: Incomplete
    signature: Incomplete
    targetoptions: Incomplete
    cache: Incomplete
    _sigs: Incomplete
    _cres: Incomplete
    writable_args: Incomplete
    def __init__(self, py_func, signature, identity: Incomplete | None = None, cache: bool = False, targetoptions={}, writable_args=()) -> None: ...
    def _finalize_signature(self, cres, args, return_type): ...
    def build_ufunc(self): ...
    def build(self, cres):
        """
        Returns (dtype numbers, function ptr, EnvironmentObject)
        """

def _get_transform_arg(py_func):
    """Return function that transform arg into index"""
