from _typeshed import Incomplete
from numba import _dynfunc as _dynfunc
from numba.core import callconv as callconv, cgutils as cgutils, codegen as codegen, config as config, externals as externals, fastmathpass as fastmathpass, intrinsics as intrinsics, types as types, utils as utils
from numba.core.base import BaseContext as BaseContext
from numba.core.callwrapper import PyCallWrapper as PyCallWrapper
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.core.cpu_options import FastMathOptions as FastMathOptions, InlineOptions as InlineOptions, ParallelOptions as ParallelOptions
from numba.core.options import TargetOptions as TargetOptions, include_default_options as include_default_options
from numba.core.runtime import rtsys as rtsys
from numba.np import ufunc_db as ufunc_db

class ClosureBody(cgutils.Structure):
    _fields: Incomplete

class EnvBody(cgutils.Structure):
    _fields: Incomplete

class CPUContext(BaseContext):
    """
    Changes BaseContext calling convention
    """
    allow_dynamic_globals: bool
    def __init__(self, typingctx, target: str = 'cpu') -> None: ...
    def create_module(self, name): ...
    is32bit: Incomplete
    _internal_codegen: Incomplete
    def init(self) -> None: ...
    def load_additional_registries(self) -> None: ...
    @property
    def target_data(self): ...
    def with_aot_codegen(self, name, **aot_options): ...
    def codegen(self): ...
    @property
    def call_conv(self): ...
    def get_env_body(self, builder, envptr):
        """
        From the given *envptr* (a pointer to a _dynfunc.Environment object),
        get a EnvBody allowing structured access to environment fields.
        """
    def get_env_manager(self, builder, return_pyobject: bool = False): ...
    def get_generator_state(self, builder, genptr, return_type):
        """
        From the given *genptr* (a pointer to a _dynfunc.Generator object),
        get a pointer to its state area.
        """
    def build_list(self, builder, list_type, items):
        """
        Build a list from the Numba *list_type* and its initial *items*.
        """
    def build_set(self, builder, set_type, items):
        """
        Build a set from the Numba *set_type* and its initial *items*.
        """
    def build_map(self, builder, dict_type, item_types, items): ...
    def post_lowering(self, mod, library) -> None: ...
    def create_cpython_wrapper(self, library, fndesc, env, call_helper, release_gil: bool = False) -> None: ...
    def create_cfunc_wrapper(self, library, fndesc, env, call_helper) -> None: ...
    def get_executable(self, library, fndesc, env):
        """
        Returns
        -------
        (cfunc, fnptr)

        - cfunc
            callable function (Can be None)
        - fnptr
            callable function address
        - env
            an execution environment (from _dynfunc)
        """
    def calc_array_sizeof(self, ndim):
        """
        Calculate the size of an array struct on the CPU target
        """
    def get_ufunc_info(self, ufunc_key): ...

_options_mixin: Incomplete

class CPUTargetOptions(_options_mixin, TargetOptions):
    def finalize(self, flags, options) -> None: ...
