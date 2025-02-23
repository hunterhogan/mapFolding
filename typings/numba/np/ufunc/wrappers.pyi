from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types
from numba.core.caching import NullCache as NullCache, make_library_cache as make_library_cache
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from typing import NamedTuple

class _wrapper_info(NamedTuple):
    library: Incomplete
    env: Incomplete
    name: Incomplete

def _build_ufunc_loop_body(load, store, context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, env): ...
def _build_ufunc_loop_body_objmode(load, store, context, func, builder, arrays, out, offsets, store_offset, signature, env, pyapi): ...
def build_slow_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, env): ...
def build_obj_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, envptr, env): ...
def build_fast_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, ind, pyapi, env): ...
def build_ufunc_wrapper(library, context, fname, signature, objmode, cres):
    """
    Wrap the scalar function with a loop that iterates over the arguments

    Returns
    -------
    (library, env, name)
    """

class UArrayArg:
    context: Incomplete
    builder: Incomplete
    fe_type: Incomplete
    dataptr: Incomplete
    abisize: Incomplete
    step: Incomplete
    is_unit_strided: Incomplete
    def __init__(self, context, builder, args, steps, i, fe_type) -> None: ...
    def load_direct(self, byteoffset):
        """
        Generic load from the given *byteoffset*.  load_aligned() is
        preferred if possible.
        """
    def load_aligned(self, ind): ...
    def store_direct(self, value, byteoffset) -> None: ...
    def store_aligned(self, value, ind) -> None: ...

GufWrapperCache: Incomplete

class _GufuncWrapper:
    py_func: Incomplete
    cres: Incomplete
    sin: Incomplete
    sout: Incomplete
    is_objectmode: Incomplete
    cache: Incomplete
    is_parfors: Incomplete
    def __init__(self, py_func, cres, sin, sout, cache, is_parfors) -> None:
        """
        The *is_parfors* argument is a boolean that indicates if the GUfunc
        being built is to be used as a ParFors kernel. If True, it disables
        the caching on the wrapper as a separate unit because it will be linked
        into the caller function and cached along with it.
        """
    @property
    def library(self): ...
    @property
    def context(self): ...
    @property
    def call_conv(self): ...
    @property
    def signature(self): ...
    @property
    def fndesc(self): ...
    @property
    def env(self): ...
    def _wrapper_function_type(self): ...
    def _build_wrapper(self, library, name) -> None:
        """
        The LLVM IRBuilder code to create the gufunc wrapper.
        The *library* arg is the CodeLibrary to which the wrapper should
        be added.  The *name* arg is the name of the wrapper function being
        created.
        """
    def _compile_wrapper(self, wrapper_name): ...
    def build(self): ...
    def gen_loop_body(self, builder, pyapi, func, args): ...
    def gen_prologue(self, builder, pyapi) -> None: ...
    def gen_epilogue(self, builder, pyapi) -> None: ...

class _GufuncObjectWrapper(_GufuncWrapper):
    def gen_loop_body(self, builder, pyapi, func, args): ...
    gil: Incomplete
    def gen_prologue(self, builder, pyapi) -> None: ...
    def gen_epilogue(self, builder, pyapi) -> None: ...

def build_gufunc_wrapper(py_func, cres, sin, sout, cache, is_parfors): ...
def _prepare_call_to_object_mode(context, builder, pyapi, func, signature, args): ...

class GUArrayArg:
    context: Incomplete
    builder: Incomplete
    data: Incomplete
    _loader: Incomplete
    def __init__(self, context, builder, args, steps, i, step_offset, typ, syms, sym_dim) -> None: ...
    def get_array_at_offset(self, ind): ...

class _ScalarArgLoader:
    """
    Handle GFunc argument loading where a scalar type is used in the core
    function.
    Note: It still has a stride because the input to the gufunc can be an array
          for this argument.
    """
    dtype: Incomplete
    stride: Incomplete
    def __init__(self, dtype, stride) -> None: ...
    def load(self, context, builder, data, ind): ...

class _ArrayArgLoader:
    """
    Handle GUFunc argument loading where an array is expected.
    """
    dtype: Incomplete
    ndim: Incomplete
    core_step: Incomplete
    as_scalar: Incomplete
    shape: Incomplete
    strides: Incomplete
    def __init__(self, dtype, ndim, core_step, as_scalar, shape, strides) -> None: ...
    def load(self, context, builder, data, ind): ...
    def _shape_and_strides(self, context, builder): ...

class _ArrayAsScalarArgLoader(_ArrayArgLoader):
    '''
    Handle GUFunc argument loading where the shape signature specifies
    a scalar "()" but a 1D array is used for the type of the core function.
    '''
    def _shape_and_strides(self, context, builder): ...
