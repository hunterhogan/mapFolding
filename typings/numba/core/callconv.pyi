from _typeshed import Incomplete
from numba.core import cgutils as cgutils, errors as errors, types as types
from numba.core.base import GENERIC_POINTER as GENERIC_POINTER, PYOBJECT as PYOBJECT
from typing import NamedTuple

class TryStatus(NamedTuple):
    in_try: Incomplete
    excinfo: Incomplete

class Status(NamedTuple):
    code: Incomplete
    is_ok: Incomplete
    is_none: Incomplete
    is_error: Incomplete
    is_stop_iteration: Incomplete
    is_python_exc: Incomplete
    is_user_exc: Incomplete
    excinfoptr: Incomplete

int32_t: Incomplete
int64_t: Incomplete
errcode_t = int32_t

def _const_int(code): ...

RETCODE_OK: Incomplete
RETCODE_EXC: Incomplete
RETCODE_NONE: Incomplete
RETCODE_STOPIT: Incomplete
FIRST_USEREXC: int
RETCODE_USEREXC: Incomplete

class BaseCallConv:
    context: Incomplete
    def __init__(self, context) -> None: ...
    def return_optional_value(self, builder, retty, valty, value) -> None: ...
    def return_native_none(self, builder) -> None: ...
    def return_exc(self, builder) -> None: ...
    def return_stop_iteration(self, builder) -> None: ...
    def get_return_type(self, ty):
        """
        Get the actual type of the return argument for Numba type *ty*.
        """
    def init_call_helper(self, builder):
        """
        Initialize and return a call helper object for the given builder.
        """
    def _get_call_helper(self, builder): ...
    def unpack_exception(self, builder, pyapi, status): ...
    def raise_error(self, builder, pyapi, status) -> None:
        """
        Given a non-ok *status*, raise the corresponding Python exception.
        """
    def decode_arguments(self, builder, argtypes, func):
        """
        Get the decoded (unpacked) Python arguments with *argtypes*
        from LLVM function *func*.  A tuple of LLVM values is returned.
        """
    def _get_arg_packer(self, argtypes):
        """
        Get an argument packer for the given argument types.
        """

class MinimalCallConv(BaseCallConv):
    """
    A minimal calling convention, suitable for e.g. GPU targets.
    The implemented function signature is:

        retcode_t (<Python return type>*, ... <Python arguments>)

    The return code will be one of the RETCODE_* constants or a
    function-specific user exception id (>= RETCODE_USEREXC).

    Caller is responsible for allocating a slot for the return value
    (passed as a pointer in the first argument).
    """
    def _make_call_helper(self, builder): ...
    def return_value(self, builder, retval) -> None: ...
    def return_user_exc(self, builder, exc, exc_args: Incomplete | None = None, loc: Incomplete | None = None, func_name: Incomplete | None = None) -> None: ...
    def return_status_propagate(self, builder, status) -> None: ...
    def _return_errcode_raw(self, builder, code) -> None: ...
    def _get_return_status(self, builder, code):
        """
        Given a return *code*, get a Status instance.
        """
    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
        """
    def decorate_function(self, fn, args, fe_argtypes, noalias: bool = False) -> None:
        """
        Set names and attributes of function arguments.
        """
    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """

class _MinimalCallHelper:
    '''
    A call helper object for the "minimal" calling convention.
    User exceptions are represented as integer codes and stored in
    a mapping for retrieval from the caller.
    '''
    exceptions: Incomplete
    def __init__(self) -> None: ...
    def _add_exception(self, exc, exc_args, locinfo):
        """
        Add a new user exception to this helper. Returns an integer that can be
        used to refer to the added exception in future.

        Parameters
        ----------
        exc :
            exception type
        exc_args : None or tuple
            exception args
        locinfo : tuple
            location information
        """
    def get_exception(self, exc_id):
        """
        Get information about a user exception. Returns a tuple of
        (exception type, exception args, location information).

        Parameters
        ----------
        id : integer
            The ID of the exception to look up
        """

PICKLE_BUF_IDX: int
PICKLE_BUFSZ_IDX: int
HASH_BUF_IDX: int
UNWRAP_FUNC_IDX: int
ALLOC_FLAG_IDX: int
excinfo_t: Incomplete
excinfo_ptr_t: Incomplete

class CPUCallConv(BaseCallConv):
    """
    The calling convention for CPU targets.
    The implemented function signature is:

        retcode_t (<Python return type>*, excinfo **, ... <Python arguments>)

    The return code will be one of the RETCODE_* constants.
    If RETCODE_USEREXC, the exception info pointer will be filled with
    a pointer to a constant struct describing the raised exception.

    Caller is responsible for allocating slots for the return value
    and the exception info pointer (passed as first and second arguments,
    respectively).
    """
    _status_ids: Incomplete
    def _make_call_helper(self, builder) -> None: ...
    def return_value(self, builder, retval) -> None: ...
    def build_excinfo_struct(self, exc, exc_args, loc, func_name): ...
    def set_static_user_exc(self, builder, exc, exc_args: Incomplete | None = None, loc: Incomplete | None = None, func_name: Incomplete | None = None) -> None: ...
    def return_user_exc(self, builder, exc, exc_args: Incomplete | None = None, loc: Incomplete | None = None, func_name: Incomplete | None = None) -> None: ...
    def unpack_dynamic_exception(self, builder, pyapi, status): ...
    def unpack_exception(self, builder, pyapi, status): ...
    def emit_unwrap_dynamic_exception_fn(self, module, st_type, nb_types): ...
    def emit_wrap_args_insts(self, builder, pyapi, struct_type, exc_args):
        """
        Create an anonymous struct containing the given LLVM *values*.
        """
    def set_dynamic_user_exc(self, builder, exc, exc_args, nb_types, loc: Incomplete | None = None, func_name: Incomplete | None = None) -> None:
        """
        Compute the required bits to emit an exception with dynamic (runtime)
        values
        """
    def return_dynamic_user_exc(self, builder, exc, exc_args, nb_types, loc: Incomplete | None = None, func_name: Incomplete | None = None) -> None:
        """
        Same as ::return_user_exc but for dynamic exceptions
        """
    def _get_try_state(self, builder): ...
    def check_try_status(self, builder): ...
    def set_try_status(self, builder) -> None: ...
    def unset_try_status(self, builder) -> None: ...
    def return_status_propagate(self, builder, status) -> None: ...
    def _return_errcode_raw(self, builder, code) -> None: ...
    def _get_return_status(self, builder, code, excinfoptr):
        """
        Given a return *code* and *excinfoptr*, get a Status instance.
        """
    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
        """
    def decorate_function(self, fn, args, fe_argtypes, noalias: bool = False):
        """
        Set names of function arguments, and add useful attributes to them.
        """
    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
    def _get_return_argument(self, func): ...
    def _get_excinfo_argument(self, func): ...
    def call_function(self, builder, callee, resty, argtys, args, attrs: Incomplete | None = None):
        '''
        Call the Numba-compiled *callee*.
        Parameters:
        -----------
        attrs: LLVM style string or iterable of individual attributes, default
               is None which specifies no attributes. Examples:
               LLVM style string: "noinline fast"
               Equivalent iterable: ("noinline", "fast")
        '''

class ErrorModel:
    call_conv: Incomplete
    def __init__(self, call_conv) -> None: ...
    def fp_zero_division(self, builder, exc_args: Incomplete | None = None, loc: Incomplete | None = None): ...

class PythonErrorModel(ErrorModel):
    """
    The Python error model.  Any invalid FP input raises an exception.
    """
    raise_on_fp_zero_division: bool

class NumpyErrorModel(ErrorModel):
    """
    In the Numpy error model, floating-point errors don't raise an
    exception.  The FPU exception state is inspected by Numpy at the
    end of a ufunc's execution and a warning is raised if appropriate.

    Note there's no easy way to set the FPU exception state from LLVM.
    Instructions known to set an FP exception can be optimized away:
        https://llvm.org/bugs/show_bug.cgi?id=6050
        http://lists.llvm.org/pipermail/llvm-dev/2014-September/076918.html
        http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140929/237997.html
    """
    raise_on_fp_zero_division: bool

error_models: Incomplete

def create_error_model(model_name, context):
    """
    Create an error model instance for the given target context.
    """
