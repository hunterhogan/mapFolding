from _typeshed import Incomplete
from numba.core import cgutils as cgutils, config as config, types as types

class _ArgManager:
    """
    A utility class to handle argument unboxing and cleanup
    """
    context: Incomplete
    builder: Incomplete
    api: Incomplete
    env_manager: Incomplete
    arg_count: int
    cleanups: Incomplete
    nextblk: Incomplete
    def __init__(self, context, builder, api, env_manager, endblk, nargs) -> None: ...
    def add_arg(self, obj, ty):
        """
        Unbox argument and emit code that handles any error during unboxing.
        Args are cleaned up in reverse order of the parameter list, and
        cleanup begins as soon as unboxing of any argument fails. E.g. failure
        on arg2 will result in control flow going through:

            arg2.err -> arg1.err -> arg0.err -> arg.end (returns)
        """
    def emit_cleanup(self) -> None:
        """
        Emit the cleanup code after returning from the wrapped function.
        """

class _GilManager:
    """
    A utility class to handle releasing the GIL and then re-acquiring it
    again.
    """
    builder: Incomplete
    api: Incomplete
    argman: Incomplete
    thread_state: Incomplete
    def __init__(self, builder, api, argman) -> None: ...
    def emit_cleanup(self) -> None: ...

class PyCallWrapper:
    context: Incomplete
    module: Incomplete
    func: Incomplete
    fndesc: Incomplete
    env: Incomplete
    release_gil: Incomplete
    def __init__(self, context, module, func, fndesc, env, call_helper, release_gil) -> None: ...
    def build(self): ...
    def build_wrapper(self, api, builder, closure, args, kws) -> None: ...
    def get_env(self, api, builder):
        """Get the Environment object which is declared as a global
        in the module of the wrapped function.
        """
    def _simplified_return_type(self):
        """
        The NPM callconv has already converted simplified optional types.
        We can simply use the value type from it.
        """
    def debug_print(self, builder, msg) -> None: ...
