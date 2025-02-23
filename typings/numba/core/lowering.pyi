from _typeshed import Incomplete
from numba.core import cgutils as cgutils, config as config, debuginfo as debuginfo, funcdesc as funcdesc, generators as generators, ir as ir, ir_utils as ir_utils, removerefctpass as removerefctpass, targetconfig as targetconfig, types as types, typing as typing, utils as utils
from numba.core.analysis import compute_use_defs as compute_use_defs, must_use_alloca as must_use_alloca
from numba.core.environment import Environment as Environment
from numba.core.errors import LiteralTypingError as LiteralTypingError, LoweringError as LoweringError, NumbaDebugInfoWarning as NumbaDebugInfoWarning, TypingError as TypingError, UnsupportedError as UnsupportedError, new_error_context as new_error_context
from numba.core.funcdesc import default_mangler as default_mangler
from numba.misc.coverage_support import get_registered_loc_notify as get_registered_loc_notify
from numba.misc.firstlinefinder import get_func_body_first_lineno as get_func_body_first_lineno
from typing import NamedTuple

class _VarArgItem(NamedTuple):
    vararg: Incomplete
    index: Incomplete

class BaseLower:
    """
    Lower IR to LLVM
    """
    library: Incomplete
    fndesc: Incomplete
    blocks: Incomplete
    func_ir: Incomplete
    generator_info: Incomplete
    metadata: Incomplete
    flags: Incomplete
    module: Incomplete
    env: Incomplete
    blkmap: Incomplete
    pending_phis: Incomplete
    varmap: Incomplete
    firstblk: Incomplete
    loc: int
    context: Incomplete
    defn_loc: Incomplete
    debuginfo: Incomplete
    _loc_notify_registry: Incomplete
    def __init__(self, context, library, fndesc, func_ir, metadata: Incomplete | None = None) -> None: ...
    @property
    def call_conv(self): ...
    def init(self) -> None: ...
    pyapi: Incomplete
    env_manager: Incomplete
    env_body: Incomplete
    envarg: Incomplete
    def init_pyapi(self) -> None:
        """
        Init the Python API and Environment Manager for the function being
        lowered.
        """
    def _compute_def_location(self): ...
    def pre_lower(self) -> None:
        """
        Called before lowering all blocks.
        """
    def post_lower(self) -> None:
        """
        Called after all blocks are lowered
        """
    def pre_block(self, block) -> None:
        """
        Called before lowering a block.
        """
    def post_block(self, block) -> None:
        """
        Called after lowering a block.
        """
    def return_dynamic_exception(self, exc_class, exc_args, nb_types, loc: Incomplete | None = None) -> None: ...
    def return_exception(self, exc_class, exc_args: Incomplete | None = None, loc: Incomplete | None = None) -> None:
        """Propagate exception to the caller.
        """
    def set_exception(self, exc_class, exc_args: Incomplete | None = None, loc: Incomplete | None = None) -> None:
        """Set exception state in the current function.
        """
    def emit_environment_object(self) -> None:
        """Emit a pointer to hold the Environment object.
        """
    genlower: Incomplete
    gentype: Incomplete
    def lower(self) -> None: ...
    fnargs: Incomplete
    def extract_function_arguments(self): ...
    def lower_normal_function(self, fndesc) -> None:
        """
        Lower non-generator *fndesc*.
        """
    def lower_function_body(self):
        """
        Lower the current function's body, and return the entry block.
        """
    def lower_block(self, block) -> None:
        """
        Lower the given block.
        """
    def create_cpython_wrapper(self, release_gil: bool = False) -> None:
        """
        Create CPython wrapper(s) around this function (or generator).
        """
    def create_cfunc_wrapper(self) -> None:
        """
        Create C wrapper around this function.
        """
    function: Incomplete
    entry_block: Incomplete
    builder: Incomplete
    call_helper: Incomplete
    def setup_function(self, fndesc) -> None: ...
    def typeof(self, varname): ...
    def notify_loc(self, loc: ir.Loc) -> None:
        """Called when a new instruction with the given `loc` is about to be
        lowered.
        """
    def debug_print(self, msg) -> None: ...
    def print_variable(self, msg, varname) -> None:
        """Helper to emit ``print(msg, varname)`` for debugging.

        Parameters
        ----------
        msg : str
            Literal string to be printed.
        varname : str
            A variable name whose value will be printed.
        """

class Lower(BaseLower):
    GeneratorLower: Incomplete
    def init(self) -> None: ...
    @property
    def _disable_sroa_like_opt(self):
        """Flags that the SROA like optimisation that Numba performs (which
        prevent alloca and subsequent load/store for locals) should be disabled.
        Currently, this is conditional solely on the presence of a request for
        the emission of debug information."""
    _singly_assigned_vars: Incomplete
    _blk_local_varmap: Incomplete
    def _find_singly_assigned_variable(self) -> None: ...
    _cur_ir_block: Incomplete
    def pre_block(self, block) -> None: ...
    def post_block(self, block) -> None: ...
    def lower_inst(self, inst): ...
    def lower_setitem(self, target_var, index_var, value_var, signature): ...
    def lower_try_dynamic_raise(self, inst) -> None: ...
    def lower_dynamic_raise(self, inst) -> None: ...
    def lower_static_raise(self, inst) -> None: ...
    def lower_static_try_raise(self, inst) -> None: ...
    def lower_assign(self, ty, inst): ...
    def lower_yield(self, retty, inst): ...
    def lower_binop(self, resty, expr, op): ...
    def lower_getitem(self, resty, expr, value, index, signature): ...
    def _cast_var(self, var, ty):
        """
        Cast a Numba IR variable to the given Numba type, returning a
        low-level value.
        """
    def fold_call_args(self, fnty, signature, pos_args, vararg, kw_args): ...
    def lower_print(self, inst) -> None:
        """
        Lower a ir.Print()
        """
    def lower_call(self, resty, expr): ...
    def _lower_call_ObjModeDispatcher(self, fnty, expr, signature): ...
    def _lower_call_ExternalFunction(self, fnty, expr, signature): ...
    def _lower_call_ExternalFunctionPointer(self, fnty, expr, signature): ...
    def _lower_call_RecursiveCall(self, fnty, expr, signature): ...
    def _lower_call_FunctionType(self, fnty, expr, signature): ...
    def __call_first_class_function_pointer(self, ftype, fname, sig, argvals):
        """
        Calls a first-class function pointer.

        This function is responsible for calling a first-class function pointer,
        which can either be a JIT-compiled function or a Python function. It
        determines if a JIT address is available, and if so, calls the function
        using the JIT address. Otherwise, it calls the function using a function
        pointer obtained from the `__get_first_class_function_pointer` method.

        Args:
            ftype: The type of the function.
            fname: The name of the function.
            sig: The signature of the function.
            argvals: The argument values to pass to the function.

        Returns:
            The result of calling the function.
        """
    def __get_first_class_function_pointer(self, ftype, fname, sig): ...
    def _lower_call_normal(self, fnty, expr, signature): ...
    def lower_expr(self, resty, expr): ...
    def _alloca_var(self, name, fetype) -> None:
        """
        Ensure the given variable has an allocated stack slot (if needed).
        """
    def getvar(self, name):
        """
        Get a pointer to the given variable's slot.
        """
    def loadvar(self, name):
        """
        Load the given variable's value.
        """
    def storevar(self, value, name, argidx: Incomplete | None = None) -> None:
        """
        Store the value into the given variable.
        """
    def delvar(self, name) -> None:
        """
        Delete the given variable.
        """
    def alloca(self, name, type): ...
    def alloca_lltype(self, name, lltype, datamodel: Incomplete | None = None): ...
    def incref(self, typ, val) -> None: ...
    def decref(self, typ, val) -> None: ...

def _lit_or_omitted(value):
    """Returns a Literal instance if the type of value is supported;
    otherwise, return `Omitted(value)`.
    """
