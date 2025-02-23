from _typeshed import Incomplete
from numba.core import errors as errors, ir as ir, ir_utils as ir_utils, sigutils as sigutils, types as types
from numba.core.ir_utils import build_definitions as build_definitions
from numba.core.transforms import find_region_inout_vars as find_region_inout_vars
from numba.core.typing.typeof import typeof_impl as typeof_impl

class WithContext:
    """A dummy object for use as contextmanager.
    This can be used as a contextmanager.
    """
    is_callable: bool
    def __enter__(self) -> None: ...
    def __exit__(self, typ: type[BaseException] | None, val: BaseException | None, tb: types.TracebackType | None) -> None: ...
    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra) -> None:
        """Mutate the *blocks* to implement this contextmanager.

        Parameters
        ----------
        func_ir : FunctionIR
        blocks : dict[ir.Block]
        blk_start, blk_end : int
            labels of the starting and ending block of the context-manager.
        body_block: sequence[int]
            A sequence of int's representing labels of the with-body
        dispatcher_factory : callable
            A callable that takes a `FunctionIR` and returns a `Dispatcher`.
        """

def typeof_contextmanager(val, c): ...
def _get_var_parent(name):
    """Get parent of the variable given its name
    """
def _clear_blocks(blocks, to_clear) -> None:
    """Remove keys in *to_clear* from *blocks*.
    """

class _ByPassContextType(WithContext):
    """A simple context-manager that tells the compiler to bypass the body
    of the with-block.
    """
    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra) -> None: ...

bypass_context: Incomplete

class _CallContextType(WithContext):
    """A simple context-manager that tells the compiler to lift the body of the
    with-block as another function.
    """
    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra): ...

call_context: Incomplete

class _ObjModeContextType(WithContext):
    '''Creates a contextmanager to be used inside jitted functions to enter
    *object-mode* for using interpreter features.  The body of the with-context
    is lifted into a function that is compiled in *object-mode*.  This
    transformation process is limited and cannot process all possible
    Python code.  However, users can wrap complicated logic in another
    Python function, which will then be executed by the interpreter.

    Use this as a function that takes keyword arguments only.
    The argument names must correspond to the output variables from the
    with-block.  Their respective values can be:

    1. strings representing the expected types; i.e. ``"float32"``.
    2. compile-time bound global or nonlocal variables referring to the
       expected type. The variables are read at compile time.

    When exiting the with-context, the output variables are converted
    to the expected nopython types according to the annotation.  This process
    is the same as passing Python objects into arguments of a nopython
    function.

    Example::

        import numpy as np
        from numba import njit, objmode, types

        def bar(x):
            # This code is executed by the interpreter.
            return np.asarray(list(reversed(x.tolist())))

        # Output type as global variable
        out_ty = types.intp[:]

        @njit
        def foo():
            x = np.arange(5)
            y = np.zeros_like(x)
            with objmode(y=\'intp[:]\', z=out_ty):  # annotate return type
                # this region is executed by object-mode.
                y += bar(x)
                z = y
            return y, z

    .. note:: Known limitations:

        - with-block cannot use incoming list objects.
        - with-block cannot use incoming function objects.
        - with-block cannot ``yield``, ``break``, ``return`` or ``raise``           such that the execution will leave the with-block immediately.
        - with-block cannot contain `with` statements.
        - random number generator states do not synchronize; i.e.           nopython-mode and object-mode uses different RNG states.

    .. note:: When used outside of no-python mode, the context-manager has no
        effect.

    .. warning:: This feature is experimental.  The supported features may
        change with or without notice.

    '''
    is_callable: bool
    def _legalize_args(self, func_ir, args, kwargs, loc, func_globals, func_closures):
        """
        Legalize arguments to the context-manager

        Parameters
        ----------
        func_ir: FunctionIR
        args: tuple
            Positional arguments to the with-context call as IR nodes.
        kwargs: dict
            Keyword arguments to the with-context call as IR nodes.
        loc: numba.core.ir.Loc
            Source location of the with-context call.
        func_globals: dict
            The globals dictionary of the calling function.
        func_closures: dict
            The resolved closure variables of the calling function.
        """
    def _legalize_arg_type(self, name, typ, loc) -> None:
        """Legalize the argument type

        Parameters
        ----------
        name: str
            argument name.
        typ: numba.core.types.Type
            argument type.
        loc: numba.core.ir.Loc
            source location for error reporting.
        """
    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra): ...
    def __call__(self, *args, **kwargs): ...

objmode_context: Incomplete

def _bypass_with_context(blocks, blk_start, blk_end, forwardvars) -> None:
    """Given the starting and ending block of the with-context,
    replaces the head block with a new block that jumps to the end.

    *blocks* is modified inplace.
    """
def _mutate_with_block_caller(dispatcher, blocks, blk_start, blk_end, inputs, outputs):
    """Make a new block that calls into the lifeted with-context.

    Parameters
    ----------
    dispatcher : Dispatcher
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-manager.
    inputs: sequence[str]
        Input variable names
    outputs: sequence[str]
        Output variable names
    """
def _mutate_with_block_callee(blocks, blk_start, blk_end, inputs, outputs) -> None:
    """Mutate *blocks* for the callee of a with-context.

    Parameters
    ----------
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-manager.
    inputs: sequence[str]
        Input variable names
    outputs: sequence[str]
        Output variable names
    """

class _ParallelChunksize(WithContext):
    is_callable: bool
    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra) -> None: ...
    chunksize: Incomplete
    def __call__(self, *args, **kwargs):
        """Act like a function and enforce the contract that
        setting the chunksize takes only one integer input.
        """
    orig_chunksize: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, typ: type[BaseException] | None, val: BaseException | None, tb: types.TracebackType | None) -> None: ...

parallel_chunksize: Incomplete
