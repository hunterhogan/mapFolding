from _typeshed import Incomplete
from numba.core import config as config, extending as extending, registry as registry, sigutils as sigutils
from numba.core.errors import DeprecationError as DeprecationError, NumbaDeprecationWarning as NumbaDeprecationWarning
from numba.stencils.stencil import stencil as stencil
from typing import Any as Any
from collections.abc import Callable as Callable

_logger: Incomplete
_msg_deprecated_signature_arg: str

def jit(signature_or_function: Incomplete | None = None, locals: dict[str, Any]={}, cache: bool = False, pipeline_class: Incomplete | None = None, boundscheck: Incomplete | None = None, **options: bool | str | Callable[..., Any] | None) -> Callable[..., Any]:
    '''
    This decorator is used to compile a Python function into native code.

    Args
    -----
    signature_or_function:
        The (optional) signature or list of signatures to be compiled.
        If not passed, required signatures will be compiled when the
        decorated function is called, depending on the argument values.
        As a convenience, you can directly pass the function to be compiled
        instead.

    locals: dict
        Mapping of local variable names to Numba types. Used to override the
        types deduced by Numba\'s type inference engine.

    pipeline_class: type numba.compiler.CompilerBase
            The compiler pipeline type for customizing the compilation stages.

    options:
        For a cpu target, valid options are:
            nopython: bool
                Set to True to disable the use of PyObjects and Python API
                calls. The default behavior is to allow the use of PyObjects
                and Python API. Default value is True.

            forceobj: bool
                Set to True to force the use of PyObjects for every value.
                Default value is False.

            looplift: bool
                Set to True to enable jitting loops in nopython mode while
                leaving surrounding code in object mode. This allows functions
                to allocate NumPy arrays and use Python objects, while the
                tight loops in the function can still be compiled in nopython
                mode. Any arrays that the tight loop uses should be created
                before the loop is entered. Default value is True.

            error_model: str
                The error-model affects divide-by-zero behavior.
                Valid values are \'python\' and \'numpy\'. The \'python\' model
                raises exception.  The \'numpy\' model sets the result to
                *+/-inf* or *nan*. Default value is \'python\'.

            inline: str or callable
                The inline option will determine whether a function is inlined
                at into its caller if called. String options are \'never\'
                (default) which will never inline, and \'always\', which will
                always inline. If a callable is provided it will be called with
                the call expression node that is requesting inlining, the
                caller\'s IR and callee\'s IR as arguments, it is expected to
                return Truthy as to whether to inline.
                NOTE: This inlining is performed at the Numba IR level and is in
                no way related to LLVM inlining.

            boundscheck: bool or None
                Set to True to enable bounds checking for array indices. Out
                of bounds accesses will raise IndexError. The default is to
                not do bounds checking. If False, bounds checking is disabled,
                out of bounds accesses can produce garbage results or segfaults.
                However, enabling bounds checking will slow down typical
                functions, so it is recommended to only use this flag for
                debugging. You can also set the NUMBA_BOUNDSCHECK environment
                variable to 0 or 1 to globally override this flag. The default
                value is None, which under normal execution equates to False,
                but if debug is set to True then bounds checking will be
                enabled.

    Returns
    --------
    A callable usable as a compiled function.  Actual compiling will be
    done lazily if no explicit signatures are passed.

    Examples
    --------
    The function can be used in the following ways:

    1) jit(signatures, **targetoptions) -> jit(function)

        Equivalent to:

            d = dispatcher(function, targetoptions)
            for signature in signatures:
                d.compile(signature)

        Create a dispatcher object for a python function.  Then, compile
        the function with the given signature(s).

        Example:

            @jit("int32(int32, int32)")
            def foo(x, y):
                return x + y

            @jit(["int32(int32, int32)", "float32(float32, float32)"])
            def bar(x, y):
                return x + y

    2) jit(function, **targetoptions) -> dispatcher

        Create a dispatcher function object that specializes at call site.

        Examples:

            @jit
            def foo(x, y):
                return x + y

            @jit(nopython=True)
            def bar(x, y):
                return x + y

    '''
def _jit(sigs, locals, target, cache, targetoptions, **dispatcher_args): ...
def njit(*args, **kws):
    """
    Equivalent to jit(nopython=True)

    See documentation for jit function/decorator for full description.
    """
def cfunc(sig, locals={}, cache: bool = False, pipeline_class: Incomplete | None = None, **options):
    '''
    This decorator is used to compile a Python function into a C callback
    usable with foreign C libraries.

    Usage::
        @cfunc("float64(float64, float64)", nopython=True, cache=True)
        def add(a, b):
            return a + b

    '''
def jit_module(**kwargs) -> None:
    """ Automatically ``jit``-wraps functions defined in a Python module

    Note that ``jit_module`` should only be called at the end of the module to
    be jitted. In addition, only functions which are defined in the module
    ``jit_module`` is called from are considered for automatic jit-wrapping.
    See the Numba documentation for more information about what can/cannot be
    jitted.

    :param kwargs: Keyword arguments to pass to ``jit`` such as ``nopython``
                   or ``error_model``.

    """
