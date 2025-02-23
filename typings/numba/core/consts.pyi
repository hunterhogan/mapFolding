from _typeshed import Incomplete
from numba.core import ir as ir
from numba.core.errors import ConstantInferenceError as ConstantInferenceError, NumbaError as NumbaError

class ConstantInference:
    """
    A constant inference engine for a given interpreter.
    Inference inspects the IR to try and compute a compile-time constant for
    a variable.

    This shouldn't be used directly, instead call Interpreter.infer_constant().
    """
    _func_ir: Incomplete
    _cache: Incomplete
    def __init__(self, func_ir) -> None: ...
    def infer_constant(self, name, loc: Incomplete | None = None):
        """
        Infer a constant value for the given variable *name*.
        If no value can be inferred, numba.errors.ConstantInferenceError
        is raised.
        """
    def _fail(self, val) -> None: ...
    def _do_infer(self, name): ...
    def _infer_expr(self, expr): ...
    def _infer_call(self, func, expr): ...
    def _infer_getattr(self, value, expr): ...
