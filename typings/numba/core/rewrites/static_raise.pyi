from _typeshed import Incomplete
from numba.core import consts as consts, errors as errors, ir as ir
from numba.core.rewrites import Rewrite as Rewrite, register_rewrite as register_rewrite

class RewriteConstRaises(Rewrite):
    """
    Rewrite IR statements of the kind `raise(value)`
    where `value` is the result of instantiating an exception with
    constant arguments
    into `static_raise(exception_type, constant args)`.

    This allows lowering in nopython mode, where one can't instantiate
    exception instances from runtime data.
    """
    def _is_exception_type(self, const): ...
    def _break_constant(self, const, loc):
        """
        Break down constant exception.
        """
    def _try_infer_constant(self, func_ir, inst): ...
    raises: Incomplete
    tryraises: Incomplete
    block: Incomplete
    def match(self, func_ir, block, typemap, calltypes): ...
    def apply(self):
        """
        Rewrite all matching setitems as static_setitems.
        """
