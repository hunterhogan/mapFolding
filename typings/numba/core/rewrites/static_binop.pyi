from _typeshed import Incomplete
from numba.core import errors as errors, ir as ir
from numba.core.rewrites import Rewrite as Rewrite, register_rewrite as register_rewrite

class DetectStaticBinops(Rewrite):
    """
    Detect constant arguments to select binops.
    """
    rhs_operators: Incomplete
    static_lhs: Incomplete
    static_rhs: Incomplete
    block: Incomplete
    def match(self, func_ir, block, typemap, calltypes): ...
    def apply(self):
        """
        Store constant arguments that were detected in match().
        """
