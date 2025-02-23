from _typeshed import Incomplete
from numba.core import errors as errors, ir as ir
from numba.core.rewrites import Rewrite as Rewrite, register_rewrite as register_rewrite

class RewritePrintCalls(Rewrite):
    """
    Rewrite calls to the print() global function to dedicated IR print() nodes.
    """
    prints: Incomplete
    block: Incomplete
    def match(self, func_ir, block, typemap, calltypes): ...
    def apply(self):
        """
        Rewrite `var = call <print function>(...)` as a sequence of
        `print(...)` and `var = const(None)`.
        """

class DetectConstPrintArguments(Rewrite):
    """
    Detect and store constant arguments to print() nodes.
    """
    consts: Incomplete
    block: Incomplete
    def match(self, func_ir, block, typemap, calltypes): ...
    def apply(self):
        """
        Store detected constant arguments on their nodes.
        """
