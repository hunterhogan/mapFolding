from _typeshed import Incomplete
from llvmlite.ir.transforms import CallVisitor, Visitor

class FastFloatBinOpVisitor(Visitor):
    """
    A pass to add fastmath flag to float-binop instruction if they don't have
    any flags.
    """
    float_binops: Incomplete
    flags: Incomplete
    def __init__(self, flags) -> None: ...
    def visit_Instruction(self, instr) -> None: ...

class FastFloatCallVisitor(CallVisitor):
    """
    A pass to change all float function calls to use fastmath.
    """
    flags: Incomplete
    def __init__(self, flags) -> None: ...
    def visit_Call(self, instr) -> None: ...

def rewrite_module(mod, options) -> None:
    """
    Rewrite the given LLVM module to use fastmath everywhere.
    """
