from _typeshed import Incomplete
from llvmlite import ir

class _DivmodFixer(ir.Visitor):
    def visit_Instruction(self, instr) -> None: ...

def fix_divmod(mod) -> None:
    """Replace division and reminder instructions to builtins calls
    """

INTR_TO_CMATH: Incomplete
OTHER_CMATHS: Incomplete
INTR_MATH: Incomplete
