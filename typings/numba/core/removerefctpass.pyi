from _typeshed import Incomplete
from llvmlite.ir.transforms import CallVisitor
from numba.core import types as types

class _MarkNrtCallVisitor(CallVisitor):
    """
    A pass to mark all NRT_incref and NRT_decref.
    """
    marked: Incomplete
    def __init__(self) -> None: ...
    def visit_Call(self, instr) -> None: ...

def _rewrite_function(function) -> None: ...

_accepted_nrtfns: Incomplete

def _legalize(module, dmm, fndesc):
    """
    Legalize the code in the module.
    Returns True if the module is legal for the rewrite pass that removes
    unnecessary refcounts.
    """
def remove_unnecessary_nrt_usage(function, context, fndesc):
    """
    Remove unnecessary NRT incref/decref in the given LLVM function.
    It uses highlevel type info to determine if the function does not need NRT.
    Such a function does not:

    - return array object(s);
    - take arguments that need refcounting except array;
    - call function(s) that return refcounted object.

    In effect, the function will not capture or create references that extend
    the lifetime of any refcounted objects beyond the lifetime of the function.

    The rewrite is performed in place.
    If rewrite has happened, this function returns True, otherwise, it returns False.
    """
