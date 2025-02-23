from numba.core import cgutils as cgutils, errors as errors, types as types
from numba.core.extending import intrinsic as intrinsic

def exception_check(typingctx):
    """An intrinsic to check if an exception is raised
    """
def mark_try_block(typingctx):
    """An intrinsic to mark the start of a *try* block.
    """
def end_try_block(typingctx):
    """An intrinsic to mark the end of a *try* block.
    """
def exception_match(typingctx, exc_value, exc_class):
    """Basically do ``isinstance(exc_value, exc_class)`` for exception objects.
    Used in ``except Exception:`` syntax.
    """
