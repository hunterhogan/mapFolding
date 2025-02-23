from _typeshed import Incomplete
from numba.core import config as config

class Rewrite:
    """Defines the abstract base class for Numba rewrites.
    """
    def __init__(self, state: Incomplete | None = None) -> None:
        """Constructor for the Rewrite class.
        """
    def match(self, func_ir, block, typemap, calltypes):
        """Overload this method to check an IR block for matching terms in the
        rewrite.
        """
    def apply(self) -> None:
        """Overload this method to return a rewritten IR basic block when a
        match has been found.
        """

class RewriteRegistry:
    """Defines a registry for Numba rewrites.
    """
    _kinds: Incomplete
    rewrites: Incomplete
    def __init__(self) -> None:
        """Constructor for the rewrite registry.  Initializes the rewrites
        member to an empty list.
        """
    def register(self, kind):
        """
        Decorator adding a subclass of Rewrite to the registry for
        the given *kind*.
        """
    def apply(self, kind, state) -> None:
        """Given a pipeline and a dictionary of basic blocks, exhaustively
        attempt to apply all registered rewrites to all basic blocks.
        """

rewrite_registry: Incomplete
register_rewrite: Incomplete
