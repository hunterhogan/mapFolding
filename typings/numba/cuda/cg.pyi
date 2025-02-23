from numba.core import types as types
from numba.core.extending import overload as overload, overload_method as overload_method
from numba.core.typing import signature as signature
from numba.cuda import nvvmutils as nvvmutils
from numba.cuda.extending import intrinsic as intrinsic
from numba.cuda.types import grid_group as grid_group

class GridGroup:
    """A cooperative group representing the entire grid"""
    def sync() -> None:
        """Synchronize this grid group"""

def this_grid() -> GridGroup:
    """Get the current grid group."""
def _this_grid(typingctx): ...
def _ol_this_grid(): ...
def _grid_group_sync(typingctx, group): ...
def _ol_grid_group_sync(group): ...
