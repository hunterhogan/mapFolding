from _typeshed import Incomplete
from numba.core import types as types

class Dim3(types.Type):
    """
    A 3-tuple (x, y, z) representing the position of a block or thread.
    """
    def __init__(self) -> None: ...

class GridGroup(types.Type):
    """
    The grid of all threads in a cooperative kernel launch.
    """
    def __init__(self) -> None: ...

dim3: Incomplete
grid_group: Incomplete

class CUDADispatcher(types.Dispatcher):
    """The type of CUDA dispatchers"""
