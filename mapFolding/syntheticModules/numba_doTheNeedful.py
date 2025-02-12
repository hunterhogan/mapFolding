from mapFolding import indexMy
from numba import jit
from numba import int64
from numba import uint8
from numpy import dtype
from numpy import integer
from numpy import ndarray
from typing import Tuple
from typing import Any
from mapFolding.syntheticModules.numba_countInitialize import countInitialize
from mapFolding.syntheticModules.numba_countParallel import countParallel
from mapFolding.syntheticModules.numba_countSequential import countSequential

@jit((uint8[:, :, ::1], int64[::1], uint8[::1], uint8[::1], uint8[::1], uint8[:, ::1]), _nrt=True, boundscheck=True, cache=True, error_model='python', fastmath=False, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def doTheNeedful(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]], foldGroups: ndarray[Tuple[int], dtype[integer[Any]]], gapsWhere: ndarray[Tuple[int], dtype[integer[Any]]], mapShape: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
    countInitialize(connectionGraph, gapsWhere, my, track)
    if my[indexMy.taskDivisions.value] > 0:
        countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
    else:
        countSequential(connectionGraph, foldGroups, gapsWhere, my, track)