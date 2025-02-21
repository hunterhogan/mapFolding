from mapFolding import indexMy
from mapFolding.syntheticModules.numbaCount import countParallel, countSequential, countInitialize
from numba import jit, int64, uint16
from numpy import ndarray, dtype, integer
from typing import Tuple, Any

@jit((uint16[:, :, ::1], int64[::1], uint16[::1], uint16[::1], uint16[::1], uint16[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def doTheNeedful(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]], foldGroups: ndarray[Tuple[int], dtype[integer[Any]]], gapsWhere: ndarray[Tuple[int], dtype[integer[Any]]], mapShape: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
    countInitialize(connectionGraph, gapsWhere, my, track)
    if my[indexMy.taskDivisions.value] > 0:
        countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
    else:
        countSequential(connectionGraph, foldGroups, gapsWhere, my, track)