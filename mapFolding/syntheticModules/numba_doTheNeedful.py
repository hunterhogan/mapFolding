from mapFolding import indexMy
from mapFolding.syntheticModules.numbaCount import countInitialize, countParallel, countSequential
from numba import uint16, jit, int64
from numpy import ndarray, dtype, integer
from typing import Any

@jit((uint16[:, :, ::1], int64[::1], uint16[::1], uint16[::1], uint16[::1], uint16[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def doTheNeedful(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], foldGroups: ndarray[tuple[int], dtype[integer[Any]]], gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], mapShape: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
    countInitialize(connectionGraph, gapsWhere, my, track)
    if my[indexMy.taskDivisions.value] > 0:
        countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
    else:
        countSequential(connectionGraph, foldGroups, gapsWhere, my, track)