from mapFolding import indexMy, indexThe, indexTrack
from numpy import integer
from numpy.typing import NDArray
from typing import Any, Optional
import numba
import numpy

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def ifComputationDivisions(my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
    if the[indexThe.taskDivisions.value] == 0:
        return True
    return my[indexMy.leaf1ndex.value] != the[indexThe.taskDivisions.value] or \
            (my[indexMy.leafConnectee.value] % the[indexThe.taskDivisions.value]) == my[indexMy.taskIndex.value]

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def insertUnconstrainedLeaf(my: NDArray[integer[Any]], the: NDArray[integer[Any]], Z0Z_initializeUnconstrainedLeaf: Optional[bool]):
    if Z0Z_initializeUnconstrainedLeaf:
        return my[indexMy.dimensionsUnconstrained.value] == the[indexThe.dimensionsTotal.value]
    else:
        return False

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def initializationConditionUnconstrainedLeaf(my: NDArray[integer[Any]], Z0Z_initializeUnconstrainedLeaf: Optional[bool]):
    if Z0Z_initializeUnconstrainedLeaf is None or Z0Z_initializeUnconstrainedLeaf is False:
        return False
    else:
        if my[indexMy.gap1ndex.value] > 0:
            return True
        else:
            return False

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def doWhile(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]], Z0Z_initializeUnconstrainedLeaf: Optional[bool] ):
    while my[indexMy.leaf1ndex.value] > 0:
        if my[indexMy.leaf1ndex.value] <= 1 or track[indexTrack.leafBelow.value, 0] == 1:
            if my[indexMy.leaf1ndex.value] > the[indexThe.leavesTotal.value]:
                foldsTotal[my[indexMy.taskIndex.value]] += the[indexThe.leavesTotal.value]
            else:
                my[indexMy.dimensionsUnconstrained.value] = 0
                my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
                my[indexMy.dimension1ndex.value] = 1
                while my[indexMy.dimension1ndex.value] <= the[indexThe.dimensionsTotal.value]:
                    if connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]:
                        my[indexMy.dimensionsUnconstrained.value] += 1
                    else:
                        my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]
                        while my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]:
                            if ifComputationDivisions(my, the):
                                gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
                                if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
                                    my[indexMy.gap1ndexCeiling.value] += 1
                                track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1
                            my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]
                    my[indexMy.dimension1ndex.value] += 1
                if insertUnconstrainedLeaf(my, the, Z0Z_initializeUnconstrainedLeaf):
                    my[indexMy.indexLeaf.value] = 0
                    while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
                        gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.indexLeaf.value]
                        my[indexMy.gap1ndexCeiling.value] += 1
                        my[indexMy.indexLeaf.value] += 1
                my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]
                while my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]:
                    gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
                    if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == the[indexThe.dimensionsTotal.value] - my[indexMy.dimensionsUnconstrained.value]:
                        my[indexMy.gap1ndex.value] += 1
                    track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] = 0
                    my[indexMy.indexMiniGap.value] += 1
        while my[indexMy.leaf1ndex.value] > 0 and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]:
            my[indexMy.leaf1ndex.value] -= 1
            track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
            track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]
        if my[indexMy.leaf1ndex.value] > 0:
            my[indexMy.gap1ndex.value] -= 1
            track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
            track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
            track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
            track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
            track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
            my[indexMy.leaf1ndex.value] += 1
        if initializationConditionUnconstrainedLeaf(my, Z0Z_initializeUnconstrainedLeaf):
            break
    return foldsTotal, my, gapsWhere, track

@numba.jit(parallel=True, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def doTaskIndices(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):

    stateFoldsSubTotal = foldsTotal.copy()
    stateMy = my.copy()
    statePotentialGaps = gapsWhere.copy()
    stateTrack = track.copy()

    for indexSherpa in numba.prange(the[indexThe.taskDivisions.value]):
        my = stateMy.copy()
        my[indexMy.taskIndex.value] = indexSherpa
        foldsSubTotal, _1, _2, _3 = doWhile(connectionGraph, stateFoldsSubTotal.copy(), my, statePotentialGaps.copy(), the, stateTrack.copy(), Z0Z_initializeUnconstrainedLeaf=False)

        foldsTotal[indexSherpa] = foldsSubTotal[indexSherpa]

    return foldsTotal

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def countFoldsCompileBranch(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]],
                            my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]],
                            obviousFlagForNumba: bool):
    if obviousFlagForNumba:
        foldsTotal, _1, _2, _3 = doWhile(connectionGraph, foldsTotal, my, gapsWhere, the, track, Z0Z_initializeUnconstrainedLeaf=False)
    else:
        foldsTotal = doTaskIndices(connectionGraph, foldsTotal, my, gapsWhere, the, track)

    return foldsTotal

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def countFoldsCompiled(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]) -> int:

    _0, my, gapsWhere, track = doWhile(connectionGraph, foldsTotal, my, gapsWhere, the, track, Z0Z_initializeUnconstrainedLeaf=True)

    obviousFlagForNumba = the[indexThe.taskDivisions.value] == int(False)

    foldsTotal = countFoldsCompileBranch(connectionGraph, foldsTotal, my, gapsWhere, the, track, obviousFlagForNumba)

    return numpy.sum(foldsTotal).item()
