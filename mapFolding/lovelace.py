"""
The algorithm for counting folds.

Starting from established data structures, the algorithm initializes some baseline values. The initialization uses a loop that is not used after the first fold is counted.

After initialization, the folds are either counted sequentially or counted with inefficiently divided parallel tasks.

All three of these actions--initialization, sequential counting, and parallel counting--use nearly identical logic. Without Numba, all of the logic is in one function with exactly one additional
conditional statement for initialization and exactly one additional conditional statement for parallel counting.

Numba's just-in-time (jit) compiler, especially super jit, is capable of radically increasing throughput and dramatically reducing the size of the compiled code, especially by ejecting unused code.

The complexity of this module is due to me allegedly applying Numba's features. Allegedly.

(The flow starts with the last function.)
"""
from mapFolding import indexMy, indexThe, indexTrack
from numpy import integer
from numpy.typing import NDArray
from typing import Any, Tuple, Optional
import numba
import numpy

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def ifComputationDivisions(my: NDArray[integer[Any]], the: NDArray[integer[Any]]) -> bool:
    if the[indexThe.taskDivisions.value] == 0:
        return True
    return my[indexMy.leaf1ndex.value] != the[indexThe.taskDivisions.value] or \
            (my[indexMy.leafConnectee.value] % the[indexThe.taskDivisions.value]) == my[indexMy.taskIndex.value]

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def insertUnconstrainedLeaf(my: NDArray[integer[Any]], the: NDArray[integer[Any]], initializeUnconstrainedLeaf: Optional[bool]) -> bool:
    if initializeUnconstrainedLeaf:
        return my[indexMy.dimensionsUnconstrained.value] == the[indexThe.dimensionsTotal.value]
    else:
        return False

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def initializationConditionUnconstrainedLeaf(my: NDArray[integer[Any]], initializeUnconstrainedLeaf: Optional[bool]) -> bool:
    if initializeUnconstrainedLeaf is None or initializeUnconstrainedLeaf is False:
        return False
    else:
        if my[indexMy.gap1ndex.value] > 0:
            return True
        else:
            return False

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def doWhile(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]], initializeUnconstrainedLeaf: Optional[bool]) -> Tuple[NDArray[integer[Any]], NDArray[integer[Any]], NDArray[integer[Any]], NDArray[integer[Any]]]:
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
                            # NOTE This conditional check should only be in the parallel counting branch
                            if ifComputationDivisions(my, the):
                                gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
                                if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
                                    my[indexMy.gap1ndexCeiling.value] += 1
                                track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1
                            my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]
                    my[indexMy.dimension1ndex.value] += 1
                # NOTE This `if` statement and `while` loop should be absent from the code that does the counting
                if insertUnconstrainedLeaf(my, the, initializeUnconstrainedLeaf):
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
        # NOTE This check and break should be absent from the code that does the counting
        if initializationConditionUnconstrainedLeaf(my, initializeUnconstrainedLeaf):
            break
    return foldsTotal, my, gapsWhere, track

@numba.jit(parallel=True, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def doTaskIndices(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]) -> NDArray[integer[Any]]:
    """This is the only function with the `parallel=True` option.
    Make a copy of the initialized state because all task divisions can start from this baseline.
    Run the counting algorithm but with conditional execution of a few lines of code, so each task has an incomplete count that does not overlap with other tasks."""
    stateFoldsSubTotal = foldsTotal.copy()
    stateMy = my.copy()
    statePotentialGaps = gapsWhere.copy()
    stateTrack = track.copy()

    for indexSherpa in numba.prange(the[indexThe.taskDivisions.value]):
        my = stateMy.copy()
        my[indexMy.taskIndex.value] = indexSherpa
        foldsSubTotal, _1, _2, _3 = doWhile(connectionGraph, stateFoldsSubTotal.copy(), my, statePotentialGaps.copy(), the, stateTrack.copy(), initializeUnconstrainedLeaf=False)

        foldsTotal[indexSherpa] = foldsSubTotal[indexSherpa]

    return foldsTotal

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def countFoldsCompileBranch(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]], obviousFlagForNumba: bool) -> NDArray[integer[Any]]:
    """Allegedly, `obviousFlagForNumba` allows Numba to compile two versions: one for parallel execution and one leaner version for sequential execution."""
    if obviousFlagForNumba:
        foldsTotal, _1, _2, _3 = doWhile(connectionGraph, foldsTotal, my, gapsWhere, the, track, initializeUnconstrainedLeaf=False)
    else:
        foldsTotal = doTaskIndices(connectionGraph, foldsTotal, my, gapsWhere, the, track)

    return foldsTotal

@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def countFoldsCompiled(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], my: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]) -> int:
    # ^ Receive the data structures.

    # Initialize baseline values primarily to eliminate the need for the logic of `insertUnconstrainedLeaf`
    _0, my, gapsWhere, track = doWhile(connectionGraph, foldsTotal, my, gapsWhere, the, track, initializeUnconstrainedLeaf=True)

    obviousFlagForNumba = the[indexThe.taskDivisions.value] == int(False)

    # Call the function that will branch to sequential or parallel counting
    foldsTotal = countFoldsCompileBranch(connectionGraph, foldsTotal, my, gapsWhere, the, track, obviousFlagForNumba)

    # Return an `int` integer
    return numpy.sum(foldsTotal).item()
