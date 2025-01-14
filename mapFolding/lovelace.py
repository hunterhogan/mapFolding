from mapFolding import indexMy, indexThe, indexTrack
from numpy import integer
from numpy.typing import NDArray
from typing import Any, Optional
import numba
import numpy

# Allegedly, `-=` is an optimized, in-place operation in numpy and likely the best choice.
# TODO figure out why numba won't work unless I include ".value" in statements such as `my[indexMy.leaf1ndex.value] -= 1`

# @numba.jit(debug=True, _dbg_optnone=True)
# @numba.jit(parallel=True, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def countFoldsCompiled(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]], my: NDArray[integer[Any]], potentialGaps: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]) -> int:
    # TODO conditional compile
    def ifComputationDivisions(my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
        """This function (allegedly) allows numba to compile two different versions based on the value of `taskDivisions`. The benefit is one less conditional check in the main loop, which is important. As I write this, I have a CPU process that I estimate is about half way through counting folds: "Total Time: 20:35:45.593; Cycles: 135,149,788,103,967", so removing a statement that might account for only .1% of 135 trillion cycles, would save 135 billion cycles. That's a lot of cycles. """
        if the[indexThe.taskDivisions.value] == 0:
            return True
        return my[indexMy.leaf1ndex.value] != the[indexThe.taskDivisions.value] or numpy.equal(numpy.mod(my[indexMy.leafConnectee.value], the[indexThe.taskDivisions.value]), my[indexMy.taskIndex.value])

    # TODO conditional compile
    def insertUnconstrainedLeaf(my: NDArray[integer[Any]], potentialGaps: NDArray[integer[Any]], the: NDArray[integer[Any]], Z0Z_initializeUnconstrainedLeaf: Optional[bool]):
        if Z0Z_initializeUnconstrainedLeaf:
            if my[indexMy.dimensionsUnconstrained.value] == the[indexThe.dimensionsTotal.value]:
                my[indexMy.indexLeaf.value] = 0
                while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
                    potentialGaps[my[indexMy.gap1ndexLowerBound.value]] = my[indexMy.indexLeaf.value]
                    my[indexMy.gap1ndexLowerBound.value] += 1
                    my[indexMy.indexLeaf.value] += 1

    # TODO conditional compile
    def lolaCondition_initializeUnconstrainedLeaf(my: NDArray[integer[Any]]):
        if my[indexMy.gap1ndex.value] > 0:
            return int(True)
        return int(False)

    # TODO more semantic function identifier
    def doWhile(
        connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]],
        my: NDArray[integer[Any]], potentialGaps: NDArray[integer[Any]], the: NDArray[integer[Any]],
        track: NDArray[integer[Any]] , Z0Z_initializeUnconstrainedLeaf: Optional[bool]
        ):
        while my[indexMy.leaf1ndex.value] > 0:
            if my[indexMy.leaf1ndex.value] <= 1 or track[indexTrack.leafBelow.value, 0] == 1:
                if my[indexMy.leaf1ndex.value] > the[indexThe.leavesTotal.value]:
                    foldsTotal[my[indexMy.taskIndex.value]] += the[indexThe.leavesTotal.value]
                else:
                    my[indexMy.dimensionsUnconstrained.value] = 0
                    my[indexMy.gap1ndexLowerBound.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
                    my[indexMy.dimension1ndex.value] = 1
                    while my[indexMy.dimension1ndex.value] <= the[indexThe.dimensionsTotal.value]:
                        if connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]:
                            my[indexMy.dimensionsUnconstrained.value] += 1
                        else:
                            my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]
                            while my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]:
                                if ifComputationDivisions(my, the):
                                    potentialGaps[my[indexMy.gap1ndexLowerBound.value]] = my[indexMy.leafConnectee.value]
                                    if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
                                        my[indexMy.gap1ndexLowerBound.value] += 1
                                    track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1
                                my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]
                        my[indexMy.dimension1ndex.value] += 1
                    insertUnconstrainedLeaf(my, potentialGaps, the, Z0Z_initializeUnconstrainedLeaf)
                    my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]
                    while my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexLowerBound.value]:
                        potentialGaps[my[indexMy.gap1ndex.value]] = potentialGaps[my[indexMy.indexMiniGap.value]]
                        if track[indexTrack.countDimensionsGapped.value, potentialGaps[my[indexMy.indexMiniGap.value]]] == the[indexThe.dimensionsTotal.value] - my[indexMy.dimensionsUnconstrained.value]:
                            my[indexMy.gap1ndex.value] += 1
                        track[indexTrack.countDimensionsGapped.value, potentialGaps[my[indexMy.indexMiniGap.value]]] = 0
                        my[indexMy.indexMiniGap.value] += 1
            while my[indexMy.leaf1ndex.value] > 0 and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]:
                my[indexMy.leaf1ndex.value] -= 1
                track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
                track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]
            if my[indexMy.leaf1ndex.value] > 0:
                my[indexMy.gap1ndex.value] -= 1
                track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = potentialGaps[my[indexMy.gap1ndex.value]]
                track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
                track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
                track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
                track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
                my[indexMy.leaf1ndex.value] += 1
            if Z0Z_initializeUnconstrainedLeaf:
                if lolaCondition_initializeUnconstrainedLeaf(my):
                    return

    # TODO conditional compile
    def doTaskIndices(connectionGraph: NDArray[integer[Any]], foldsTotal: NDArray[integer[Any]],
                        my: NDArray[integer[Any]], potentialGaps: NDArray[integer[Any]],
                        the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
        stateMy = my.copy()
        statePotentialGaps = potentialGaps.copy()
        stateTrack = track.copy()
        for indexSherpa in numba.prange(the[indexThe.taskDivisions.value]):
            my = stateMy.copy()
            my[indexMy.taskIndex.value] = indexSherpa
            potentialGaps = statePotentialGaps.copy()
            track = stateTrack.copy()
            doWhile(connectionGraph, foldsTotal, my, potentialGaps, the, track, Z0Z_initializeUnconstrainedLeaf=False)

    # initializeUnconstrainedLeaf
    doWhile(connectionGraph, foldsTotal, my, potentialGaps, the, track, Z0Z_initializeUnconstrainedLeaf=True)

    # TODO is this part of the conditional compile?
    if the[indexThe.taskDivisions.value] == int(False):
        doWhile(connectionGraph, foldsTotal, my, potentialGaps, the, track, Z0Z_initializeUnconstrainedLeaf=False)
    else:
        doTaskIndices(connectionGraph, foldsTotal, my, potentialGaps, the, track)

    return numpy.sum(foldsTotal).item()
