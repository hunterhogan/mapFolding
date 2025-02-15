from mapFolding import indexMy
from mapFolding import indexTrack
from numba import int64
from numba import jit
from numba import uint8
from numpy import dtype
from numpy import ndarray
from numpy import integer
from typing import Tuple
from typing import Any

@jit((uint8[:, :, ::1], int64[::1], uint8[::1], uint8[::1], uint8[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nopython=True, parallel=False)
def countSequential(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]], foldGroups: ndarray[Tuple[int], dtype[integer[Any]]], gapsWhere: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
    leafBelow = track[indexTrack.leafBelow.value]
    gapRangeStart = track[indexTrack.gapRangeStart.value]
    countDimensionsGapped = track[indexTrack.countDimensionsGapped.value]
    leafAbove = track[indexTrack.leafAbove.value]
    leaf1ndex = my[indexMy.leaf1ndex.value]
    dimensionsUnconstrained = my[indexMy.dimensionsUnconstrained.value]
    dimensionsTotal = my[indexMy.dimensionsTotal.value]
    gap1ndexCeiling = my[indexMy.gap1ndexCeiling.value]
    indexDimension = my[indexMy.indexDimension.value]
    leafConnectee = my[indexMy.leafConnectee.value]
    indexMiniGap = my[indexMy.indexMiniGap.value]
    gap1ndex = my[indexMy.gap1ndex.value]
    taskIndex = my[indexMy.taskIndex.value]
    groupsOfFolds: int = 0
    while leaf1ndex:
        if leaf1ndex <= 1 or leafBelow[0] == 1:
            if leaf1ndex > foldGroups[-1]:
                groupsOfFolds += 1
            else:
                dimensionsUnconstrained = dimensionsTotal
                gap1ndexCeiling = gapRangeStart[leaf1ndex - 1]
                indexDimension = 0
                while indexDimension < dimensionsTotal:
                    if connectionGraph[indexDimension, leaf1ndex, leaf1ndex] == leaf1ndex:
                        dimensionsUnconstrained -= 1
                    else:
                        leafConnectee = connectionGraph[indexDimension, leaf1ndex, leaf1ndex]
                        while leafConnectee != leaf1ndex:
                            gapsWhere[gap1ndexCeiling] = leafConnectee
                            if countDimensionsGapped[leafConnectee] == 0:
                                gap1ndexCeiling += 1
                            countDimensionsGapped[leafConnectee] += 1
                            leafConnectee = connectionGraph[indexDimension, leaf1ndex, leafBelow[leafConnectee]]
                    indexDimension += 1
                indexMiniGap = gap1ndex
                while indexMiniGap < gap1ndexCeiling:
                    gapsWhere[gap1ndex] = gapsWhere[indexMiniGap]
                    if countDimensionsGapped[gapsWhere[indexMiniGap]] == dimensionsUnconstrained:
                        gap1ndex += 1
                    countDimensionsGapped[gapsWhere[indexMiniGap]] = 0
                    indexMiniGap += 1
        while leaf1ndex and gap1ndex == gapRangeStart[leaf1ndex - 1]:
            leaf1ndex -= 1
            leafBelow[leafAbove[leaf1ndex]] = leafBelow[leaf1ndex]
            leafAbove[leafBelow[leaf1ndex]] = leafAbove[leaf1ndex]
        if leaf1ndex:
            gap1ndex -= 1
            leafAbove[leaf1ndex] = gapsWhere[gap1ndex]
            leafBelow[leaf1ndex] = leafBelow[leafAbove[leaf1ndex]]
            leafBelow[leafAbove[leaf1ndex]] = leaf1ndex
            leafAbove[leafBelow[leaf1ndex]] = leaf1ndex
            gapRangeStart[leaf1ndex] = gap1ndex
            leaf1ndex += 1
    foldGroups[taskIndex] = groupsOfFolds