from numpy import integer
from typing import Any, Tuple
import numba
from mapFolding import indexMy, indexTrack
import numpy
@numba.jit((numba.uint8[:, :, ::1], numba.int64[::1], numba.uint8[::1], numba.uint8[::1], numba.uint8[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=False, inline='never', looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nopython=True, parallel=False)
def countSequential(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]):
    doFindGaps = True
    dimensionsTotal = my[indexMy.dimensionsTotal.value]
    dimensionsUnconstrained = my[indexMy.dimensionsUnconstrained.value]
    gap1ndex = my[indexMy.gap1ndex.value]
    gap1ndexCeiling = my[indexMy.gap1ndexCeiling.value]
    indexDimension = my[indexMy.indexDimension.value]
    indexMiniGap = my[indexMy.indexMiniGap.value]
    leaf1ndex = my[indexMy.leaf1ndex.value]
    leafConnectee = my[indexMy.leafConnectee.value]
    taskIndex = my[indexMy.taskIndex.value]
    leafAbove = track[indexTrack.leafAbove.value]
    leafBelow = track[indexTrack.leafBelow.value]
    countDimensionsGapped = track[indexTrack.countDimensionsGapped.value]
    gapRangeStart = track[indexTrack.gapRangeStart.value]
    groupsOfFolds: int = 0
    while leaf1ndex > 0:
        if (doFindGaps := (leaf1ndex <= 1 or leafBelow[0] == 1)) and leaf1ndex > foldGroups[-1]:
            groupsOfFolds = groupsOfFolds + 1
        elif doFindGaps:
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
        while leaf1ndex > 0 and gap1ndex == gapRangeStart[leaf1ndex - 1]:
            leaf1ndex -= 1
            leafBelow[leafAbove[leaf1ndex]] = leafBelow[leaf1ndex]
            leafAbove[leafBelow[leaf1ndex]] = leafAbove[leaf1ndex]
        if leaf1ndex > 0:
            gap1ndex -= 1
            leafAbove[leaf1ndex] = gapsWhere[gap1ndex]
            leafBelow[leaf1ndex] = leafBelow[leafAbove[leaf1ndex]]
            leafBelow[leafAbove[leaf1ndex]] = leaf1ndex
            leafAbove[leafBelow[leaf1ndex]] = leaf1ndex
            gapRangeStart[leaf1ndex] = gap1ndex
            leaf1ndex += 1
    foldGroups[taskIndex] = groupsOfFolds