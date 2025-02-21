from mapFolding import indexTrack, indexMy
from numba import int64, prange, uint16, jit
from numpy import ndarray, dtype, integer
from typing import Any

@jit((uint16[:, :, ::1], uint16[::1], uint16[::1], uint16[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def countInitialize(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
    while my[indexMy.leaf1ndex.value] > 0:
        if my[indexMy.leaf1ndex.value] <= 1 or track[indexTrack.leafBelow.value, 0] == 1:
            my[indexMy.dimensionsUnconstrained.value] = my[indexMy.dimensionsTotal.value]
            my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
            my[indexMy.indexDimension.value] = 0
            while my[indexMy.indexDimension.value] < my[indexMy.dimensionsTotal.value]:
                if connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]:
                    my[indexMy.dimensionsUnconstrained.value] -= 1
                else:
                    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]
                    while my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]:
                        gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
                        if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
                            my[indexMy.gap1ndexCeiling.value] += 1
                        track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1
                        my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]
                my[indexMy.indexDimension.value] += 1
            if not my[indexMy.dimensionsUnconstrained.value]:
                my[indexMy.indexLeaf.value] = 0
                while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
                    gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.indexLeaf.value]
                    my[indexMy.gap1ndexCeiling.value] += 1
                    my[indexMy.indexLeaf.value] += 1
            my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]
            while my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]:
                gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
                if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == my[indexMy.dimensionsUnconstrained.value]:
                    my[indexMy.gap1ndex.value] += 1
                track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] = 0
                my[indexMy.indexMiniGap.value] += 1
        if my[indexMy.leaf1ndex.value] > 0:
            my[indexMy.gap1ndex.value] -= 1
            track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
            track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
            track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
            track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
            track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
            my[indexMy.leaf1ndex.value] += 1
        if my[indexMy.gap1ndex.value] > 0:
            return

@jit((uint16[:, :, ::1], int64[::1], uint16[::1], uint16[::1], uint16[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nopython=True, parallel=True)
def countParallel(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], foldGroups: ndarray[tuple[int], dtype[integer[Any]]], gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
    gapsWherePARALLEL = gapsWhere.copy()
    myPARALLEL = my.copy()
    trackPARALLEL = track.copy()
    taskDivisionsPrange = myPARALLEL[indexMy.taskDivisions.value]
    for indexSherpa in prange(taskDivisionsPrange):
        groupsOfFolds: int = 0
        gapsWhere = gapsWherePARALLEL.copy()
        my = myPARALLEL.copy()
        track = trackPARALLEL.copy()
        my[indexMy.taskIndex.value] = indexSherpa
        while my[indexMy.leaf1ndex.value] > 0:
            if my[indexMy.leaf1ndex.value] <= 1 or track[indexTrack.leafBelow.value, 0] == 1:
                if my[indexMy.leaf1ndex.value] > foldGroups[-1]:
                    groupsOfFolds += 1
                else:
                    my[indexMy.dimensionsUnconstrained.value] = my[indexMy.dimensionsTotal.value]
                    my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
                    my[indexMy.indexDimension.value] = 0
                    while my[indexMy.indexDimension.value] < my[indexMy.dimensionsTotal.value]:
                        if connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]:
                            my[indexMy.dimensionsUnconstrained.value] -= 1
                        else:
                            my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]
                            while my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]:
                                if my[indexMy.leaf1ndex.value] != my[indexMy.taskDivisions.value] or my[indexMy.leafConnectee.value] % my[indexMy.taskDivisions.value] == my[indexMy.taskIndex.value]:
                                    gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
                                    if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
                                        my[indexMy.gap1ndexCeiling.value] += 1
                                    track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1
                                my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]
                        my[indexMy.indexDimension.value] += 1
                    my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]
                    while my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]:
                        gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
                        if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == my[indexMy.dimensionsUnconstrained.value]:
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
        foldGroups[my[indexMy.taskIndex.value]] = groupsOfFolds

@jit((uint16[:, :, ::1], int64[::1], uint16[::1], uint16[::1], uint16[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nopython=True, parallel=False)
def countSequential(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], foldGroups: ndarray[tuple[int], dtype[integer[Any]]], gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
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
    while leaf1ndex > 0:
        if leaf1ndex <= 1 or leafBelow[0] == 1:
            if leaf1ndex > foldGroups[-1]:
                groupsOfFolds += 1
            else:
                dimensionsUnconstrained = dimensionsTotal
                gap1ndexCeiling = gapRangeStart[leaf1ndex - 1]
                indexDimension = 0
                while indexDimension < dimensionsTotal:
                    leafConnectee = connectionGraph[indexDimension, leaf1ndex, leaf1ndex]
                    if leafConnectee == leaf1ndex:
                        dimensionsUnconstrained -= 1
                    else:
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