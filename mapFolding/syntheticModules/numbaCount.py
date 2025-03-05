from mapFolding.theSSOT import indexMy, indexTrack
from numba import uint16, prange, int64, jit
from numpy import ndarray, dtype, integer
from typing import Any

@jit((uint16[:, :, ::1], uint16[::1], uint16[::1], uint16[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def countInitialize(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
    while my[indexMy.leaf1ndex] > 0:
        if my[indexMy.leaf1ndex] <= 1 or track[indexTrack.leafBelow, 0] == 1:
            my[indexMy.dimensionsUnconstrained] = my[indexMy.dimensionsTotal]
            my[indexMy.gap1ndexCeiling] = track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex] - 1]
            my[indexMy.indexDimension] = 0
            while my[indexMy.indexDimension] < my[indexMy.dimensionsTotal]:
                if connectionGraph[my[indexMy.indexDimension], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]] == my[indexMy.leaf1ndex]:
                    my[indexMy.dimensionsUnconstrained] -= 1
                else:
                    my[indexMy.leafConnectee] = connectionGraph[my[indexMy.indexDimension], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]]
                    while my[indexMy.leafConnectee] != my[indexMy.leaf1ndex]:
                        gapsWhere[my[indexMy.gap1ndexCeiling]] = my[indexMy.leafConnectee]
                        if track[indexTrack.countDimensionsGapped, my[indexMy.leafConnectee]] == 0:
                            my[indexMy.gap1ndexCeiling] += 1
                        track[indexTrack.countDimensionsGapped, my[indexMy.leafConnectee]] += 1
                        my[indexMy.leafConnectee] = connectionGraph[my[indexMy.indexDimension], my[indexMy.leaf1ndex], track[indexTrack.leafBelow, my[indexMy.leafConnectee]]]
                my[indexMy.indexDimension] += 1
            if not my[indexMy.dimensionsUnconstrained]:
                my[indexMy.indexLeaf] = 0
                while my[indexMy.indexLeaf] < my[indexMy.leaf1ndex]:
                    gapsWhere[my[indexMy.gap1ndexCeiling]] = my[indexMy.indexLeaf]
                    my[indexMy.gap1ndexCeiling] += 1
                    my[indexMy.indexLeaf] += 1
            my[indexMy.indexMiniGap] = my[indexMy.gap1ndex]
            while my[indexMy.indexMiniGap] < my[indexMy.gap1ndexCeiling]:
                gapsWhere[my[indexMy.gap1ndex]] = gapsWhere[my[indexMy.indexMiniGap]]
                if track[indexTrack.countDimensionsGapped, gapsWhere[my[indexMy.indexMiniGap]]] == my[indexMy.dimensionsUnconstrained]:
                    my[indexMy.gap1ndex] += 1
                track[indexTrack.countDimensionsGapped, gapsWhere[my[indexMy.indexMiniGap]]] = 0
                my[indexMy.indexMiniGap] += 1
        if my[indexMy.leaf1ndex] > 0:
            my[indexMy.gap1ndex] -= 1
            track[indexTrack.leafAbove, my[indexMy.leaf1ndex]] = gapsWhere[my[indexMy.gap1ndex]]
            track[indexTrack.leafBelow, my[indexMy.leaf1ndex]] = track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]]
            track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]] = my[indexMy.leaf1ndex]
            track[indexTrack.leafAbove, track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]] = my[indexMy.leaf1ndex]
            track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex]] = my[indexMy.gap1ndex]
            my[indexMy.leaf1ndex] += 1
        if my[indexMy.gap1ndex] > 0:
            return

@jit((uint16[:, :, ::1], int64[::1], uint16[::1], uint16[::1], uint16[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nopython=True, parallel=True)
def countParallel(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], foldGroups: ndarray[tuple[int], dtype[integer[Any]]], gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
    gapsWherePARALLEL = gapsWhere.copy()
    myPARALLEL = my.copy()
    trackPARALLEL = track.copy()
    taskDivisionsPrange = myPARALLEL[indexMy.taskDivisions]
    for indexSherpa in prange(taskDivisionsPrange):
        groupsOfFolds: int = 0
        gapsWhere = gapsWherePARALLEL.copy()
        my = myPARALLEL.copy()
        track = trackPARALLEL.copy()
        my[indexMy.taskIndex] = indexSherpa
        while my[indexMy.leaf1ndex] > 0:
            if my[indexMy.leaf1ndex] <= 1 or track[indexTrack.leafBelow, 0] == 1:
                if my[indexMy.leaf1ndex] > foldGroups[-1]:
                    groupsOfFolds += 1
                else:
                    my[indexMy.dimensionsUnconstrained] = my[indexMy.dimensionsTotal]
                    my[indexMy.gap1ndexCeiling] = track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex] - 1]
                    my[indexMy.indexDimension] = 0
                    while my[indexMy.indexDimension] < my[indexMy.dimensionsTotal]:
                        if connectionGraph[my[indexMy.indexDimension], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]] == my[indexMy.leaf1ndex]:
                            my[indexMy.dimensionsUnconstrained] -= 1
                        else:
                            my[indexMy.leafConnectee] = connectionGraph[my[indexMy.indexDimension], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]]
                            while my[indexMy.leafConnectee] != my[indexMy.leaf1ndex]:
                                if my[indexMy.leaf1ndex] != my[indexMy.taskDivisions] or my[indexMy.leafConnectee] % my[indexMy.taskDivisions] == my[indexMy.taskIndex]:
                                    gapsWhere[my[indexMy.gap1ndexCeiling]] = my[indexMy.leafConnectee]
                                    if track[indexTrack.countDimensionsGapped, my[indexMy.leafConnectee]] == 0:
                                        my[indexMy.gap1ndexCeiling] += 1
                                    track[indexTrack.countDimensionsGapped, my[indexMy.leafConnectee]] += 1
                                my[indexMy.leafConnectee] = connectionGraph[my[indexMy.indexDimension], my[indexMy.leaf1ndex], track[indexTrack.leafBelow, my[indexMy.leafConnectee]]]
                        my[indexMy.indexDimension] += 1
                    my[indexMy.indexMiniGap] = my[indexMy.gap1ndex]
                    while my[indexMy.indexMiniGap] < my[indexMy.gap1ndexCeiling]:
                        gapsWhere[my[indexMy.gap1ndex]] = gapsWhere[my[indexMy.indexMiniGap]]
                        if track[indexTrack.countDimensionsGapped, gapsWhere[my[indexMy.indexMiniGap]]] == my[indexMy.dimensionsUnconstrained]:
                            my[indexMy.gap1ndex] += 1
                        track[indexTrack.countDimensionsGapped, gapsWhere[my[indexMy.indexMiniGap]]] = 0
                        my[indexMy.indexMiniGap] += 1
            while my[indexMy.leaf1ndex] > 0 and my[indexMy.gap1ndex] == track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex] - 1]:
                my[indexMy.leaf1ndex] -= 1
                track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]] = track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]
                track[indexTrack.leafAbove, track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]] = track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]
            if my[indexMy.leaf1ndex] > 0:
                my[indexMy.gap1ndex] -= 1
                track[indexTrack.leafAbove, my[indexMy.leaf1ndex]] = gapsWhere[my[indexMy.gap1ndex]]
                track[indexTrack.leafBelow, my[indexMy.leaf1ndex]] = track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]]
                track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]] = my[indexMy.leaf1ndex]
                track[indexTrack.leafAbove, track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]] = my[indexMy.leaf1ndex]
                track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex]] = my[indexMy.gap1ndex]
                my[indexMy.leaf1ndex] += 1
        foldGroups[my[indexMy.taskIndex]] = groupsOfFolds

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
