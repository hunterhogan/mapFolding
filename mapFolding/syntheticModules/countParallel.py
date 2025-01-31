from numpy import integer
from typing import Any, Tuple
import numba
from mapFolding import indexMy, indexTrack
import numpy
@numba.jit((numba.uint8[:, :, ::1], numba.int64[::1], numba.uint8[::1], numba.uint8[::1], numba.uint8[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=False, inline='never', looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nopython=True, parallel=True)
def countParallel(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], gapsWherePARALLEL: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], myPARALLEL: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], trackPARALLEL: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]):
    for indexSherpa in numba.prange(myPARALLEL[indexMy.taskDivisions.value]):
        gapsWhere = gapsWherePARALLEL.copy()
        my = myPARALLEL.copy()
        my[indexMy.taskIndex.value] = indexSherpa
        track = trackPARALLEL.copy()
        groupsOfFolds: int = 0
        while my[indexMy.leaf1ndex.value] > 0:
            if my[indexMy.leaf1ndex.value] <= 1 or track[indexTrack.leafBelow.value, 0] == 1:
                if my[indexMy.leaf1ndex.value] > foldGroups[-1]:
                    groupsOfFolds = groupsOfFolds + 1
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