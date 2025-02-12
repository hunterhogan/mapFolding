from mapFolding import indexTrack
from mapFolding import indexMy
from numba import jit
from numba import uint8
from numpy import ndarray
from numpy import integer
from numpy import dtype
from typing import Any
from typing import Tuple

@jit((uint8[:, :, ::1], uint8[::1], uint8[::1], uint8[:, ::1]), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def countInitialize(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]], gapsWhere: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
    while my[indexMy.leaf1ndex.value]:
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
        if my[indexMy.leaf1ndex.value]:
            my[indexMy.gap1ndex.value] -= 1
            track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
            track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
            track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
            track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
            track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
            my[indexMy.leaf1ndex.value] += 1
        if my[indexMy.gap1ndex.value] > 0:
            return