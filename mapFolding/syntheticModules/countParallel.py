import numba
from mapFolding import indexMy, indexTrack
@numba.jit((numba.uint8[:,:,::1], numba.int64[::1], numba.uint8[::1], numba.uint8[::1], numba.uint8[:,::1]), parallel=True, boundscheck=False, cache=True, error_model="numpy", fastmath=True, looplift=False, nogil=True, nopython=True)
def countParallel(connectionGraph, foldGroups, gapsWherePARALLEL, myPARALLEL, trackPARALLEL):
    for indexSherpa in numba.prange(myPARALLEL[9]):
        gapsWhere = gapsWherePARALLEL.copy()
        my = myPARALLEL.copy()
        my[10] = indexSherpa
        track = trackPARALLEL.copy()
        groupsOfFolds: int = 0
        while my[7] > 0:
            if my[7] <= 1 or track[1, 0] == 1:
                if my[7] > foldGroups[-1]:
                    groupsOfFolds = groupsOfFolds + 1
                else:
                    my[1] = my[0]
                    my[3] = track[3, my[7] - 1]
                    my[4] = 0
                    while my[4] < my[0]:
                        if connectionGraph[my[4], my[7], my[7]] == my[7]:
                            my[1] -= 1
                        else:
                            my[8] = connectionGraph[my[4], my[7], my[7]]
                            while my[8] != my[7]:
                                if my[7] != my[9] or my[8] % my[9] == my[10]:
                                    gapsWhere[my[3]] = my[8]
                                    if track[2, my[8]] == 0:
                                        my[3] += 1
                                    track[2, my[8]] += 1
                                my[8] = connectionGraph[my[4], my[7], track[1, my[8]]]
                        my[4] += 1
                    my[6] = my[2]
                    while my[6] < my[3]:
                        gapsWhere[my[2]] = gapsWhere[my[6]]
                        if track[2, gapsWhere[my[6]]] == my[1]:
                            my[2] += 1
                        track[2, gapsWhere[my[6]]] = 0
                        my[6] += 1
            while my[7] > 0 and my[2] == track[3, my[7] - 1]:
                my[7] -= 1
                track[1, track[0, my[7]]] = track[1, my[7]]
                track[0, track[1, my[7]]] = track[0, my[7]]
            if my[7] > 0:
                my[2] -= 1
                track[0, my[7]] = gapsWhere[my[2]]
                track[1, my[7]] = track[1, track[0, my[7]]]
                track[1, track[0, my[7]]] = my[7]
                track[0, track[1, my[7]]] = my[7]
                track[3, my[7]] = my[2]
                my[7] += 1
        foldGroups[my[10]] = groupsOfFolds