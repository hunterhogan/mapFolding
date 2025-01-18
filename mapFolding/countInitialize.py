import numba

@numba.jit((numba.int64[:, :, ::1], numba.int64[::1], numba.int64[::1], numba.int64[::1], numba.int64[:, ::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def countInitialize(connectionGraph, gapsWhere, my, the, track):
    while my[6] > 0:
        if my[6] <= 1 or track[1, 0] == 1:
            my[1] = 0
            my[3] = track[3, my[6] - 1]
            my[0] = 1
            while my[0] <= the[0]:
                if connectionGraph[my[0], my[6], my[6]] == my[6]:
                    my[1] += 1
                else:
                    my[7] = connectionGraph[my[0], my[6], my[6]]
                    while my[7] != my[6]:
                        gapsWhere[my[3]] = my[7]
                        if track[2, my[7]] == 0:
                            my[3] += 1
                        track[2, my[7]] += 1
                        my[7] = connectionGraph[my[0], my[6], track[1, my[7]]]
                my[0] += 1
            if my[1] == the[0]:
                my[4] = 0
                while my[4] < my[6]:
                    gapsWhere[my[3]] = my[4]
                    my[3] += 1
                    my[4] += 1
            my[5] = my[2]
            while my[5] < my[3]:
                gapsWhere[my[2]] = gapsWhere[my[5]]
                if track[2, gapsWhere[my[5]]] == the[0] - my[1]:
                    my[2] += 1
                track[2, gapsWhere[my[5]]] = 0
                my[5] += 1
        if my[6] > 0:
            my[2] -= 1
            track[0, my[6]] = gapsWhere[my[2]]
            track[1, my[6]] = track[1, track[0, my[6]]]
            track[1, track[0, my[6]]] = my[6]
            track[0, track[1, my[6]]] = my[6]
            track[3, my[6]] = my[2]
            my[6] += 1
        if my[2] > 0:
            return