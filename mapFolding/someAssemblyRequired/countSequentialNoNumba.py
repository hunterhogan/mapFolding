


def countSequential(connectionGraph, foldGroups, gapsWhere, my, track):
    doFindGaps = True
    while my[7] > 0:
        if (doFindGaps := (my[7] <= 1 or track[1, 0] == 1)) and my[7] > foldGroups[-1]:
            foldGroups[my[10]] += 1
        elif doFindGaps:
            my[1] = my[0]
            my[3] = track[3, my[7] - 1]
            my[4] = 0
            while my[4] < my[0]:
                if connectionGraph[my[4], my[7], my[7]] == my[7]:
                    my[1] -= 1
                else:
                    my[8] = connectionGraph[my[4], my[7], my[7]]
                    while my[8] != my[7]:
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