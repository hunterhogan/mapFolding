from numba import njit
a = 0
b = 1
count = 2
gapter = 3

@njit(cache=True)
def doWhile(computationDivisions, computationIndex, foldingsTotal, n, track, gap, ndim, leafConnectionGraph):
    g = 0
    l = 1
    while l > 0:
        if l <= 1 or track[b][0] == 1:
            if l > n:
                foldingsTotal += n
            else:
                dd = 0
                gg = track[gapter][l - 1]
                g = gg
                for i in range(1, ndim + 1):
                    if leafConnectionGraph[i][l][l] == l:
                        dd += 1
                    else:
                        m = leafConnectionGraph[i][l][l]
                        while m != l:
                            if computationDivisions == 0 or l != computationDivisions or m % computationDivisions == computationIndex:
                                gap[gg] = m
                                track[count][m] += 1
                                gg += 1
                            m = leafConnectionGraph[i][l][track[b][m]]
                if dd == ndim:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1
                k = g
                for j in range(g, gg):
                    if track[count][gap[j]] == ndim - dd:
                        gap[k] = gap[j]
                        k += 1
                    track[count][gap[j]] = 0
                g = k
        while l > 0 and g == track[gapter][l - 1]:
            l -= 1
            if l > 0:
                track[b][track[a][l]] = track[b][l]
                track[a][track[b][l]] = track[a][l]
        if l > 0:
            g -= 1
            track[a][l] = gap[g]
            track[b][l] = track[b][track[a][l]]
            track[b][track[a][l]] = l
            track[a][track[b][l]] = l
            track[gapter][l] = g
            l += 1
    return foldingsTotal
