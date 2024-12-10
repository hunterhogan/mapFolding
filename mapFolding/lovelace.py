from numba import njit

@njit(cache=True)
def doWhile(computationDivisions, computationIndex, foldingsTotal, n, a, b, count, gapter, gap, ndim, leafConnectionGraph):
    g = 0
    l = 1
    while l > 0:
        if l <= 1 or b[0] == 1:
            if l > n:
                foldingsTotal += n
            else:
                dd = 0
                gg = gapter[l - 1]
                g = gg
                for i in range(1, ndim + 1):
                    if leafConnectionGraph[i][l][l] == l:
                        dd += 1
                    else:
                        m = leafConnectionGraph[i][l][l]
                        while m != l:
                            if computationDivisions == 0 or l != computationDivisions or m % computationDivisions == computationIndex:
                                gap[gg] = m
                                count[m] += 1
                                gg += 1
                            m = leafConnectionGraph[i][l][b[m]]
                if dd == ndim:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1
                k = g
                for j in range(g, gg):
                    if count[gap[j]] == ndim - dd:
                        gap[k] = gap[j]
                        k += 1
                    count[gap[j]] = 0
                g = k
        while l > 0 and g == gapter[l - 1]:
            l -= 1
            if l > 0:
                b[a[l]] = b[l]
                a[b[l]] = a[l]
        if l > 0:
            g -= 1
            a[l] = gap[g]
            b[l] = b[a[l]]
            b[a[l]] = l
            a[b[l]] = l
            gapter[l] = g
            l += 1
    return foldingsTotal
