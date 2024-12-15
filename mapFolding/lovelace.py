from numba import njit
import numpy
from .lovelaceIndices import A, B, count, gapter, taskDivisions, taskIndex, leavesTotal, dimensionsTotal

#cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False 
@njit(cache=False) #cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False 
def countFoldings(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], gap: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]):
    # variables: as in, the value varies
    foldingsTotal = 0
    g = 0            # Gap index
    l = 1            # Current leaf

    while l > 0:
        if l <= 1 or track[B][0] == 1: 
            if l > the[leavesTotal]:
                foldingsTotal += the[leavesTotal]
            else:
                dd = 0     # Number of sections where leaf l is unconstrained
                gg = track[gapter][l - 1]  # Track possible gaps 
                g = gg      # Reset gap index

                for i in range(1, the[dimensionsTotal] + 1): # Count possible gaps for leaf l in each section
                    if D[i][l][l] == l:
                        dd += 1
                    else:
                        m = D[i][l][l]
                        while m != l:
                            if the[taskDivisions] == 0 or l != the[taskDivisions] or m % the[taskDivisions] == the[taskIndex]:
                                gap[gg] = m
                                if track[count][m] == 0:
                                    gg += 1
                                track[count][m] += 1
                            m = D[i][l][track[B][m]]

                if dd == the[dimensionsTotal]: # If leaf l is unconstrained in all sections, it can be inserted anywhere
                    for m in range(l):
                        gap[gg] = m
                        gg += 1

                for j in range(g, gg): # Filter gaps that are common to all sections
                    gap[g] = gap[j]
                    if track[count][gap[j]] == the[dimensionsTotal] - dd:
                        g += 1
                    track[count][gap[j]] = 0  # Reset track[count] for next iteration

        while l > 0 and g == track[gapter][l - 1]: # Recursive backtracking steps
            l -= 1
            track[B][track[A][l]] = track[B][l]
            track[A][track[B][l]] = track[A][l]

        if l > 0:
            g -= 1
            track[A][l] = gap[g]
            track[B][l] = track[B][track[A][l]]
            track[B][track[A][l]] = l
            track[A][track[B][l]] = l
            track[gapter][l] = g  # Save current gap index
            l += 1         # Move to next leaf
    return foldingsTotal
