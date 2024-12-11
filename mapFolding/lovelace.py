from numba import njit
import numpy
"""
Hypotheses:
- The counting loop should only have necessary logic and data structures
- The rest of the package should prioritize the efficiency of the counting loop
- Static values should be in static data structures
- Dynamic values should be in _well-organized_ ndarray: proven true
"""
"""
What changed               (2, 2, 2, 2, 2)  (2, 11)  (3, 3, 3)  (3, 8)  (5, 5)
almost all*: list                    60.88   323.95    2921.17 1618.19 1869.28  *leafConnectionGraph: ndarray
all: ndarray                         45.98   230.49    2165.87 1143.95 1332.87
make `track` (4, leavesTotal+1)      38.87   199.42    2043.73 1064.41 1254.42
change to measuring `doWhile`        36.44   176.10    1941.60 1046.56 1242.60
"""
# `track` indices
a = 0
b = 1
count = 2
gapter = 3

leavesTotal = -1
dimensionsTotal = -1
tasksTotal = -1
taskActive = -1

leafConnectionGraph = numpy.array(0, dtype=numpy.int64)

def carveInStone(leaves_total, dimensions_total, computationDivisions, computationIndex, theGraph):
    global leavesTotal, dimensionsTotal, tasksTotal, taskActive, leafConnectionGraph
    leavesTotal = leaves_total
    dimensionsTotal = dimensions_total
    tasksTotal = computationDivisions
    taskActive = computationIndex
    leafConnectionGraph = theGraph

# I think cache is a bad idea with global constants.
@njit(cache=False, parallel=False, nogil=True, fastmath=True, boundscheck=False, debug=False)
def doWhile(track, gap):
# def doWhile(track, gap, leafConnectionGraph):
    # print(leavesTotal, dimensionsTotal, tasksTotal, taskActive)
    foldingsTotal = 0
    g = 0
    l = 1
    while l > 0:
        if l <= 1 or track[b][0] == 1:
            if l > leavesTotal:
                foldingsTotal += leavesTotal
            else:
                dd = 0
                gg = track[gapter][l - 1]
                g = gg
                for i in range(1, dimensionsTotal + 1):
                    if leafConnectionGraph[i][l][l] == l:
                        dd += 1
                    else:
                        m = leafConnectionGraph[i][l][l]
                        while m != l:
                            if tasksTotal == 0 or l != tasksTotal or m % tasksTotal == taskActive:
                                gap[gg] = m
                                track[count][m] += 1
                                gg += 1
                            m = leafConnectionGraph[i][l][track[b][m]]
                if dd == dimensionsTotal:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1
                k = g
                for j in range(g, gg):
                    if track[count][gap[j]] == dimensionsTotal - dd:
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
