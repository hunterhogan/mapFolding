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
make `track` (4, the[leavesTotal]+1)      38.87   199.42    2043.73 1064.41 1254.42
change to measuring `doWhile`        36.44   176.10    1941.60 1046.56 1242.60
Then, something broke and I couldn't figure out how to profile `doWhile` again.
"""
# Static values
the = numpy.array(0, dtype=numpy.int64)
# `the` indices
leavesTotal = 0
dimensionsTotal = 1
tasksTotal = 2
taskActive = 3

leafConnectionGraph = numpy.array(0, dtype=numpy.int64)

# Dynamic values
# `track` indices
above = 0
below = 1
count = 2
gapter = 3

# `an` indices
incompleteTotal = 0
activeGap1ndex = 1 # index starts at 0, but 1ndex starts at 1
activeLeaf1ndex = 2 # index starts at 0, but 1ndex starts at 1
unconstrainedLeaf = 3
eniggma = 4 # gg in the original code
effingAIeffedMyCode = 5
leaf1ndex = 6 # index starts at 0, but 1ndex starts at 1

def carveInStone(static, graph):
    global the, leafConnectionGraph
    the = static
    leafConnectionGraph = graph

# I think cache is a bad idea with global constants.
@njit(cache=False, parallel=False, nogil=True, fastmath=True, boundscheck=False, debug=False)
def doWhile(track, gap):
    an = numpy.zeros(7, dtype=numpy.int64)
    an[activeLeaf1ndex] = 1
    while an[activeLeaf1ndex] > 0:
        if an[activeLeaf1ndex] <= 1 or track[below][0] == 1:
            if an[activeLeaf1ndex] > the[leavesTotal]:
                an[incompleteTotal] += the[leavesTotal]
            else:
                an[unconstrainedLeaf] = 0
                an[eniggma] = track[gapter][an[activeLeaf1ndex] - 1]
                an[activeGap1ndex] = an[eniggma]
                for dimension1ndex in range(1, the[dimensionsTotal] + 1):
                    if leafConnectionGraph[dimension1ndex][an[activeLeaf1ndex]][an[activeLeaf1ndex]] == an[activeLeaf1ndex]:
                        an[unconstrainedLeaf] += 1
                    else:
                        an[leaf1ndex] = leafConnectionGraph[dimension1ndex][an[activeLeaf1ndex]][an[activeLeaf1ndex]]
                        while an[leaf1ndex] != an[activeLeaf1ndex]:
                            if the[tasksTotal] == 0 or an[activeLeaf1ndex] != the[tasksTotal] or an[leaf1ndex] % the[tasksTotal] == the[taskActive]:
                                gap[an[eniggma]] = an[leaf1ndex]
                                track[count][an[leaf1ndex]] += 1
                                an[eniggma] += 1
                            an[leaf1ndex] = leafConnectionGraph[dimension1ndex][an[activeLeaf1ndex]][track[below][an[leaf1ndex]]]
                if an[unconstrainedLeaf] == the[dimensionsTotal]:
                    for an[leaf1ndex] in range(an[activeLeaf1ndex]):
                        gap[an[eniggma]] = an[leaf1ndex]
                        an[eniggma] += 1
                an[effingAIeffedMyCode] = an[activeGap1ndex]
                for gapRelated in range(an[activeGap1ndex], an[eniggma]):
                    if track[count][gap[gapRelated]] == the[dimensionsTotal] - an[unconstrainedLeaf]:
                        gap[an[effingAIeffedMyCode]] = gap[gapRelated]
                        an[effingAIeffedMyCode] += 1
                    track[count][gap[gapRelated]] = 0
                an[activeGap1ndex] = an[effingAIeffedMyCode]
        while an[activeLeaf1ndex] > 0 and an[activeGap1ndex] == track[gapter][an[activeLeaf1ndex] - 1]:
            an[activeLeaf1ndex] -= 1
            if an[activeLeaf1ndex] > 0:
                track[below][track[above][an[activeLeaf1ndex]]] = track[below][an[activeLeaf1ndex]]
                track[above][track[below][an[activeLeaf1ndex]]] = track[above][an[activeLeaf1ndex]]
        if an[activeLeaf1ndex] > 0:
            an[activeGap1ndex] -= 1
            track[above][an[activeLeaf1ndex]] = gap[an[activeGap1ndex]]
            track[below][an[activeLeaf1ndex]] = track[below][track[above][an[activeLeaf1ndex]]]
            track[below][track[above][an[activeLeaf1ndex]]] = an[activeLeaf1ndex]
            track[above][track[below][an[activeLeaf1ndex]]] = an[activeLeaf1ndex]
            track[gapter][an[activeLeaf1ndex]] = an[activeGap1ndex]
            an[activeLeaf1ndex] += 1
    foldingsTotal = an[incompleteTotal]
    return foldingsTotal
