from numba import njit
import numpy

# I composed this module to be used with this visibility toggle: https://marketplace.visualstudio.com/items?itemName=eliostruyf.vscode-hide-comments
from mapFolding.lovelaceIndices import taskDivisions, taskIndex, leavesTotal, dimensionsTotal # Indices of array `the`. Static integer values
from mapFolding.lovelaceIndices import A, B, count, gapter # Indices of array `track`. Dynamic values; each with length `leavesTotal + 1`

@njit(cache=True, parallel=True, fastmath=False)
def countFoldings(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                  gap: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                  the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                  D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]):
    foldingsTotal: int = 0
    l: int = 1 # activeLeaf1ndex: int = 1
    g: int = 0 # activeGap1ndex: int = 0

    while l > 0: # while activeLeaf1ndex > 0:
        if l <= 1 or track[B][0] == 1: # if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
            if l > the[leavesTotal]: # if activeLeaf1ndex > the[leavesTotal]:
                foldingsTotal += the[leavesTotal]
            else:
                dd: int = 0 # unconstrainedLeaf: int = 0
                # Track possible gaps 
                gg: int = track[gapter][l - 1] # gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1]
                # Reset gap index
                g = gg # activeGap1ndex = gap1ndexLowerBound

                # Count possible gaps for leaf l in each section
                for dimension1ndex in range(1, the[dimensionsTotal] + 1):
                    if D[dimension1ndex][l][l] == l: # connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                        dd += 1 # unconstrainedLeaf += 1
                    else:
                        m = D[dimension1ndex][l][l] # leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        while m != l: # while leaf1ndexConnectee != activeLeaf1ndex:
                            if the[taskDivisions] == 0 or l != the[taskDivisions] or m % the[taskDivisions] == the[taskIndex]: # if the[tasksTotal] == 0 or activeLeaf1ndex != the[tasksTotal] or leaf1ndexConnectee % the[tasksTotal] == the[taskActive]:
                                gap[gg] = m # potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                if track[count][m] == 0: # track[countDimensionsGapped][leaf1ndexConnectee] == 0:
                                    gg += 1 # gap1ndexLowerBound += 1
                                track[count][m] += 1 # track[countDimensionsGapped][leaf1ndexConnectee] += 1
                            m = D[dimension1ndex][l][track[B][m]] # leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]

                # If leaf l is unconstrained in all sections, it can be inserted anywhere
                if dd == the[dimensionsTotal]: # if unconstrainedLeaf == the[dimensionsTotal]:
                    for m in range(l): # for leaf1ndex in range(activeLeaf1ndex):
                        gap[gg] = m # potentialGaps[gap1ndexLowerBound] = leaf1ndex
                        gg += 1 # gap1ndexLowerBound += 1

                # Filter gaps that are common to all sections
                for j in range(g, gg): # for indexMiniGap in range(activeGap1ndex, gap1ndexLowerBound):
                    gap[g] = gap[j] # potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if track[count][gap[j]] == the[dimensionsTotal] - dd: # if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == the[dimensionsTotal] - unconstrainedLeaf:
                        g += 1 # activeGap1ndex += 1
                    # Reset track[count] for next iteration
                    track[count][gap[j]] = 0  # track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0

        # Recursive backtracking steps: is recursive backtracking the same as walking forward?
        while l > 0 and g == track[gapter][l - 1]: # while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            l -= 1 # activeLeaf1ndex -= 1
            track[B][track[A][l]] = track[B][l] # track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            track[A][track[B][l]] = track[A][l] # track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

        if l > 0: # if activeLeaf1ndex > 0:
            g -= 1 # activeGap1ndex -= 1
            track[A][l] = gap[g] # track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            track[B][l] = track[B][track[A][l]] # track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
            track[B][track[A][l]] = l # track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
            track[A][track[B][l]] = l # track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
            # Save current gap index
            track[gapter][l] = g # track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex 
            # Move to next leaf
            l += 1 # activeLeaf1ndex += 1
    return foldingsTotal
