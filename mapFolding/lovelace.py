from numba import njit
import numpy
"""
Key concepts
    - A "leaf" is a unit square in the map
    - A "gap" is a potential position where a new leaf can be folded
    - Connections track how leaves can connect above/below each other
    - The algorithm builds foldings incrementally by placing one leaf at a time
    - Backtracking explores all valid combinations
    - Leaves and dimensions are enumerated starting from 1, not 0; hence, leaf1ndex not leafIndex

Algorithm flow
    For each leaf
        - Find valid gaps in each dimension
        - Place leaf in valid position
            - Try to find another lead to put in the adjacent position
            - Repeat until the map is completely folded
        - Backtrack when no valid positions remain

Identifiers
This module has two sets of identifiers. One set is active, and the other set is uniformly formatted comments 
at the end of every line that includes an identifier that has an alternative identifier. First, that might be 
distracting. In Visual Studio Code, the following extension will hide all comments but not docstrings: 
https://marketplace.visualstudio.com/items?itemName=eliostruyf.vscode-hide-comments

Second, you can swap the sets of identifiers or delete one set of identifiers permanently. See the regex instructions below.

Third, the alternatives probably depend on statements that are imported from the module that defines the indices.
For example, that module might include `A = leafAbove = 0`. If those statements don't align with the sets of identifiers
in this module, swapping will break in unpredictable ways.

Identifier annotations
One reason some variable identifers are defined in another module is because VS Code is more likely to display
the variable annotations if the identifiers are imported.
"""
# NOTE: To modify the sets of identifiers: 
# Step 1: regex find 
    # ^(?!#)( *?)(\S.+?)( # )(.+)

# Step 2: choose a regex replace option
    # A) To SWAP the sets of identifiers
    # $1$4$3$2
    # B) To PERMANENTLY replace the active set of identifiers
    # $1$4
    # C) To PERMANENTLY delete the inactive set of identifiers, which are in the comments
    # $1$2


# Indices of array `the`, which holds unchanging, small, unsigned, integer values.
from mapFolding.lovelaceIndices import taskDivisions, taskIndex, leavesTotal, dimensionsTotal 
# Indices of array `track`, which is a collection of one-dimensional arrays each of length `leavesTotal + 1`. 
# The values in the array cells are dynamic, small, unsigned integers.
from mapFolding.lovelaceIndices import A, B, count, gapter # from mapFolding.lovelaceIndices import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart

# numba warnings say there is nothing to parallelize here
# @njit(cache=True, parallel=True, fastmath=False)
@njit(cache=True, fastmath=False)
def countFoldings(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                    gap: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], # potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                    D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]): # connectionGraph: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]):
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
                for dimension1ndex in range(the[dimensionsTotal], 0, -1):
                # for dimension1ndex in range(1, the[dimensionsTotal] + 1):
                    if D[dimension1ndex][l][l] == l: # if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                        dd += 1 # unconstrainedLeaf += 1
                    else:
                        m: int = D[dimension1ndex][l][l] # leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        while m != l: # while leaf1ndexConnectee != activeLeaf1ndex:
                            if the[taskDivisions] == 0 or l != the[taskDivisions] or m % the[taskDivisions] == the[taskIndex]: # if the[taskDivisions] == 0 or activeLeaf1ndex != the[taskDivisions] or leaf1ndexConnectee % the[taskDivisions] == the[taskIndex]:
                                gap[gg] = m # potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                if track[count][m] == 0: # if track[countDimensionsGapped][leaf1ndexConnectee] == 0:
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

        # Recursive backtracking steps
        while l > 0 and g == track[gapter][l - 1]: # while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            l -= 1 # activeLeaf1ndex -= 1
            track[B][track[A][l]] = track[B][l] # track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            track[A][track[B][l]] = track[A][l] # track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

        # Place leaf in valid position
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
