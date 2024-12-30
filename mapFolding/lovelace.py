from mapFolding.lovelaceIndices import A, B, count, gapter # from mapFolding.lovelaceIndices import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
import numba
import numpy

@numba.njit(cache=True, fastmath=False)
def countFoldings(
    s: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], # track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    gap: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], # potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], # connectionGraph: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    n: int, # leavesTotal: int,
    d: int, # dimensionsTotal: int,
    mod: int, # computationDivisions: int,
    res: int, # computationIndex: int,
    ) -> int:

    foldingsTotal: int = 0
    l: int = 1 # activeLeaf1ndex: int = 1
    g: int = 0 # activeGap1ndex: int = 0

    while l > 0: # while activeLeaf1ndex > 0:
        if l <= 1 or s[B][0] == 1: # if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
            if l > n: # if activeLeaf1ndex > leavesTotal:
                foldingsTotal += n # foldingsTotal += leavesTotal
            else:
                dd: int = 0 # unconstrainedLeaf: int = 0
                """Track possible gaps for leaf l in each section""" # """Track possible gaps for activeLeaf1ndex in each section"""
                gg: int = s[gapter][l - 1] # gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1]
                """Reset gap index"""
                g = gg # activeGap1ndex = gap1ndexLowerBound

                """Count possible gaps for leaf l in each section""" # """Count possible gaps for activeLeaf1ndex in each section"""
                for i in range(1, d + 1): # for dimension1ndex in range(1, dimensionsTotal + 1):
                    if D[i][l][l] == l: # if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                        dd += 1 # unconstrainedLeaf += 1
                    else:
                        m: int = D[i][l][l] # leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        while m != l: # while leaf1ndexConnectee != activeLeaf1ndex:
                            if mod == 0 or l != mod or m % mod == res: # if computationDivisions == 0 or activeLeaf1ndex != computationDivisions or leaf1ndexConnectee % computationDivisions == computationIndex:
                                gap[gg] = m # potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                if s[count][m] == 0: # if track[countDimensionsGapped][leaf1ndexConnectee] == 0:
                                    gg += 1 # gap1ndexLowerBound += 1
                                s[count][m] += 1 # track[countDimensionsGapped][leaf1ndexConnectee] += 1
                            m = D[i][l][s[B][m]] # leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]

                """If leaf l is unconstrained in all sections, it can be inserted anywhere""" # """If activeLeaf1ndex is unconstrained in all sections, it can be inserted anywhere"""
                if dd == d: # if unconstrainedLeaf == dimensionsTotal:
                    for m in range(l): # for leaf1ndex in range(activeLeaf1ndex):
                        gap[gg] = m # potentialGaps[gap1ndexLowerBound] = leaf1ndex
                        gg += 1 # gap1ndexLowerBound += 1

                """Filter gaps that are common to all sections"""
                for j in range(g, gg): # for indexMiniGap in range(activeGap1ndex, gap1ndexLowerBound):
                    gap[g] = gap[j] # potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if s[count][gap[j]] == d - dd: # if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == dimensionsTotal - unconstrainedLeaf:
                        g += 1 # activeGap1ndex += 1
                    """Reset s[count] for next iteration""" # """Reset track[countDimensionsGapped] for next iteration"""
                    s[count][gap[j]] = 0  # track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0

        """Recursive backtracking steps"""
        while l > 0 and g == s[gapter][l - 1]: # while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            l -= 1 # activeLeaf1ndex -= 1
            s[B][s[A][l]] = s[B][l] # track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            s[A][s[B][l]] = s[A][l] # track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

        """Place leaf in valid position"""
        if l > 0: # if activeLeaf1ndex > 0:
            g -= 1 # activeGap1ndex -= 1
            s[A][l] = gap[g] # track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            s[B][l] = s[B][s[A][l]] # track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
            s[B][s[A][l]] = l # track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
            s[A][s[B][l]] = l # track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
            """Save current gap index"""
            s[gapter][l] = g # track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex
            """Move to next leaf"""
            l += 1 # activeLeaf1ndex += 1
    return foldingsTotal
