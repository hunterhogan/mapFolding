from typing import List
import numba
import numpy
from mapFolding import outfitFoldings, validateTaskDivisions

leafAbove = 0
leafBelow = 1
countDimensionsGapped = 2
gapRangeStart = 3

def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions)
    computationDivisions, computationIndex = validateTaskDivisions(computationDivisions, computationIndex, leavesTotal)

    dimensionsTotal: int = len(listDimensions)

    foldingsTotal = countFoldings(
        track, potentialGaps, connectionGraph,
        leavesTotal, dimensionsTotal,
        computationDivisions, computationIndex
        )

    return foldingsTotal


@numba.njit(cache=True, fastmath=False)
def countFoldings(
    track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    connectionGraph: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    leavesTotal: int,
    dimensionsTotal: int,
    computationDivisions: int,
    computationIndex: int,
    ) -> int:

    foldingsTotal: int = 0
    activeLeaf1ndex: int = 1
    activeGap1ndex: int = 0

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
            if activeLeaf1ndex > leavesTotal:
                foldingsTotal += leavesTotal
            else:
                unconstrainedLeaf: int = 0
                """Track possible gaps for activeLeaf1ndex in each section"""
                gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1]
                """Reset gap index"""
                activeGap1ndex = gap1ndexLowerBound

                """Count possible gaps for activeLeaf1ndex in each section"""
                for dimension1ndex in range(1, dimensionsTotal + 1):
                    if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                        unconstrainedLeaf += 1
                    else:
                        leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            if computationDivisions == 0 or activeLeaf1ndex != computationDivisions or leaf1ndexConnectee % computationDivisions == computationIndex:
                                potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                if track[countDimensionsGapped][leaf1ndexConnectee] == 0:
                                    gap1ndexLowerBound += 1
                                track[countDimensionsGapped][leaf1ndexConnectee] += 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]

                """If activeLeaf1ndex is unconstrained in all sections, it can be inserted anywhere"""
                if unconstrainedLeaf == dimensionsTotal:
                    for leaf1ndex in range(activeLeaf1ndex):
                        potentialGaps[gap1ndexLowerBound] = leaf1ndex
                        gap1ndexLowerBound += 1

                """Filter gaps that are common to all sections"""
                for indexMiniGap in range(activeGap1ndex, gap1ndexLowerBound):
                    potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == dimensionsTotal - unconstrainedLeaf:
                        activeGap1ndex += 1
                    """Reset track[countDimensionsGapped] for next iteration"""
                    track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0

        """Recursive backtracking steps"""
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

        """Place leaf in valid position"""
        if activeLeaf1ndex > 0:
            activeGap1ndex -= 1
            track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
            """Save current gap index"""
            track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex
            """Move to next leaf"""
            activeLeaf1ndex += 1
    return foldingsTotal
