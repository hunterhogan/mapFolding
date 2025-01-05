from mapFolding import outfitFoldings, getLeavesTotal
from numpy.typing import NDArray
from typing import List

def countFolds(listDimensions: List[int], computationDivisions: bool = False) -> int:
    def backtrack() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeLeaf1ndex -= 1
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

    def checkActiveLeafGreaterThan0():
        return activeLeaf1ndex > 0

    def checkActiveLeafGreaterThanLeavesTotal():
        return activeLeaf1ndex > leavesTotal

    def checkActiveLeafIs1orLess():
        if activeLeaf1ndex <= 1:
            return True
        else:
            return checkLeafBelowSentinelIs1()

    def checkLeafBelowSentinelIs1():
        return track[leafBelow, 0] == 1

    def countGaps():
        potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
        if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
            incrementGap1ndexLowerBound()
        track[countDimensionsGapped, leaf1ndexConnectee] += 1

    def filterCommonGaps() -> None:
        nonlocal indexMiniGap
        indexMiniGap = activeGap1ndex
        while indexMiniGap < gap1ndexLowerBound:
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                incrementActiveGap()
            track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
            incrementIndexMiniGap()

    def incrementActiveGap():
        nonlocal activeGap1ndex
        activeGap1ndex += 1

    def incrementDimensionsUnconstrained():
        nonlocal dimensionsUnconstrained
        dimensionsUnconstrained += 1

    def incrementFoldsTotal():
        nonlocal foldsTotal
        foldsTotal += leavesTotal

    def incrementGap1ndexLowerBound():
        nonlocal gap1ndexLowerBound
        gap1ndexLowerBound += 1

    def incrementIndexMiniGap():
        nonlocal indexMiniGap
        indexMiniGap += 1

    def insertUnconstrainedLeaf():
        nonlocal gap1ndexLowerBound
        indexLeaf = 0
        while indexLeaf < activeLeaf1ndex:
            potentialGaps[gap1ndexLowerBound] = indexLeaf
            gap1ndexLowerBound += 1
            indexLeaf += 1
        return gap1ndexLowerBound

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    def initializeState():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound, dimension1ndex
                    dimensionsUnconstrained= 0
                    gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]
                    activeGap1ndex = gap1ndexLowerBound
                    dimension1ndex = 1
                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            incrementDimensionsUnconstrained()
                        else:
                            nonlocal leaf1ndexConnectee
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                if computationDivisions == False or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                    if not activeLeaf1ndex != leavesTotal and leaf1ndexConnectee % leavesTotal == leavesTotal - 1:
                                        return
                                    countGaps()
                                leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                        dimension1ndex += 1
                    if dimensionsUnconstrained == dimensionsTotal:
                        insertUnconstrainedLeaf()
                    filterCommonGaps()
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()
            if activeGap1ndex > 0:
                return

    def middleGame():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound, dimension1ndex
                    dimensionsUnconstrained= 0
                    gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]
                    activeGap1ndex = gap1ndexLowerBound
                    dimension1ndex = 1
                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            incrementDimensionsUnconstrained()
                        else:
                            nonlocal leaf1ndexConnectee
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                if computationDivisions == False or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                    countGaps()
                                leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                        dimension1ndex += 1
                    if dimensionsUnconstrained == dimensionsTotal:
                        insertUnconstrainedLeaf()
                    filterCommonGaps()
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()

    leafAbove, leafBelow, countDimensionsGapped, gapRangeStart = 0, 1, 2, 3

    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions)
    activeGap1ndex: int = 0
    activeLeaf1ndex: int = 1
    dimension1ndex: int = 0
    dimensionsTotal: int = len(listDimensions)
    dimensionsUnconstrained: int = 0
    gap1ndexLowerBound: int = 0
    indexMiniGap: int = 0
    leaf1ndexConnectee: int = 0
    taskIndex: int = 0

    if computationDivisions:
        listTaskIndices = list(range(leavesTotal))
    else:
        listTaskIndices = [taskIndex]

    foldsTotal: int = 0

    initializeState()

    taskState = (
        activeGap1ndex,
        activeLeaf1ndex,
        # connectionGraph,
        # dimension1ndex,
        # dimensionsTotal,
        # dimensionsUnconstrained,
        # gap1ndexLowerBound,
        # indexMiniGap,
        # leaf1ndexConnectee,
        # leavesTotal,
        # listDimensions,
        potentialGaps.copy(),
        track.copy(),
    )
    print(f"{listTaskIndices=}")
    for index in listTaskIndices:
        taskIndex = index
        (
        activeGap1ndex,
        activeLeaf1ndex,
        # connectionGraph,
        # dimension1ndex,
        # dimensionsTotal,
        # dimensionsUnconstrained,
        # gap1ndexLowerBound,
        # indexMiniGap,
        # leaf1ndexConnectee,
        # leavesTotal,
        # listDimensions,
        potentialGaps,
        track,
            ) = taskState
        middleGame()

    return foldsTotal
