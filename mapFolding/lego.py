from mapFolding import outfitFoldings, validateTaskDivisions
from typing import List, Final
import numba
import numba.cuda
import numpy

@numba.jit(cache=True, fastmath=False)
def foldings(listDimensions: List[int], computationDivisions=0, computationIndex=0):
    useGPU = False
    # if numba.cuda.is_available():
    #     useGPU = True

    dtypeDefault = numpy.uint8
    dtypeMaximum = numpy.uint16

    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions, dtypeDefault, dtypeMaximum)
    computationDivisions, computationIndex = validateTaskDivisions(computationDivisions, computationIndex, leavesTotal)

    dimensionsTotal = len(listDimensions)

    if useGPU:
        track = numba.cuda.to_device(track)
        potentialGaps = numba.cuda.to_device(potentialGaps)
        connectionGraph = numba.cuda.to_device(connectionGraph)
        leavesTotal = numba.cuda.to_device(leavesTotal)
        dimensionsTotal = numba.cuda.to_device(dimensionsTotal)
        computationDivisions = numba.cuda.to_device(computationDivisions)
        computationIndex = numba.cuda.to_device(computationIndex)

        # This part smells like bullshit to me
        # foldingsTotal = integerLarge(0)
        # foldingsTotal = numba.cuda.to_device(foldingsTotal)

        # # Launch the GPU kernel
        # countFoldings[1, 1](  # 1 block, 1 thread (for now)
        #     track, potentialGaps, connectionGraph, leavesTotal, dimensionsTotal,
        #     computationDivisions, computationIndex)

        # f = foldingsTotal.copy_to_host()[0]
        # return int(f)

    else:
        foldingsTotal = countFoldings(
            track, potentialGaps, connectionGraph, leavesTotal, dimensionsTotal,
            computationDivisions, computationIndex)

        return foldingsTotal

# @cuda.jit
# def countFoldings(
#     track, potentialGaps, connectionGraph, leavesTotal, dimensionsTotal,
#     taskDivisions, taskIndex, foldingsTotal
# ):
#     # Get thread ID for this GPU thread
#     thread_id = numba.cuda.threadIdx.x + numba.cuda.blockIdx.x * numba.cuda.blockDim.x

#     # Ensure only a single thread executes for now (no unnecessary parallelization)
#     if thread_id == 0:
#         # Initialize variables exactly as in your CPU function
#         foldingsTotal[0] = 0  # Use indexable array to store the result on the GPU

@numba.njit(cache=True, fastmath=False)
def countFoldings(track: numpy.ndarray, potentialGaps: numpy.ndarray, D: numpy.ndarray, n, d, computationDivisions, computationIndex):
    def integerSmall(value):
        return numpy.uint8(value)
        # return numba.uint8(value)

    def integerLarge(value):
        return numpy.uint64(value)
        # return numba.uint64(value)

    leafAbove = numba.literally(0)
    leafBelow = numba.literally(1)
    countDimensionsGapped = numba.literally(2)
    gapRangeStart = numba.literally(3)

    connectionGraph: Final = D
    leavesTotal = integerSmall(n)
    dimensionsTotal = integerSmall(d)
    taskDivisions = integerSmall(computationDivisions)
    taskIndex = integerSmall(computationIndex)
    
    foldingsTotal = integerLarge(0)
    activeLeaf1ndex = integerSmall(1)
    activeGap1ndex = integerSmall(0)

    def countGaps(gap1ndexLowerBound, leaf1ndexConnectee):
        if taskDivisions == 0 or activeLeaf1ndex != taskDivisions or leaf1ndexConnectee % taskDivisions == taskIndex:
            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
            if track[countDimensionsGapped][leaf1ndexConnectee] == 0:
                gap1ndexLowerBound += 1
            track[countDimensionsGapped][leaf1ndexConnectee] += 1
        return gap1ndexLowerBound

    def inspectConnectees(gap1ndexLowerBound, dimension1ndex):
        leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
        while leaf1ndexConnectee != activeLeaf1ndex:
            gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
            leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]
        return gap1ndexLowerBound

    def findGaps():
        nonlocal activeGap1ndex

        dimensionsUnconstrained = integerSmall(0)
        gap1ndexLowerBound = track[gapRangeStart][activeLeaf1ndex - 1]
        activeGap1ndex = gap1ndexLowerBound
        dimension1ndex = integerSmall(1) 

        while dimension1ndex <= dimensionsTotal:
            if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                dimensionsUnconstrained += 1
            else:
                gap1ndexLowerBound = inspectConnectees(gap1ndexLowerBound, dimension1ndex)
            dimension1ndex += 1

        return dimensionsUnconstrained, gap1ndexLowerBound

    def insertUnconstrainedLeaf(unconstrainedCount, gapNumberLowerBound):
        # NOTE I suspect this is really an initialization function that should not be in the main loop
        """If activeLeaf1ndex is unconstrained in all dimensions, it can be inserted anywhere"""
        if unconstrainedCount == dimensionsTotal:
            index = integerSmall(0)
            while index < activeLeaf1ndex:
                potentialGaps[gapNumberLowerBound] = index
                gapNumberLowerBound += 1
                index += 1
        return gapNumberLowerBound

    def filterCommonGaps(unconstrainedCount, gapNumberLowerBound) -> None:
        nonlocal activeGap1ndex
        indexMiniGap = activeGap1ndex
        while indexMiniGap < gapNumberLowerBound:
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == dimensionsTotal - unconstrainedCount:
                activeGap1ndex += 1
            track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0
            indexMiniGap += 1

    def backtrack() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
        track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
            if activeLeaf1ndex > leavesTotal:
                foldingsTotal += leavesTotal
            else:
                dimensionsUnconstrained, gap1ndexLowerBound = findGaps()
                gap1ndexLowerBound = insertUnconstrainedLeaf(dimensionsUnconstrained, gap1ndexLowerBound)
                filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)

        backtrack()
        if activeLeaf1ndex > 0:
            placeLeaf()

    return int(foldingsTotal)
