from mapFolding import outfitFoldings
from numba import prange
from typing import List
import numba
import numba.cuda
import numpy
import numpy.typing

useGPU = False
if numba.cuda.is_available():
    useGPU = True
    import cupy

# On Colab, at least, if this is jitted, it will tell me `numba.cuda.to_device` to_device is not an attribute
# @numba.jit(cache=True, fastmath=False)
def countFolds(listDimensions: List[int], computationDivisions: bool = False):

    # dtypeDefault = numpy.uint8
    # dtypeMaximum = numpy.uint16
    dtypeDefault = numpy.int64
    dtypeMaximum = numpy.int64

    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions, dtypeDefault, dtypeMaximum)
    computationIndex = 0

    dimensionsTotal = len(listDimensions)
    arraySubTotals = numpy.zeros(leavesTotal, dtype=dtypeMaximum)

    if useGPU:
        D = numba.cuda.to_device(connectionGraph)
        n = numba.cuda.to_device(leavesTotal)
        d = numba.cuda.to_device(dimensionsTotal)
        mod = numba.cuda.to_device(int(computationDivisions))
        res = numba.cuda.to_device(computationIndex)
        f = numba.cuda.to_device(arraySubTotals)

        threadsPerBlock = 1
        if computationDivisions:
            # Each worker gets their own copy of mutable data
            trackPerWorker = numpy.tile(track, leavesTotal)
            gapsPerWorker = numpy.tile(potentialGaps, leavesTotal)
            
            s = numba.cuda.to_device(trackPerWorker)
            gap = numba.cuda.to_device(gapsPerWorker)
            
            blocksPerGrid = leavesTotal
        else:
            s = numba.cuda.to_device(track)
            gap = numba.cuda.to_device(potentialGaps)
            blocksPerGrid = 1

        countFoldings[blocksPerGrid, threadsPerBlock](s, gap, D, n, d, mod, res, f)
        foldingsSubTotals = f.copy_to_host()

    else:
        foldingsSubTotals = arraySubTotals.copy()
        # PRANGE IS NOT DOING SHIT
        for computationIndex in prange(leavesTotal if computationDivisions else 1):
            foldingsSubTotals[computationIndex] = countFoldings(track, potentialGaps, connectionGraph, leavesTotal, dimensionsTotal, 
                                                            int(computationDivisions), computationIndex, arraySubTotals)

    foldingsTotal = numpy.sum(foldingsSubTotals).item()
    return foldingsTotal

# @numba.cuda.jit()
# def countFoldingsDivisionsKernel(track, potentialGaps, connectionGraph, n, d, arraySubTotals):
#     computationIndex = numba.cuda.grid(1)
#     if computationIndex >= n[()]:
#         return
# Assume this is Google Colab T4 GPU.
@numba.cuda.jit() if useGPU else numba.jit(nopython=True, cache=True, fastmath=False)
def countFoldings(track: numpy.ndarray, potentialGaps: numpy.ndarray, connectionGraph: numpy.ndarray, n, d, computationDivisions, computationIndex, arraySubTotals: numpy.typing.NDArray[numpy.int64]) :
    def integerSmall(value):
        if useGPU:
            return cupy.int64(value)
        return numpy.int64(value)
        #     return cupy.uint8(value)
        # return numpy.uint8(value)

    def integerLarge(value):
        if useGPU:
            return cupy.int64(value)
        return numpy.int64(value)

    leafAbove, leafBelow, countDimensionsGapped, gapRangeStart = 0, 1, 2, 3
    # leafAbove = numba.literally(0)
    # leafBelow = numba.literally(1)
    # countDimensionsGapped = numba.literally(2)
    # gapRangeStart = numba.literally(3)

    # connectionGraph = D

    if useGPU:
        leavesTotal = n[()]
        dimensionsTotal = d[()]
        taskDivisions = computationDivisions[()]
        if taskDivisions:
            taskIndex = numba.cuda.grid(1)
            if taskIndex >= leavesTotal:
                return
            track = track[..., taskIndex]
            potentialGaps = potentialGaps[..., taskIndex]
        else:
            taskIndex = computationIndex[()]
    else:    
        leavesTotal = integerSmall(n)
        dimensionsTotal = integerSmall(d)
        taskDivisions = integerSmall(computationDivisions)
        taskIndex = integerSmall(computationIndex)
    # activeLeaf1ndex = integerSmall(1)
    # activeGap1ndex = integerSmall(0)

    activeLeaf1ndex = 1
    activeGap1ndex = 0

    def countGaps(gap1ndexLowerBound, leaf1ndexConnectee):
        if taskDivisions == 0 or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
            if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                gap1ndexLowerBound += 1
            track[countDimensionsGapped, leaf1ndexConnectee] += 1
        return gap1ndexLowerBound

    def inspectConnectees(gap1ndexLowerBound, dimension1ndex):
        leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
        while leaf1ndexConnectee != activeLeaf1ndex:
            gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
        return gap1ndexLowerBound

    def findGaps():
        nonlocal activeGap1ndex

        dimensionsUnconstrained = integerSmall(0)
        gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]
        activeGap1ndex = gap1ndexLowerBound
        dimension1ndex = integerSmall(1) 

        while dimension1ndex <= dimensionsTotal:
            if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
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
            if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - unconstrainedCount:
                activeGap1ndex += 1
            track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
            indexMiniGap += 1

    def backtrack() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
            track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
            if activeLeaf1ndex > leavesTotal:
                arraySubTotals[taskIndex] += leavesTotal
            else:
                dimensionsUnconstrained, gap1ndexLowerBound = findGaps()
                gap1ndexLowerBound = insertUnconstrainedLeaf(dimensionsUnconstrained, gap1ndexLowerBound)
                filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)

        backtrack()
        if activeLeaf1ndex > 0:
            placeLeaf()

    # Add explicit return for GPU mode
    if useGPU:
        numba.cuda.threadfence()  # Ensure all writes are visible
        return
    else:
        return arraySubTotals[taskIndex]
