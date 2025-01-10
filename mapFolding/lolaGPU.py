"""GPU version optimized for multiple independent tasks using CUDA streams."""
from mapFolding import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
import numpy
import numba
import numba.cuda
from typing import List, Tuple

def doWhileGPU(activeGap1ndex: numpy.uint8,
                activeLeaf1ndex: numpy.uint8,
                connectionGraph: numpy.ndarray,
                dimensionsTotal: numpy.uint8,
                leavesTotal: numpy.uint8,
                potentialGaps: numpy.ndarray,
                track: numpy.ndarray) -> numpy.uint64:
    """Launch GPU computation with one stream per task."""

    listStreams: List = [
        numba.cuda.stream()
        for streamIndex in range(leavesTotal)
    ]

    cuda_connectionGraph = numba.cuda.to_device(connectionGraph)
    allTasks_potentialGaps = numpy.tile(potentialGaps, int(leavesTotal))
    allTasks_track = numpy.tile(track, int(leavesTotal))

    arraySubtotals = numpy.zeros(leavesTotal, dtype=numpy.uint64)
    cuda_arraySubtotals = numba.cuda.to_device(arraySubtotals)

    for taskIndex in range(leavesTotal):
        doWhileKernel[1, 1, listStreams[taskIndex]](
            activeGap1ndex,
            activeLeaf1ndex,
            cuda_connectionGraph,
            dimensionsTotal,
            cuda_arraySubtotals,
            leavesTotal,
            numba.cuda.to_device(allTasks_potentialGaps[..., taskIndex]),
            numba.cuda.to_device(allTasks_track[..., taskIndex]),
        )

    for taskIndex, cudaStream in enumerate(listStreams):
        cudaStream.synchronize()
        arraySubtotals[taskIndex] = cuda_arraySubtotals[taskIndex].copy_to_host()[0]

    return numpy.sum(arraySubtotals)

@numba.cuda.jit
def doWhileKernel(
    activeGap1ndex: numpy.uint8,
    activeLeaf1ndex: numpy.uint8,
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    arraySubtotals: numpy.ndarray,
    leavesTotal: numpy.uint8,
    potentialGaps: numpy.ndarray,
    track: numpy.ndarray,
                    ):
    """CUDA kernel that processes a single independent task."""
    taskIndex = numba.cuda.grid(1)
    if taskIndex >= arraySubtotals.shape[0]:
        return

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
            if activeLeaf1ndex > leavesTotal:
                arraySubtotals[taskIndex] += leavesTotal
            else:
                dimensionsUnconstrained = 0
                gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]
                dimension1ndex = 1

                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained = dimensionsUnconstrained + 1
                    else:
                        leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            if activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                    gap1ndexLowerBound += 1
                                track[countDimensionsGapped, leaf1ndexConnectee] += 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                    dimension1ndex = dimension1ndex + 1

                indexMiniGap = activeGap1ndex
                while indexMiniGap < gap1ndexLowerBound:
                    potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                        activeGap1ndex += 1
                    track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
                    indexMiniGap += 1
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
            track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]
        if activeLeaf1ndex > 0:
            activeGap1ndex -= 1
            track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
            track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
            track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
            track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
            activeLeaf1ndex += 1
