"""GPU version optimized for multiple independent tasks using CUDA streams."""
from mapFolding import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
import numpy
import numba
import numba.cuda
from numba.cuda.cudadrv.driver import Stream as cudaStream
from typing import List, Tuple

def doWhileGPU(activeGap1ndex: numpy.uint8,
                activeLeaf1ndex: numpy.uint8,
                connectionGraph: numpy.ndarray,
                dimensionsTotal: numpy.uint8,
                leavesTotal: numpy.uint8,
                potentialGaps: numpy.ndarray,
                track: numpy.ndarray) -> numpy.uint64:
    """Launch GPU computation with one stream per task."""

    listStreams: List[cudaStream] = [numba.cuda.stream() for indexStream in range(leavesTotal)]

    cudaConnectionGraph = numba.cuda.to_device(connectionGraph)

    arraySubtotals = numpy.zeros(leavesTotal, dtype=numpy.uint64)
    cudaArraySubtotals = numba.cuda.to_device(arraySubtotals)

    for taskIndex in range(leavesTotal):
        contiguousPotentialGaps = numpy.ascontiguousarray(potentialGaps.copy())
        contiguousTrack = numpy.ascontiguousarray(track.copy())

        gpuPotentialGaps = numba.cuda.to_device(contiguousPotentialGaps, stream=listStreams[taskIndex])
        gpuTrack = numba.cuda.to_device(contiguousTrack, stream=listStreams[taskIndex])

        doWhileKernel[1, 1, listStreams[taskIndex]](
            numpy.uint8(taskIndex),
            activeGap1ndex,
            activeLeaf1ndex,
            cudaConnectionGraph,
            dimensionsTotal,
            cudaArraySubtotals,
            leavesTotal,
            gpuPotentialGaps,
            gpuTrack,
        )

    for taskIndex, gpuStream in enumerate(listStreams):
        gpuStream.synchronize()
        arraySubtotals[taskIndex] = cudaArraySubtotals.copy_to_host()[taskIndex]

    return numpy.sum(arraySubtotals)

@numba.cuda.jit
def doWhileKernel(
    taskIndex: numpy.uint8,
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
    foldsSubtotal = numpy.uint64(0)

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
            if activeLeaf1ndex > leavesTotal:
                foldsSubtotal = foldsSubtotal + leavesTotal
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

    arraySubtotals[taskIndex] = foldsSubtotal
