from mapFolding import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
import numba
import numpy

def saveJob(pathJob: str,
    activeGap1ndex: numpy.uint8,
    activeLeaf1ndex: numpy.uint8,
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    leavesTotal: numpy.uint8,
    potentialGaps: numpy.ndarray,
    track: numpy.ndarray):
    import pathlib
    pathFilenameJob = pathlib.Path(pathJob, "stateJob.npz")
    pathFilenameJob.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(pathFilenameJob,
        activeGap1ndex=activeGap1ndex,
        activeLeaf1ndex=activeLeaf1ndex,
        connectionGraph=connectionGraph,
        dimensionsTotal=dimensionsTotal,
        leavesTotal=leavesTotal,
        potentialGaps=potentialGaps,
        track=track
    )
    return -1

def doJob(pathFilenameJob):
    foldsTotal = doWhileOne(
        activeGap1ndex = numpy.uint8(numpy.load(pathFilenameJob)['activeGap1ndex']),
        activeLeaf1ndex = numpy.uint8(numpy.load(pathFilenameJob)['activeLeaf1ndex']),
        connectionGraph = numpy.load(pathFilenameJob)['connectionGraph'],
        dimensionsTotal = numpy.uint8(numpy.load(pathFilenameJob)['dimensionsTotal']),
        leavesTotal = numpy.uint8(numpy.load(pathFilenameJob)['leavesTotal']),
        potentialGaps = numpy.load(pathFilenameJob)['potentialGaps'],
        track = numpy.load(pathFilenameJob)['track']
        )
    print(foldsTotal)
    import pathlib
    pathlib.Path(pathFilenameJob).with_name("foldsTotal.txt").write_text(str(foldsTotal))

@numba.jit(nopython=True, cache=True, fastmath=True)
def doWhileOne(activeGap1ndex: numpy.uint8,
    activeLeaf1ndex: numpy.uint8,
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    leavesTotal: numpy.uint8,
    potentialGaps: numpy.ndarray,
    track: numpy.ndarray):

    foldsTotal = numpy.uint64(0)
    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
            if activeLeaf1ndex > leavesTotal:
                foldsTotal = foldsTotal + leavesTotal
            else:
                dimensionsUnconstrained = numpy.uint8(0)
                gap1ndexLowerBound: numpy.uint8 = track[gapRangeStart, activeLeaf1ndex - 1]
                dimension1ndex = numpy.uint8(1)
                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained += 1
                    else:
                        leaf1ndexConnectee: numpy.uint8 = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                            if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                gap1ndexLowerBound += 1
                            track[countDimensionsGapped, leaf1ndexConnectee] += 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                    dimension1ndex += 1
                indexMiniGap: numpy.uint8 = activeGap1ndex
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
    return foldsTotal
