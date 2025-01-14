from mapFolding.lolaRun import doWhileConcurrent as concurrent, doWhile as one
import numpy
import numba
import numpy.typing

@numba.jit(nopython=True, cache=True, fastmath=True)
def doWhile(connectionGraph: numpy.ndarray, dimensionsTotal: numpy.uint8, leavesTotal: numpy.uint8, potentialGaps: numpy.ndarray, track: numpy.ndarray, ):
    return one(connectionGraph, dimensionsTotal, leavesTotal, potentialGaps, track)

@numba.jit(nopython=True, cache=True, fastmath=True)
def doWhileConcurrent(connectionGraph: numpy.ndarray, dimensionsTotal: numpy.uint8, leavesTotal: numpy.uint8, potentialGaps: numpy.ndarray, track: numpy.ndarray, taskDivisions: numpy.uint8, ):
    return concurrent(connectionGraph, dimensionsTotal, leavesTotal, potentialGaps, track, taskDivisions)
