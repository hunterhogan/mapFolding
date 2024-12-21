from mapFolding.benchmarks import recordBenchmarks
from mapFolding.lovelaceIndices import taskDivisions, taskIndex, leavesTotal, dimensionsTotal
from typing import List
import numpy

def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    from mapFolding.beDRY import validateParametersFoldings
    listDimensions, computationDivisions, computationIndex, n, D = validateParametersFoldings(listDimensions, computationDivisions, computationIndex)

    d = len(listDimensions)  # Number of dimensions

    track = numpy.zeros((4, n + 1), dtype=numpy.int64)
    gap = numpy.zeros(n * n + 1, dtype=numpy.int64) # Stack of potential gaps
    static = numpy.zeros(4, dtype=numpy.int64)
    static[taskDivisions] = computationDivisions
    static[taskIndex] = computationIndex
    static[leavesTotal] = n
    static[dimensionsTotal] = d
    # Pass listDimensions and taskDivisions to _sherpa for benchmarking
    foldingsTotal = _sherpa(track, gap, static, D, listDimensions, computationDivisions)
    return foldingsTotal

# @recordBenchmarks()
def _sherpa(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], gap: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], static: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], p: List[int], tasks: int) -> int:
    """Performance critical section that counts foldings.
    
    Parameters:
        track: Array tracking folding state
        gap: Array for potential gaps
        static: Array containing static configuration values
        D: Array of leaf connections
        p: List of dimensions for benchmarking
        tasks: Number of computation divisions for benchmarking
    """
    from mapFolding.lovelace import countFoldings
    foldingsTotal = countFoldings(track, gap, static, D)
    return foldingsTotal
