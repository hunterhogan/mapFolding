from mapFolding.benchmarks import recordBenchmarks
from mapFolding.lovelaceIndices import taskDivisions, taskIndex, leavesTotal, dimensionsTotal
# from mapFolding.piderIndices import taskDivisions, taskIndex, leavesTotal, dimensionsTotal
# taskDivisions, taskIndex, leavesTotal, dimensionsTotal = int(taskDivisions), int(taskIndex), int(leavesTotal), int(dimensionsTotal)
from typing import List
import numpy

def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    from mapFolding.beDRY import validateParametersFoldings
    listDimensions, computationDivisions, computationIndex, n, D = validateParametersFoldings(listDimensions, computationDivisions, computationIndex)

    d = len(listDimensions)  # Number of dimensions
    # P = numpy.ones(d + 1, dtype=numpy.int64)
    # for i in range(1, d + 1):
    #     P[i] = P[i - 1] * listDimensions[i - 1]

    # # C[i][m] holds the i-th coordinate of leaf m
    # C = numpy.zeros((d + 1, n + 1), dtype=numpy.int64)
    # for i in range(1, d + 1):
    #     for m in range(1, n + 1):
    #         C[i][m] = ((m - 1) // P[i - 1]) % listDimensions[i - 1] + 1

    # # D[i][l][m] computes the leaf connected to m in section i when inserting l
    # D = numpy.zeros((d + 1, n + 1, n + 1), dtype=numpy.int64)
    # for i in range(1, d + 1):
    #     for l in range(1, n + 1):
    #         for m in range(1, l + 1):
    #             delta = C[i][l] - C[i][m]
    #             if delta % 2 == 0: # If delta is even
    #                 if C[i][m] == 1:
    #                     D[i][l][m] = m
    #                 else:
    #                     D[i][l][m] = m - P[i - 1]
    #             else: # If delta is odd
    #                 if C[i][m] == listDimensions[i - 1] or m + P[i - 1] > l:
    #                     D[i][l][m] = m
    #                 else:
    #                     D[i][l][m] = m + P[i - 1]

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
    # from mapFolding.pider import countFoldings
    foldingsTotal = countFoldings(track, gap, static, D)
    return foldingsTotal
