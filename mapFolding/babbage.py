from typing import List
import numpy
from .benchmarks import recordBenchmarks

def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    listDimensions = _validateListDimensions(listDimensions)

    from mapFolding import  getLeavesTotal
    leavesTotal = getLeavesTotal(listDimensions)

    taskDivisions, taskIndex = _validateTaskDivisions(computationDivisions, computationIndex, leavesTotal)

    d = len(listDimensions)  # Number of dimensions
    P = numpy.ones(d + 1, dtype=numpy.int64)
    for i in range(1, d + 1):
        P[i] = P[i - 1] * listDimensions[i - 1]

    # C[i][m] holds the i-th coordinate of leaf m
    C = numpy.zeros((d + 1, leavesTotal + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for m in range(1, leavesTotal + 1):
            C[i][m] = ((m - 1) // P[i - 1]) % listDimensions[i - 1] + 1

    # D[i][l][m] computes the leaf connected to m in section i when inserting l
    D = numpy.zeros((d + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for l in range(1, leavesTotal + 1):
            for m in range(1, l + 1):
                delta = C[i][l] - C[i][m]
                if delta % 2 == 0: # If delta is even
                    if C[i][m] == 1:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m - P[i - 1]
                else: # If delta is odd
                    if C[i][m] == listDimensions[i - 1] or m + P[i - 1] > l:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m + P[i - 1]

    track = numpy.zeros((4, leavesTotal + 1), dtype=numpy.int64)
    gap = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=numpy.int64) # Stack of potential gaps

    from .lovelace import setGlobalValues
    setGlobalValues(taskDivisions, taskIndex, leavesTotal, d, D)

    foldingsTotal = _sherpa(track, gap, listDimensions, taskDivisions)
    return foldingsTotal

@recordBenchmarks()
def _sherpa(track, gap, p, tasks):
    from .lovelace import countFoldings
    # start Performance Critical Area
    foldingsTotal = countFoldings(track, gap)
    # end Performance Critical Area
    return foldingsTotal

def _validateTaskDivisions(computationDivisions, computationIndex, n):
    if computationDivisions > n:
        raise ValueError(f"computationDivisions, {computationDivisions}, must be less than or equal to the total number of leaves, {n}.")
    if computationDivisions > 1 and computationIndex >= computationDivisions:
        raise ValueError(f"computationIndex, {computationIndex}, must be less than computationDivisions, {computationDivisions}.")
    if computationDivisions < 0 or computationIndex < 0 or not isinstance(computationDivisions, int) or not isinstance(computationIndex, int):
        raise ValueError(f"computationDivisions, {computationDivisions}, and computationIndex, {computationIndex}, must be non-negative integers.")
    return computationDivisions, computationIndex

def _validateListDimensions(listDimensions):
    from mapFolding import parseListDimensions
    if listDimensions is None:
        raise ValueError(f"listDimensions is a required parameter.")
    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listPositive = [dimension for dimension in listNonNegative if dimension > 0]
    if len(listPositive) < 2:
        from typing import get_args
        from mapFolding.oeis import OEISsequenceID
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. Other functions in this package implement the sequences {get_args(OEISsequenceID)}. You may want to look at https://oeis.org/.")
    listDimensions = listPositive
    return listDimensions

