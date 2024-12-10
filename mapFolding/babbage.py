from numba import njit
import numpy
from .lovelace import doWhile
import cProfile

def foldings(dimensionsMap: list[int], computationDivisions: int = 0, computationIndex: int = 0):
    foldingsTotal = 0

    n = 1
    for dimension in dimensionsMap:
        n *= dimension
    ndim = len(dimensionsMap)

    a = numpy.zeros(n + 1, dtype=numpy.int64)
    b = numpy.zeros(n + 1, dtype=numpy.int64)
    count = numpy.zeros(n + 1, dtype=numpy.int64)
    gapter = numpy.zeros(n + 1, dtype=numpy.int64)
    gap = numpy.zeros(n * n + 1, dtype=numpy.int64)
    bigP = numpy.ones(ndim + 1, dtype=numpy.int64)
    c = numpy.zeros((ndim + 1, n + 1), dtype=numpy.int64)
    
    leafConnectionGraph = numpy.zeros((ndim + 1, n + 1, n + 1), dtype=numpy.int64)
    for i in range(1, ndim + 1):
        bigP[i] = bigP[i - 1] * dimensionsMap[i - 1]
    for i in range(1, ndim + 1):
        for m in range(1, n + 1):
            c[i][m] = (m - 1) // bigP[i - 1] - ((m - 1) // bigP[i]) * dimensionsMap[i - 1] + 1
    for i in range(1, ndim + 1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = c[i][l] - c[i][m]
                if (delta & 1) == 0:
                    leafConnectionGraph[i][l][m] = m if c[i][m] == 1 else m - bigP[i - 1]
                else:
                    leafConnectionGraph[i][l][m] = m if c[i][m] == dimensionsMap[i - 1] or m + bigP[i - 1] > l else m + bigP[i - 1]

    foldingsTotal = doWhile(computationDivisions, computationIndex, foldingsTotal, n, a, b, count, gapter, gap, ndim, leafConnectionGraph)

    return foldingsTotal
