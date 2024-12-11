import numpy

def foldings(dimensionsMap: list[int], computationDivisions: int = 0, computationIndex: int = 0):
    leavesTotal = 1
    for dimension in dimensionsMap:
        leavesTotal *= dimension
    dimensionsTotal = len(dimensionsMap)

    return _makeDataStructures(dimensionsMap, computationDivisions, computationIndex, leavesTotal, dimensionsTotal)

def _makeDataStructures(dimensionsMap, computationDivisions, computationIndex, leavesTotal, dimensionsTotal):
    static = numpy.zeros(4, dtype=numpy.int64)
    track = numpy.zeros((4, leavesTotal + 1), dtype=numpy.int64)
    gap = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=numpy.int64)

    c = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=numpy.int64)
    bigP = numpy.ones(dimensionsTotal + 1, dtype=numpy.int64)
    leafConnectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int64)
    for i in range(1, dimensionsTotal + 1):
        bigP[i] = bigP[i - 1] * dimensionsMap[i - 1]
    for i in range(1, dimensionsTotal + 1):
        for m in range(1, leavesTotal + 1):
            c[i][m] = (m - 1) // bigP[i - 1] - ((m - 1) // bigP[i]) * dimensionsMap[i - 1] + 1
    for i in range(1, dimensionsTotal + 1):
        for l in range(1, leavesTotal + 1):
            for m in range(1, l + 1):
                delta = c[i][l] - c[i][m]
                if (delta & 1) == 0:
                    leafConnectionGraph[i][l][m] = m if c[i][m] == 1 else m - bigP[i - 1]
                else:
                    leafConnectionGraph[i][l][m] = m if c[i][m] == dimensionsMap[i - 1] or m + bigP[i - 1] > l else m + bigP[i - 1]

    static[0] = leavesTotal
    static[1] = dimensionsTotal
    static[2] = computationDivisions
    static[3] = computationIndex
    from .lovelace import carveInStone
    carveInStone(static, leafConnectionGraph)

    from .lovelace import doWhile
    babayaga = doWhile(track, gap)

    return babayaga
