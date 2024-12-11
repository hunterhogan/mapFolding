from mapFolding import getLeavesTotal
from typing import List
import numpy

def foldings(dimensionsMap: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    """
    Calculate number of ways to fold a map of the dimensions, `dimensionsMap`.

    Parameters:
        dimensionsMap:
            list of dimensions [n, m ...]. All dimensions must be non-negative integers. At least two
            dimensions must be greater than 0.
        computationDivisions (0):
            divide the computation into `computationDivisions` parts. The value must
            be less than or equal to the total number of leaves. (The total number of leaves is the product of
            all dimensions of the map.)
        computationIndex (0):
            the index of the computation part to calculate. The sum of all indices from 0 to `computationDivisions` - 1
            is equal to the total number of valid foldings. The value of `computationIndex` must be less than `computationDivisions`.

    Returns:
        foldingsTotal:
            Total number of valid foldings

    Key concepts
        - A "leaf" is a unit square in the map
        - A "gap" is a potential position where a new leaf can be folded
        - Connections track how leaves can connect above/below each other
        - The algorithm builds foldings incrementally by placing one leaf at a time
        - Backtracking explores all valid combinations
        - Leaves and dimensions are enumerated starting from 1, not 0; hence, leafNumber not leafIndex

    Key data structures
        - mapFoldingConnections[D][L][M]: How leaf L connects to leaf M in dimension D
        - listCountDimensionsWithGap[L]: Number of dimensions with valid gaps at leaf L
        - listGapIndicesRange[L]: Index ranges of gaps available for leaf L
        - listPotentialGapPositions[]: List of all potential gap positions

    Algorithm flow
        1. Initialize coordinate system and connection graph
        2. For each leaf:
            - Find valid gaps in each dimension
            - Place leaf in valid position
            - Backtrack when no valid positions remain
        3. Count total valid foldings found
    """
    if dimensionsMap is None:
        raise ValueError(f"dimensionsMap is a required parameter.")
    else:
        dimensionsMap = sorted(dimensionsMap)

    if not all(isinstance(dimension, int) and dimension >= 0 for dimension in dimensionsMap):
        raise ValueError(f"dimensionsMap, {dimensionsMap}, must have non-negative integers as dimensions.")
    else:
        dimensionsTotal = len(dimensionsMap)

    if any(dimension == 0 for dimension in dimensionsMap) or dimensionsTotal < 2:
        dimensionsNonZero = [dimension for dimension in dimensionsMap if dimension > 0]
        if len(dimensionsNonZero) < 2:
            from mapFolding import OEISsequenceID
            from typing import get_args
            raise NotImplementedError(f"This function requires dimensionsMap, {dimensionsMap}, to have at least two dimensions greater than 0. Other functions in this package implement the sequences {get_args(OEISsequenceID)}. You may want to look at https://oeis.org/.")
        else:
            dimensionsMap = dimensionsNonZero

    leavesTotal = getLeavesTotal(dimensionsMap)

    if computationDivisions > leavesTotal:
        raise ValueError(f"computationDivisions, {computationDivisions}, must be less than or equal to the total number of leaves, {leavesTotal}.")
    if computationDivisions > 1 and computationIndex >= computationDivisions:
        raise ValueError(f"computationIndex, {computationIndex}, must be less than computationDivisions, {computationDivisions}.")
    if computationDivisions < 0 or computationIndex < 0 or not isinstance(computationDivisions, int) or not isinstance(computationIndex, int):
        raise ValueError(f"computationDivisions, {computationDivisions}, and computationIndex, {computationIndex}, must be non-negative integers.")

    return _makeDataStructures(dimensionsMap, computationDivisions, computationIndex, leavesTotal, dimensionsTotal)

def _makeDataStructures(
    dimensionsMap: List[int],
    computationDivisions: int,
    computationIndex: int,
    leavesTotal: int,
    dimensionsTotal: int
) -> int:
    static: numpy.ndarray = numpy.zeros(4, dtype=numpy.int64)
    track: numpy.ndarray = numpy.zeros((4, leavesTotal + 1), dtype=numpy.int64)
    gap: numpy.ndarray = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=numpy.int64)

    c: numpy.ndarray = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=numpy.int64)
    bigP: numpy.ndarray = numpy.ones(dimensionsTotal + 1, dtype=numpy.int64)
    leafConnectionGraph: numpy.ndarray = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int64)

    for i in range(1, dimensionsTotal + 1):
        bigP[i] = bigP[i - 1] * dimensionsMap[i - 1]

    for i in range(1, dimensionsTotal + 1):
        for m in range(1, leavesTotal + 1):
            c[i][m] = (m - 1) // bigP[i - 1] - ((m - 1) // bigP[i]) * dimensionsMap[i - 1] + 1

    for i in range(1, dimensionsTotal + 1):
        for l in range(1, leavesTotal + 1):
            for m in range(1, l + 1):
                delta: int = c[i][l] - c[i][m]
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
    foldingsTotal: int = doWhile(track, gap)

    return foldingsTotal
