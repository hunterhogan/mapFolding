from typing import List
import numpy

def foldings(dimensionsMap: List[int]) -> int:
    """
    Calculate number of ways to fold a map of the dimensions, `dimensionsMap`.

    Parameters:
        dimensionsMap: list of dimensions [n, m ...]

    Returns:
        foldingsTotal: Total number of valid foldings

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
    arrayDimensionsMap = numpy.array(dimensionsMap, dtype=numpy.int64)

    listLeaves = list(range(1, numpy.prod(dimensionsMap) + 1))
    arrayLeaves = numpy.array(listLeaves, dtype=numpy.int64)

    from .lovelace import _makeDataStructures

    return _makeDataStructures(arrayDimensionsMap, arrayLeaves)
