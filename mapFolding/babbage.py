from typing import List, Optional
from Z0Z_tools import defineConcurrencyLimit
import numba
import numpy

def foldings(dimensionsMap: List[int], CPUlimit: Optional[int | float | bool] = None) -> int:
    """
    Calculate number of ways to fold a map of the dimensions, `dimensionsMap`.

    Parameters:
        dimensionsMap: list of dimensions [n, m ...]
        CPUlimit: whether and how to limit the CPU usage. See notes for details. 

    Returns:
        foldingsTotal: Total number of valid foldings

    Limits on CPU usage `CPUlimit`:
        - `True`: Yes, limit the CPU usage; limits to 1 CPU.
        - `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs.
        - Integer `>= 1`: Limits usage to the specified number of CPUs.
        - Float `0 < limit < 1`: Fraction of total CPUs to use.
        - Float `-1 < limit < 0`: Subtract a fraction of CPUs from the total.
        - Integer `<= -1`: Subtract the absolute value from total CPUs.

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

    # Set the number of threads for parallel processing
    numba.set_num_threads(defineConcurrencyLimit(CPUlimit))

    from .lovelace import _makeDataStructures

    return _makeDataStructures(arrayDimensionsMap, arrayLeaves)
