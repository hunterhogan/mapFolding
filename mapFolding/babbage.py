"""
Key concepts
    - A "leaf" is a unit square in the map
    - A "gap" is a potential position where a new leaf can be folded
    - Connections track how leaves can connect above/below each other
    - The algorithm builds foldings incrementally by placing one leaf at a time
    - Backtracking explores all valid combinations
    - Leaves and dimensions are enumerated starting from 1, not 0; hence, leaf1ndex not leafIndex

Algorithm flow
    For each leaf
        - Find valid gaps in each dimension
        - Place leaf in valid position
            - Try to find another lead to put in the adjacent position
            - Repeat until the map is completely folded
        - Backtrack when no valid positions remain

Identifiers
    This module has two sets of identifiers. One set is active, and the other set is in uniformly formatted comments
    at the end of every line that includes an identifier that has an alternative identifier. 

    First, that might be distracting. In Visual Studio Code, the following extension will hide all comments but not docstrings:
    https://marketplace.visualstudio.com/items?itemName=eliostruyf.vscode-hide-comments

    Second, you can swap the sets of identifiers or delete one set of identifiers permanently.

    Step 1: regex find:
"""
# ^(?! *#)( *?)(\S.+?)( # )(.+) # This line is a comment and not a docstring because the Python interpreter handles `\S` better in a comment
"""
    Step 2: choose a regex replace option:
        A) To SWAP the sets of identifiers
        $1$4$3$2
        B) To PERMANENTLY replace the active set of identifiers
        $1$4
        C) To PERMANENTLY delete the inactive set of identifiers, which are in the comments
        $1$2

    Equivalent identifiers:
    A = leafAbove
    B = leafBelow
    C = coordinateSystem
    count = countDimensionsGapped
    D = connectionGraph
    d = dimensionsTotal
    dd = unconstrainedLeaf
    delta = distance
    g = activeGap1ndex
    gap = potentialGaps
    gapter = gapRangeStart
    gg = gap1ndexLowerBound
    i = dimension1ndex
    j = indexMiniGap
    l = activeLeaf1ndex
    m = leaf1ndex or leaf1ndexConnectee
    mod = computationDivisions
    n = leavesTotal
    P = cumulativeProduct
    p = listDimensions
    res = computationIndex
    s = track
"""
from typing import List
import numpy

# The following functions are used by the test modules
from mapFolding import validateListDimensions, getLeavesTotal, validateTaskDivisions

def foldings(p: List[int], mod: int = 0, res: int = 0) -> int: # def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    listDimensionsPositive = validateListDimensions(p) # listDimensionsPositive = validateListDimensions(listDimensions)

    n: int = getLeavesTotal(listDimensionsPositive) # leavesTotal: int = getLeavesTotal(listDimensionsPositive)

    mod, res = validateTaskDivisions(mod, res, n) # computationDivisions, computationIndex = validateTaskDivisions(computationDivisions, computationIndex, leavesTotal)

    d: int = len(p) # dimensionsTotal: int = len(listDimensions)

    # I am quite frustrated by Python's namespace system. I put this here because I am overly cautious.
    from mapFolding.beDRY import makeConnectionGraph
    D = makeConnectionGraph(p) # connectionGraph = makeConnectionGraph(listDimensions)

    """For numba, a single array is faster than four separate arrays"""
    # I don't like that `4` is hardcoded instead of dynamically calculated, but I haven't figured out a clever way to handle it.
    s = numpy.zeros((4, n + 1), dtype=numpy.int64) # track = numpy.zeros((4, leavesTotal + 1), dtype=numpy.int64)

    gap = numpy.zeros(n * n + 1, dtype=numpy.int64) # potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=numpy.int64)

    from mapFolding.lovelace import countFoldings
    foldingsTotal = countFoldings(
        s, gap, D, # track, potentialGaps, connectionGraph,
        n, d, # leavesTotal, dimensionsTotal,
        mod, res # computationDivisions, computationIndex
        )

    return foldingsTotal
