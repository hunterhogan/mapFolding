"""
The module enumerates all possible ways to fold a multi-dimensional map with given dimensions.

Implements algorithm from
W. F. Lunnon, Multi-dimensional map-folding, "The Computer Journal", Volume 14, Issue 1, 1971, Pages 75-80,
https://doi.org/10.1093/comjnl/14.1.75 (see also "./citations/Lunnon.bibtex")
but directly based on Sean Irvine's Java port of Fred Lunnon's C version.

See https://github.com/archmageirvine/joeis/blob/80e3e844b11f149704acbab520bc3a3a25ac34ff/src/irvine/oeis/a001/A001415.java
"""
from numba import njit

@njit
def foldings(mapShape: list[int], computationDivisions: int = 0, computationIndex: int = 0, normalFoldings: bool = True) -> int:
    """
    Enumerate map foldings.

    Parameters:
        mapShape: dimensions of the map
        computationDivisions: attempt* to split computation into this many parts. (*See `mapFolding.countMinimumParsePoints()`)
        computationIndex: an integer in `range(0, computationDivisions)` to select the part to compute
        normalFoldings: when True, enumerate only normal foldings

    Returns:
        foldingsTotal: total number of foldings
    """

    """
    # Scalars
        mapCellsQuantity 
        mapAxesQuantity      # if mapShape were ndarray, then == `.ndim`
        g                    # Gap counter
        Leaf                 # Current leaf

    # 1-axis arrays
        mapShape         # Input dimensions
        LeafAboveIndices        
        LeafBelowIndices        
        countGapsForLeaf[m]: counts sections with gaps for new leaf l below leaf m
        gapIndexer   
        listAllGaps  
        listAllGaps[gapIndexer[l-1] + j]: holds j-th possible/actual gap for leaf l
        cumulativeProducts      

    # 2-axis array
        coordinates

    # 3-axis array
        ConnectionMapping

    # Loop Variables/Iterators
        pp     # Element in mapShape array when calculating total leaves (n)
        i      # Dimension index (1 to dim)
        m      # Leaf index (1 to n or l)
        j      # Gap array index
        k      # Filtered gap array index

    # Calculated Values
        delta  # Difference between coordinates coordinates[i][l] and coordinates[i][m]
        dd     # Counter for unconstrained dimensions
        gg     # Temporary gap counter/index for current leaf
    """
    foldingsTotal = 0

    mapCellsQuantity = 1
    for axis in mapShape:
        mapCellsQuantity *= axis

    LeafAboveIndices = [0] * (mapCellsQuantity + 1)
    LeafBelowIndices = [0] * (mapCellsQuantity + 1)
    countGapsForLeaf = [0] * (mapCellsQuantity + 1)
    gapIndexer       = [0] * (mapCellsQuantity + 1)
    listAllGaps      = [0] * (mapCellsQuantity * mapCellsQuantity + 1)

    mapAxesQuantity = len(mapShape)
    cumulativeProducts = [1] * (mapAxesQuantity + 1)
    coordinates        = [[0] * (mapCellsQuantity + 1) for axis in range(mapAxesQuantity + 1)]
    ConnectionMapping  = [[[0] * (mapCellsQuantity + 1) for cell in range(mapCellsQuantity + 1)] for axis in range(mapAxesQuantity + 1)]

    for i in range(1, mapAxesQuantity + 1):
        cumulativeProducts[i] = cumulativeProducts[i - 1] * mapShape[i - 1]

    for i in range(1, mapAxesQuantity + 1):
        for m in range(1, mapCellsQuantity + 1):
            coordinates[i][m] = (m - 1) // cumulativeProducts[i - 1] - ((m - 1) // cumulativeProducts[i]) * mapShape[i - 1] + 1

    for i in range(1, mapAxesQuantity + 1):
        for Leaf in range(1, mapCellsQuantity + 1):
            for m in range(1, Leaf + 1):
                delta = coordinates[i][Leaf] - coordinates[i][m]
                if (delta & 1) == 0:
                    ConnectionMapping[i][Leaf][m] = m if coordinates[i][m] == 1 else m - cumulativeProducts[i - 1]
                else:
                    ConnectionMapping[i][Leaf][m] = m if coordinates[i][m] == mapShape[i - 1] or m + cumulativeProducts[i - 1] > Leaf else m + cumulativeProducts[i - 1]

    g = 0  # Gap counter
    Leaf = 1

    # Main backtrack loop - implements Lunnon's state machine:
    # 1. Try to extend current folding by adding new leaf
    # 2. If no extension possible, backtrack
    # 3. Process completed foldings
    while Leaf > 0:
        if not normalFoldings or Leaf <= 1 or LeafBelowIndices[0] == 1:
            if Leaf > mapCellsQuantity:
                foldingsTotal += mapCellsQuantity
            else:
                dd = 0
                gg = gapIndexer[Leaf - 1]
                g = gg

                # For each dimension, track gaps
                for i in range(1, mapAxesQuantity + 1):
                    if ConnectionMapping[i][Leaf][Leaf] == Leaf:
                        dd += 1
                    else:
                        m = ConnectionMapping[i][Leaf][Leaf]
                        while m != Leaf:
                                # The parse point
                            if computationDivisions == 0 or Leaf != computationDivisions or m % computationDivisions == computationIndex:
                                listAllGaps[gg] = m
                                countGapsForLeaf[m] += 1  # Increment count here
                                gg += 1 # Increment gg only if a new gap is added.
                            m = ConnectionMapping[i][Leaf][LeafBelowIndices[m]]

                # Handle unconstrained case
                if dd == mapAxesQuantity:
                    for m in range(Leaf):
                        listAllGaps[gg] = m
                        gg += 1

                # Gap filtering and update
                k = g
                for j in range(g, gg):
                    if countGapsForLeaf[listAllGaps[j]] == mapAxesQuantity - dd:
                        listAllGaps[k] = listAllGaps[j]
                        k += 1
                    countGapsForLeaf[listAllGaps[j]] = 0
                g = k # Update g

        # Backtrack when no more gaps
        while Leaf > 0 and g == gapIndexer[Leaf - 1]:
            Leaf -= 1
            if Leaf > 0:
                LeafBelowIndices[LeafAboveIndices[Leaf]] = LeafBelowIndices[Leaf]
                LeafAboveIndices[LeafBelowIndices[Leaf]] = LeafAboveIndices[Leaf]

        # Make next move if possible
        if Leaf > 0:
            g -= 1
            LeafAboveIndices[Leaf] = listAllGaps[g]
            LeafBelowIndices[Leaf] = LeafBelowIndices[LeafAboveIndices[Leaf]]
            LeafBelowIndices[LeafAboveIndices[Leaf]] = Leaf
            LeafAboveIndices[LeafBelowIndices[Leaf]] = Leaf
            gapIndexer[Leaf] = g
            Leaf += 1
    return foldingsTotal
