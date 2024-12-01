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
def foldings(mapDimensions: list[int], computationDivisions: int = 0, computationIndex: int = 0, normalFoldings: bool = True) -> int:
    """
    Enumerate map foldings.

    Parameters:
        mapDimensions: dimensions of the map
        computationDivisions: attempt* to split computation into this many parts. (*See `mapFolding.countMinimumParsePoints()`)
        computationIndex: an integer in `range(0, computationDivisions)` to select the part to compute
        normalFoldings: when True, enumerate only normal foldings

    Returns:
        foldingsTotal: total number of foldings
    """

    """
    # Scalars
        leavesTotal 
        dimensionsTotal      
        g                    # Gap counter
        Leaf                 # Current leaf

    # 1-dimension arrays
        mapDimensions         # Input dimensions
        LeafAboveIndices        
        LeafBelowIndices        
        countGapsForLeaf[m]: counts sections with gaps for new leaf l below leaf m
        gapIndexer   
        listAllGaps  
        listAllGaps[gapIndexer[l-1] + j]: holds j-th possible/actual gap for leaf l
        initializerCumulativeProducts      

    # 2-dimension array
        initializerSpatialCoordinates

    # 3-dimension array
        ConnectionMapping

    # Loop Variables/Iterators
        pp     # Element in mapDimensions array when calculating total leaves (n)
        i      # Dimension index (1 to dim)
        m      # Leaf index (1 to n or l)
        j      # Gap array index
        k      # Filtered gap array index

    # Calculated Values
        delta  # Difference between initializerSpatialCoordinates initializerSpatialCoordinates[i][l] and initializerSpatialCoordinates[i][m]
        dd     # Counter for unconstrained dimensions
        gg     # Temporary gap counter/index for current leaf
    """

    leavesTotal, dimensionsTotal, LeafConnectionTracker = makeLeafConnectionTracker(mapDimensions)

    LeafAboveIndices = [0] * (leavesTotal + 1)
    LeafBelowIndices = [0] * (leavesTotal + 1)
    countGapsForLeaf = [0] * (leavesTotal + 1)
    gapIndexer       = [0] * (leavesTotal + 1)
    listAllGaps      = [0] * (leavesTotal * leavesTotal + 1)

    g = 0  # Gap counter
    Leaf = 1

    #   protected void process(final int[] a, final int[] b, final int n) {
    #     mCount += n;
    #   }
#   private boolean isSymmetric(final int[] c, final int delta) {
#     for (int k = 0; k < (c.length - 1) / 2; ++k) {
#       if (c[(delta + k) % c.length] != c[(delta + c.length - 2 - k) % c.length]) {
#         return false;
#       }
#     }
#     return true;
#   }


#   @Override
#   protected void process(final int[] a, final int[] b, final int n) {
#     final int[] c = new int[a.length];
#     int j = 0;
#     for (int k = 0; k < b.length; k++) {
#       c[k] = b[j] - j;
#       j = b[j];
#     }
#     for (int k = 0; k < a.length; ++k) {
#       if (isSymmetric(c, k)) {
#         ++mCount;
#       }
#     }
#   }

    foldingsTotal = 0
    def process(LeafAboveIndices: list[int], LeafBelowIndices: list[int], leavesTotal: int):
        nonlocal foldingsTotal
        foldingsTotal += leavesTotal
    # Main backtrack loop - implements Lunnon's state machine:
    # 1. Try to extend current folding by adding new leaf
    # 2. If no extension possible, backtrack
    # 3. Process completed foldings
    while Leaf > 0:
        if not normalFoldings or Leaf <= 1 or LeafBelowIndices[0] == 1:
            if Leaf > leavesTotal:
                process(LeafAboveIndices, LeafBelowIndices, leavesTotal)
            else:
                dd = 0
                gg = gapIndexer[Leaf - 1]
                g = gg

                # For each dimension, track gaps
                for someDimension in range(1, dimensionsTotal + 1):
                    if LeafConnectionTracker[someDimension][Leaf][Leaf] == Leaf:
                        dd += 1
                    else:
                        someLeaf = LeafConnectionTracker[someDimension][Leaf][Leaf]
                        while someLeaf != Leaf:
                                # The parse point
                            if computationDivisions == 0 or Leaf != computationDivisions or someLeaf % computationDivisions == computationIndex:
                                listAllGaps[gg] = someLeaf
                                countGapsForLeaf[someLeaf] += 1  # Increment count here
                                gg += 1 # Increment gg only if a new gap is added.
                            someLeaf = LeafConnectionTracker[someDimension][Leaf][LeafBelowIndices[someLeaf]]

                # Handle unconstrained case
                if dd == dimensionsTotal:
                    for someLeaf in range(Leaf):
                        listAllGaps[gg] = someLeaf
                        gg += 1

                # Gap filtering and update
                k = g
                for j in range(g, gg):
                    if countGapsForLeaf[listAllGaps[j]] == dimensionsTotal - dd:
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

@njit
def makeLeafConnectionTracker(mapDimensions) -> tuple[int, int, list[list[list[int]]]]:
    leavesTotal = 1
    for dimension in mapDimensions:
        leavesTotal *= dimension
    dimensionsTotal = len(mapDimensions)

    productOfDimensions = [1] * (dimensionsTotal + 1)
    
    for someDimension in range(1, dimensionsTotal + 1):
        productOfDimensions[someDimension] = productOfDimensions[someDimension - 1] * mapDimensions[someDimension - 1]
    
    leafPositionsGrid = [[0] * (leavesTotal + 1) for dimension in range(dimensionsTotal + 1)]

    # Calculate positions
    for someDimension in range(1, dimensionsTotal + 1):
        for someLeaf in range(1, leavesTotal + 1):
            leafPositionsGrid[someDimension][someLeaf] = (
                ((someLeaf - 1) // productOfDimensions[someDimension - 1]) 
                - ((someLeaf - 1) // productOfDimensions[someDimension]) 
                * mapDimensions[someDimension - 1]
                + 1)

    LeafConnectionTracker = [[[0] * (leavesTotal + 1) for leaf in range(leavesTotal + 1)] 
                           for dimension in range(dimensionsTotal + 1)]

    for someDimension in range(1, dimensionsTotal + 1):
        for nexusLeaf in range(1, leavesTotal + 1):
            for someLeaf in range(1, nexusLeaf + 1):
                distance = leafPositionsGrid[someDimension][nexusLeaf] - leafPositionsGrid[someDimension][someLeaf]

                if (distance % 2) == 0:
                    LeafConnectionTracker[someDimension][nexusLeaf][someLeaf] = (
                        someLeaf if leafPositionsGrid[someDimension][someLeaf] == 1
                        else someLeaf - productOfDimensions[someDimension - 1]
                    )
                else:
                    LeafConnectionTracker[someDimension][nexusLeaf][someLeaf] = (
                        someLeaf if (leafPositionsGrid[someDimension][someLeaf] == mapDimensions[someDimension - 1] 
                        or someLeaf + productOfDimensions[someDimension - 1] > nexusLeaf)
                        else someLeaf + productOfDimensions[someDimension - 1]
                    )

    return leavesTotal, dimensionsTotal, LeafConnectionTracker