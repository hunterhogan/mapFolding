import sys
from typing import List, Tuple

import numpy
from Z0Z_tools import intInnit


def parseListDimensions(listDimensions: List[int], parameterName: str = 'unnamed parameter') -> List[int]:
    """
    Parse and validate a list of dimensions.

    Parameters:
        listDimensions: List of integers representing dimensions
        parameterName ('unnamed parameter'): Name of the parameter for error messages. Defaults to 'unnamed parameter'
    Returns:
        listNonNegative: List of validated non-negative integers
    Raises:
        ValueError: If any dimension is negative or if the list is empty
        TypeError: If any element cannot be converted to integer (raised by parseListInt)
    """
    listValidated = intInnit(listDimensions, parameterName)
    listNonNegative = []
    for dimension in listValidated:
        if dimension < 0:
            raise ValueError(f"Dimension {dimension} must be non-negative")
        listNonNegative.append(dimension)

    if not listNonNegative:
        raise ValueError("At least one dimension must be non-negative")
    
    return listNonNegative

def validateListDimensions(listDimensions: List[int]) -> List[int]:
    """
    Validates and processes a list of dimensions.

    This function ensures that the input list of dimensions is not None,
    parses it to ensure all dimensions are non-negative, and then filters
    out any dimensions that are not greater than zero. If the resulting
    list has fewer than two dimensions, a NotImplementedError is raised.

    Parameters:
        listDimensions: A list of integer dimensions to be validated.

    Returns:
        listDimensionsPositive: A list of positive dimensions.

    Raises:
        ValueError: If the input listDimensions is None.
        NotImplementedError: If the resulting list of positive dimensions has fewer than two elements.
    """
    if not listDimensions:
        raise ValueError(f"listDimensions is a required parameter.")
    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listDimensionsPositive = [dimension for dimension in listNonNegative if dimension > 0]
    if len(listDimensionsPositive) < 2:
        from typing import get_args

        from mapFolding.oeis import OEISsequenceID
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. Other functions in this package implement the sequences {get_args(OEISsequenceID)}. You may want to look at https://oeis.org/.")
    return listDimensionsPositive

def getLeavesTotal(listDimensions: List[int]) -> int:
    """
    Calculate the product of non-zero, non-negative integers in the given list.

    Parameters:
        listDimensions: A list of integers representing dimensions.

    Returns:
        productDimensions: The product of all positive integer dimensions. Returns 0 if all dimensions are 0.
    """
    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listPositive = [dimension for dimension in listNonNegative if dimension > 0]
        
    if not listPositive:
        return 0
    else:
        productDimensions = 1
        for dimension in listPositive:
            if dimension > sys.maxsize // productDimensions:
                raise OverflowError("Product would exceed maximum integer size")
            productDimensions *= dimension
                
        return productDimensions

def makeConnectionGraph(p: List[int]) -> numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]:
    """
    Constructs a connection graph for a given list of dimensions.
    This function generates a multi-dimensional connection graph based on the provided list of dimensions.
    The graph represents the connections between leaves in a Cartesian product decomposition or dimensional product mapping.
    
    Parameters:
        listDimensions: A _valid_ list of integers representing the dimensions of the map.
    Returns:
        D (connectionGraph): A 3D numpy array representing the connection graph. The shape of the array is (d+1, n+1, n+1),
                                        where d is the number of dimensions and n is the total number of leaves.
    """

    n = getLeavesTotal(p)
    d = len(p)

    """How to build a leaf connection graph, also called a "Cartesian Product Decomposition" 
    or a "Dimensional Product Mapping", with sentinels: 
    Step 1: find the cumulative product of the map's dimensions"""
    P = numpy.ones(d + 1, dtype=numpy.int64) # cumulativeProduct = numpy.ones(dimensionsTotal + 1, dtype=numpy.int64)
    for i in range(1, d + 1): # for dimension1ndex in range(1, dimensionsTotal + 1):
        P[i] = P[i - 1] * p[i - 1] # cumulativeProduct[dimension1ndex] = cumulativeProduct[dimension1ndex - 1] * listDimensions[dimension1ndex - 1]

    """Step 2: for each dimension, create a coordinate system """
    """C[i][m] holds the i-th coordinate of leaf m""" # """coordinateSystem[dimension1ndex][leaf1ndex] holds the dimension1ndex-th coordinate of leaf leaf1ndex"""
    C = numpy.zeros((d + 1, n + 1), dtype=numpy.int64) # coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=numpy.int64)
    for i in range(1, d + 1): # for dimension1ndex in range(1, dimensionsTotal + 1):
        for m in range(1, n + 1): # for leaf1ndex in range(1, leavesTotal + 1):
            C[i][m] = ((m - 1) // P[i - 1]) % p[i - 1] + 1 # coordinateSystem[dimension1ndex][leaf1ndex] = ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) % listDimensions[dimension1ndex - 1] + 1

    """Step 3: create a huge empty connection graph"""
    D = numpy.zeros((d + 1, n + 1, n + 1), dtype=numpy.int64) # connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int64)

    """D[i][l][m] computes the leaf connected to m in dimension i when inserting l""" # """connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndex] computes the leaf1ndex connected to leaf1ndex in dimension1ndex when inserting activeLeaf1ndex"""
    """Step for... for... for...: fill the connection graph"""
    for i in range(1, d + 1): # for dimension1ndex in range(1, dimensionsTotal + 1):
        for l in range(1, n + 1): # for activeLeaf1ndex in range(1, leavesTotal + 1):
            for m in range(1, l + 1): # for leaf1ndexConnectee in range(1, activeLeaf1ndex + 1):
                delta = C[i][l] - C[i][m] # distance = coordinateSystem[dimension1ndex][activeLeaf1ndex] - coordinateSystem[dimension1ndex][leaf1ndexConnectee]
                """If delta is even""" # """If distance is even"""
                if delta % 2 == 0: # if distance % 2 == 0:
                    if C[i][m] == 1: # if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == 1:
                        D[i][l][m] = m # connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee
                    else:
                        D[i][l][m] = m - P[i - 1] # connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee - cumulativeProduct[dimension1ndex - 1]
                else: 
                    """If delta is odd""" # """If distance is odd"""
                    if C[i][m] == p[i - 1] or m + P[i - 1] > l: # if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == listDimensions[dimension1ndex - 1] or leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1] > activeLeaf1ndex:
                        D[i][l][m] = m # connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee
                    else:
                        D[i][l][m] = m + P[i - 1] # connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1]

    return D

def outfitFoldings(listDimensions: List[int]) -> Tuple[List[int], int, numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]]:
    """
    Validates and processes the parameters for the folding computation.

    Parameters:
        listDimensions: A list of dimensions for the folding task.
        computationDivisions: The number of divisions for the computation task.
        computationIndex: The index of the current computation task.

    Returns:
        listDimensions,computationDivisions,computationIndex,leavesTotal,connectionGraph: 
            A tuple containing the validated list of dimensions, the validated number of 
            computation divisions, the validated computation index, and the total number of leaves.
    """
    arrayTrackingHeightHARDCODED = 4
    arrayTrackingHeight = arrayTrackingHeightHARDCODED
    listDimensions = validateListDimensions(listDimensions)
    leavesTotal = getLeavesTotal(listDimensions)
    connectionGraph = makeConnectionGraph(listDimensions)
    arrayTracking = numpy.zeros((arrayTrackingHeight, leavesTotal + 1), dtype=numpy.int64)
    potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=numpy.int64)
    return listDimensions, leavesTotal, connectionGraph, arrayTracking, potentialGaps

def validateTaskDivisions(computationDivisions: int, computationIndex: int, n: int) -> Tuple[int, int]:
    """
    Validates the task divisions for a computation process.

    Parameters:
        computationDivisions: The number of divisions for the computation
        computationIndex: The index of the current computation division
        n: The total number of leaves

    Returns:
        computationDivisions,computationIndex: Tuple containing the validated computationDivisions and computationIndex

    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameters are not integers
    """
    # First validate types
    computationDivisions = intInnit([computationDivisions], 'computationDivisions').pop(0)
    computationIndex = intInnit([computationIndex], 'computationIndex').pop(0)

    # Then validate ranges
    if computationDivisions < 0 or computationIndex < 0:
        raise ValueError(f"computationDivisions, {computationDivisions}, and computationIndex, {computationIndex}, must be non-negative integers.")

    if computationDivisions > n:
        raise ValueError(f"computationDivisions, {computationDivisions}, must be less than or equal to the total number of leaves, {n}.")

    if computationDivisions > 1 and computationIndex >= computationDivisions:
        raise ValueError(f"computationIndex, {computationIndex}, must be less than computationDivisions, {computationDivisions}.")

    return computationDivisions, computationIndex
