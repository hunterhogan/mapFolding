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
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")
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

def makeConnectionGraph(listDimensions: List[int], dtype: type = numpy.int64):
    """
    Constructs a connection graph for a given list of dimensions.
    This function generates a multi-dimensional connection graph based on the provided list of dimensions.
    The graph represents the connections between leaves in a Cartesian product decomposition or dimensional product mapping.
    
    Parameters:
        listDimensions: A validated list of integers representing the dimensions of the map.
    Returns:
        connectionGraph: A 3D numpy array with shape of (dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1).
    """

    leavesTotal = getLeavesTotal(listDimensions)
    dimensionsTotal = len(listDimensions)

    """How to build a leaf connection graph, also called a "Cartesian Product Decomposition" 
    or a "Dimensional Product Mapping", with sentinels: 
    Step 1: find the cumulative product of the map's dimensions"""
    cumulativeProduct = numpy.ones(dimensionsTotal + 1, dtype=dtype)
    for dimension1ndex in range(1, dimensionsTotal + 1):
        cumulativeProduct[dimension1ndex] = cumulativeProduct[dimension1ndex - 1] * listDimensions[dimension1ndex - 1]

    """Step 2: for each dimension, create a coordinate system """
    """coordinateSystem[dimension1ndex][leaf1ndex] holds the dimension1ndex-th coordinate of leaf leaf1ndex"""
    coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=dtype)
    for dimension1ndex in range(1, dimensionsTotal + 1):
        for leaf1ndex in range(1, leavesTotal + 1):
            coordinateSystem[dimension1ndex][leaf1ndex] = ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) % listDimensions[dimension1ndex - 1] + 1

    """Step 3: create a huge empty connection graph"""
    connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=dtype)

    """connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndex] computes the leaf1ndex connected to leaf1ndex in dimension1ndex when inserting activeLeaf1ndex"""
    """Step for... for... for...: fill the connection graph"""
    for dimension1ndex in range(1, dimensionsTotal + 1):
        for activeLeaf1ndex in range(1, leavesTotal + 1):
            for leaf1ndexConnectee in range(1, activeLeaf1ndex + 1):
                """If distance is even"""
                if (coordinateSystem[dimension1ndex][activeLeaf1ndex] & 1) == (coordinateSystem[dimension1ndex][leaf1ndexConnectee] & 1):
                    if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == 1:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee
                    else:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee - cumulativeProduct[dimension1ndex - 1]
                else: 
                    """If distance is odd"""
                    if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == listDimensions[dimension1ndex - 1] or leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1] > activeLeaf1ndex:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee
                    else:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1]

    return connectionGraph

def outfitFoldings(listDimensions: List[int], dtypeDefault: type = numpy.int64, dtypeMaximum: type = numpy.int64):
    """
    Outfits the folding process with the necessary data structures.

    Parameters:
        listDimensions: A list of integers representing the dimensions of the map.

    Returns:
        listDimensions, leavesTotal, connectionGraph, arrayTracking, potentialGaps: Tuple containing the validated list of dimensions, the total number of leaves, the connection graph, an array for tracking, and an array for potential gaps.
    """
    arrayTrackingHeightHARDCODED = 4
    arrayTrackingHeight = arrayTrackingHeightHARDCODED

    listDimensions = validateListDimensions(listDimensions)
    leavesTotal = getLeavesTotal(listDimensions)

    # connectionGraph = makeConnectionGraph(listDimensions, dtype=int)
    connectionGraph = makeConnectionGraph(listDimensions, dtype=dtypeDefault)
    # connectionGraph = makeConnectionGraph(listDimensions)
    arrayTracking = numpy.zeros((arrayTrackingHeight, leavesTotal + 1), dtype=dtypeDefault)
    potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)

    return listDimensions, leavesTotal, connectionGraph, arrayTracking, potentialGaps
