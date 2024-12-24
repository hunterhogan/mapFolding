import sys
from typing import List, Tuple

import numpy
import numpy.typing
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

def makeConnectionGraph(listDimensions: List[int]) -> numpy.typing.NDArray[numpy.int64]:
    """
    Constructs a connection graph for a given list of dimensions.
    This function generates a multi-dimensional connection graph based on the provided list of dimensions.
    The graph represents the connections between leaves in a Cartesian product decomposition or dimensional product mapping.
    
    Parameters:
        listDimensions: A list of integers representing the dimensions of the map.
    Returns:
        D (connectionGraph): A 3D numpy array representing the connection graph. The shape of the array is (d+1, n+1, n+1),
                                        where d is the number of dimensions and n is the total number of leaves.
    """

    listDimensions = validateListDimensions(listDimensions)
    n = getLeavesTotal(listDimensions)
    d = len(listDimensions)

    # How to build a numpy.ndarray connectionGraph with sentinel values: 
    # ("Cartesian Product Decomposition" or "Dimensional Product Mapping")
    # Step 1: find the cumulative product of the map dimensions
    P = numpy.ones(d + 1, dtype=numpy.int64)
    for i in range(1, d + 1):
        P[i] = P[i - 1] * listDimensions[i - 1]

    # Step 2: for each dimension, create a coordinate system
    # C[i][m] holds the i-th coordinate of leaf m
    C = numpy.zeros((d + 1, n + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for m in range(1, n + 1):
            C[i][m] = ((m - 1) // P[i - 1]) % listDimensions[i - 1] + 1

    # Step 3: create a huge empty leafConnectionGraph
    D = numpy.zeros((d + 1, n + 1, n + 1), dtype=numpy.int64)

    # Step for... for... for...: fill the leafConnectionGraph
    for i in range(1, d + 1):
    # D[i][l][m] computes the leaf connected to m in dimension i when inserting l
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = C[i][l] - C[i][m]
                if delta % 2 == 0: # If delta is even
                    if C[i][m] == 1:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m - P[i - 1]
                else: # If delta is odd
                    if C[i][m] == listDimensions[i - 1] or m + P[i - 1] > l:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m + P[i - 1]
    return D

def validateParametersFoldings(listDimensions: List[int]):
    """
    Validates and processes the parameters for the folding computation.

    Parameters:
        listDimensions: A list of dimensions for the folding task.

    Returns:
        listDimensions,leavesTotal,connectionGraph: 
            A tuple containing the validated list of dimensions, the validated number of 
            computation divisions, the validated computation index, and the total number of leaves.
    """
    # I don't know if I should put all of these steps in series or if each function should validate its own parameters.
    # In the future, I might not call the entire series. Also, it feels weird to return listDimensions from makeConnectionGraph.
    listDimensions = validateListDimensions(listDimensions)
    leavesTotal = getLeavesTotal(listDimensions)
    connectionGraph = makeConnectionGraph(listDimensions)
    return listDimensions, leavesTotal, connectionGraph
