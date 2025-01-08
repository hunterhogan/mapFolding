from Z0Z_tools import intInnit
from typing import Any, List, Tuple
import numba
import numba.extending
import numpy
import numpy.typing
import sys

@numba.extending.overload(intInnit)
def intInnit_jitInnit(listDimensions, parameterName):
    if isinstance(listDimensions, numba.types.List) and isinstance(parameterName, numba.types.StringLiteral):
        def intInnit_jitInnit_implementInnit(listDimensions, parameterName):
            return listDimensions
        return intInnit_jitInnit_implementInnit
    return None

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

def makeConnectionGraph(listDimensions: List[int], dtype: type = numpy.int64) -> numpy.typing.NDArray[numpy.integer[Any]]:
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
    arrayDimensions = numpy.array(listDimensions, dtype=dtype)
    dimensionsTotal = len(arrayDimensions)

    # Step 1: find the cumulative product of the map's dimensions
    cumulativeProduct = numpy.ones(dimensionsTotal + 1, dtype=dtype)
    for index in range(1, dimensionsTotal + 1):
        cumulativeProduct[index] = cumulativeProduct[index - 1] * arrayDimensions[index - 1]

    # Step 2: create a coordinate system
    coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=dtype)

    for dimension1ndex in range(1, dimensionsTotal + 1):
        for leaf1ndex in range(1, leavesTotal + 1):
            coordinateSystem[dimension1ndex, leaf1ndex] = (
                ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) %
                arrayDimensions[dimension1ndex - 1] + 1
            )

    # Step 3: create and fill the connection graph
    connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=dtype)

    for dimension1ndex in range(1, dimensionsTotal + 1):
        for activeLeaf1ndex in range(1, leavesTotal + 1):
            for connectee1ndex in range(1, activeLeaf1ndex + 1):
                # Base coordinate conditions
                isFirstCoord = coordinateSystem[dimension1ndex, connectee1ndex] == 1
                isLastCoord = coordinateSystem[dimension1ndex, connectee1ndex] == arrayDimensions[dimension1ndex - 1]
                exceedsActive = connectee1ndex + cumulativeProduct[dimension1ndex - 1] > activeLeaf1ndex

                # Parity check
                isEvenParity = (coordinateSystem[dimension1ndex, activeLeaf1ndex] & 1) == \
                                (coordinateSystem[dimension1ndex, connectee1ndex] & 1)

                # Determine connection value
                if (isEvenParity and isFirstCoord) or (not isEvenParity and (isLastCoord or exceedsActive)):
                    connectionGraph[dimension1ndex, activeLeaf1ndex, connectee1ndex] = connectee1ndex
                elif isEvenParity and not isFirstCoord:
                    connectionGraph[dimension1ndex, activeLeaf1ndex, connectee1ndex] = connectee1ndex - cumulativeProduct[dimension1ndex - 1]
                elif not isEvenParity and not (isLastCoord or exceedsActive):
                    connectionGraph[dimension1ndex, activeLeaf1ndex, connectee1ndex] = connectee1ndex + cumulativeProduct[dimension1ndex - 1]
                else:
                    connectionGraph[dimension1ndex, activeLeaf1ndex, connectee1ndex] = connectee1ndex

    return connectionGraph

def outfitFoldings(listDimensions: List[int], dtypeDefault: type = numpy.int64, dtypeMaximum: type = numpy.int64) -> Tuple[List[int], int, numpy.typing.NDArray[numpy.integer[Any]], numpy.typing.NDArray[numpy.integer[Any]], numpy.typing.NDArray[numpy.integer[Any]]]:
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

    connectionGraph = makeConnectionGraph(listDimensions, dtype=dtypeDefault)
    arrayTracking = numpy.zeros((arrayTrackingHeight, leavesTotal + 1), dtype=dtypeDefault)
    potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)

    return listDimensions, leavesTotal, connectionGraph, arrayTracking, potentialGaps

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
        TypeError: If any element cannot be converted to integer (raised by intInnit)
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

numba.jit_module(cache=True, fastmath=True)
