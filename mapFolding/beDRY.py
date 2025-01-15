"""A relatively stable API for oft-needed functionality."""
from mapFolding.importPackages import intInnit, defineConcurrencyLimit, oopsieKwargsie
from mapFolding import indexMy, indexThe, indexTrack, computationState
from typing import Any, List, Optional, Sequence, Type, Union
import numpy
import numba
import numba.extending
import numpy.typing
import sys

def getLeavesTotal(listDimensions: Sequence[int]) -> int:
    """
    Calculate the product of non-zero, non-negative integers in the given list.

    Parameters:
        listDimensions: A list of integers representing dimensions.

    Returns:
        productDimensions: The product of all positive integer dimensions. Returns 0 if all dimensions are 0.
    """
    listNonNegative = parseDimensions(listDimensions, 'listDimensions')
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

def getTaskDivisions(CPUlimit, computationDivisions: Optional[Union[int, str]], concurrencyLimit: int, listDimensions, the: numpy.typing.NDArray[numpy.integer[Any]], ):
    # TODO remove after restructuring the tests
    if isinstance(computationDivisions, bool) and computationDivisions:
        computationDivisions = "maximum"

    if not computationDivisions:
        # Coding it this way should cover `None`, `False`, and `0`.
        the[indexThe.taskDivisions] = 0
    elif isinstance(computationDivisions, int):
        the[indexThe.taskDivisions] = computationDivisions
    elif isinstance(computationDivisions, str):
        computationDivisions = computationDivisions.lower()
        if computationDivisions == "maximum":
            the[indexThe.taskDivisions] = the[indexThe.leavesTotal]
        elif computationDivisions == "cpu":
            the[indexThe.taskDivisions] = min(concurrencyLimit, the[indexThe.leavesTotal])
    else:
        raise ValueError(f"I received {computationDivisions} for the parameter, `computationDivisions`, but the so-called programmer didn't implement code for that.")

    if the[indexThe.taskDivisions] > the[indexThe.leavesTotal]:
        raise ValueError(f"Problem: `taskDivisions`, ({the[indexThe.taskDivisions]}), is greater than `leavesTotal`, ({the[indexThe.leavesTotal]}), which will cause duplicate counting of the folds.\n\nChallenge: you cannot directly set `taskDivisions` or `leavesTotal`. They are derived from parameters that may or may not still be named `computationDivisions`, `CPUlimit` , and `listDimensions` and from dubious-quality Python code.\n\nFor those parameters, I received {computationDivisions=}, {CPUlimit=}, and {listDimensions=}.\n\nPotential solutions: get a different hobby or set `computationDivisions` to a different value.")

    return the

def makeConnectionGraph(listDimensions: Sequence[int], dtype: Optional[Type] = numpy.int64) -> numpy.typing.NDArray[numpy.integer[Any]]:
    """
    Constructs a connection graph for a given list of dimensions.
    This function generates a multi-dimensional connection graph based on the provided list of dimensions.
    The graph represents the connections between leaves in a Cartesian product decomposition or dimensional product mapping.

    Parameters:
        listDimensions: A validated sequence of integers representing the dimensions of the map.
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

def outfitFoldings(
    listDimensions: Sequence[int],
    computationDivisions: Optional[Union[int, str]] = None,
    CPUlimit: Optional[Union[int, float, bool]] = None,
    dtypeDefault: Optional[Type] = numpy.int64, # TODO consider allowing a type or a "signal", such as "minimum", "safe", "maximum"
    dtypeLarge: Optional[Type] = numpy.int64, # Can/should I use numba types?
    ) -> computationState:
    the = numpy.zeros(len(indexThe), dtype=dtypeDefault)

    mapShape = tuple(sorted(validateListDimensions(listDimensions)))
    the[indexThe.leavesTotal] = getLeavesTotal(mapShape)
    the[indexThe.dimensionsTotal] = len(mapShape)
    concurrencyLimit = setCPUlimit(CPUlimit)

    the = getTaskDivisions(CPUlimit, computationDivisions, concurrencyLimit, listDimensions, the)

    stateInitialized = computationState(
        connectionGraph = makeConnectionGraph(mapShape, dtype=dtypeDefault),
        foldsTotal = numpy.zeros(the[indexThe.leavesTotal], dtype=numpy.int64),
        mapShape = mapShape,
        my = numpy.zeros(len(indexMy), dtype=dtypeLarge),
        gapsWhere = numpy.zeros(int(the[indexThe.leavesTotal]) * int(the[indexThe.leavesTotal]) + 1, dtype=dtypeDefault),
        the = the,
        track = numpy.zeros((len(indexTrack), the[indexThe.leavesTotal] + 1), dtype=dtypeLarge)
        )

    stateInitialized['my'][indexMy.leaf1ndex.value] = 1

    return stateInitialized

def parseDimensions(dimensions: Sequence[int], parameterName: str = 'unnamed parameter') -> List[int]:
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
    listValidated = intInnit(dimensions, parameterName)
    listNonNegative = []
    for dimension in listValidated:
        if dimension < 0:
            raise ValueError(f"Dimension {dimension} must be non-negative")
        listNonNegative.append(dimension)

    if not listNonNegative:
        raise ValueError("At least one dimension must be non-negative")

    return listNonNegative

def setCPUlimit(CPUlimit: Union[int, float, bool, None]):
    """Sets CPU limit for concurrent operations using Numba.
    This function configures the number of CPU threads that Numba can use for parallel execution.
    Note that this setting only affects Numba-jitted functions that have not yet been imported.
    Parameters:
        CPUlimit (Union[int, float, bool, None]): The CPU limit to set.
            - If int/float: Specifies number of CPU threads to use
            - If bool: True uses all available CPUs, False uses 1 CPU
            - If None: Uses system default
    Returns:
        concurrencyLimit: The actual concurrency limit that was set
    Raises:
        TypeError: If CPUlimit is not of the expected types
    """
    if not (CPUlimit is None or isinstance(CPUlimit, (bool, int, float))):
        CPUlimit = oopsieKwargsie(CPUlimit)

    concurrencyLimit = defineConcurrencyLimit(CPUlimit)
    # NOTE `set_num_threads` only affects "jitted" functions that have _not_ yet been "imported"
    numba.set_num_threads(concurrencyLimit)

    return concurrencyLimit

def validateListDimensions(listDimensions: Sequence[int]) -> List[int]:
    """
    Validates and processes a list of dimensions.

    This function ensures that the input list of dimensions is not None,
    parses it to ensure all dimensions are non-negative, and then filters
    out any dimensions that are not greater than zero. If the resulting
    list has fewer than two dimensions, a NotImplementedError is raised.

    Parameters:
        listDimensions: A list of integer dimensions to be validated.

    Returns:
        validDimensions: A list, with at least two elements, of only positive integers.

    Raises:
        ValueError: If the input listDimensions is None.
        NotImplementedError: If the resulting list of positive dimensions has fewer than two elements.
    """
    if not listDimensions:
        raise ValueError(f"listDimensions is a required parameter.")
    listNonNegative = parseDimensions(listDimensions, 'listDimensions')
    validDimensions = [dimension for dimension in listNonNegative if dimension > 0]
    if len(validDimensions) < 2:
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")
    return validDimensions
