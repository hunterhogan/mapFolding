"""A relatively stable API for oft-needed functionality."""
from mapFolding import intInnit, defineConcurrencyLimit, oopsieKwargsie
from mapFolding import indexMy, indexThe, indexTrack, computationState
from mapFolding import dtypeDefault, dtypeLarge, dtypeSmall
from typing import Any, List, Optional, Sequence, Type, Union
import numpy
import numba
from numpy.typing import NDArray
from numpy import integer
import sys
import operator

def getLeavesTotal(listDimensions: Sequence[int]) -> int:
    """
    How many leaves are in the map.

    Parameters:
        listDimensions: A list of integers representing dimensions.

    Returns:
        productDimensions: The product of all positive integer dimensions.
    """
    listNonNegative = parseDimensions(listDimensions, 'listDimensions')
    listPositive = [dimension for dimension in listNonNegative if dimension > 0]

    if not listPositive:
        return 0
    else:
        productDimensions = 1
        for dimension in listPositive:
            if dimension > sys.maxsize // productDimensions:
                raise OverflowError(f"I received {dimension=} in {listDimensions=}, but the product of the dimensions exceeds the maximum size of an integer on this system.")
            productDimensions *= dimension

        return productDimensions

def getTaskDivisions(computationDivisions: Optional[Union[int, str]], concurrencyLimit: int, CPUlimit: Optional[Union[bool, float, int]], listDimensions: Sequence[int]):
    """
    Determines whether or how to divide the computation into tasks.

    Parameters
    ----------
    computationDivisions (None):
        Specifies how to divide computations:
        - None: no division of the computation into tasks; sets task divisions to 0
        - int: direct set the number of task divisions; cannot exceed the map's total leaves
        - "maximum": divides into `leavesTotal`-many `taskDivisions`
        - "cpu": limits the divisions to the number of available CPUs, i.e. `concurrencyLimit`
    concurrencyLimit:
        Maximum number of concurrent tasks allowed
    listDimensions: for error reporting
    CPUlimit: for error reporting

    Returns
    -------
    taskDivisions:

    Raises
    ------
    ValueError
        If computationDivisions is an unsupported type or if resulting task divisions exceed total leaves

    Notes
    -----
    Task divisions cannot exceed total leaves to prevent duplicate counting of folds.
    """
    if not computationDivisions:
        return 0
    else:
        leavesTotal = getLeavesTotal(listDimensions)
    if isinstance(computationDivisions, int):
        taskDivisions = computationDivisions
    elif isinstance(computationDivisions, str):
        computationDivisions = computationDivisions.lower()
        if computationDivisions == "maximum":
            taskDivisions = leavesTotal
        elif computationDivisions == "cpu":
            taskDivisions = min(concurrencyLimit, leavesTotal)
    else:
        raise ValueError(f"I received {computationDivisions} for the parameter, `computationDivisions`, but the so-called programmer didn't implement code for that.")

    if taskDivisions > leavesTotal:
        raise ValueError(f"Problem: `taskDivisions`, ({taskDivisions}), is greater than `leavesTotal`, ({leavesTotal}), which will cause duplicate counting of the folds.\n\nChallenge: you cannot directly set `taskDivisions` or `leavesTotal`. They are derived from parameters that may or may not still be named `computationDivisions`, `CPUlimit` , and `listDimensions` and from dubious-quality Python code.\n\nFor those parameters, I received {computationDivisions=}, {CPUlimit=}, and {listDimensions=}.\n\nPotential solutions: get a different hobby or set `computationDivisions` to a different value.")

    return taskDivisions

def makeConnectionGraph(listDimensions: Sequence[int], **keywordArguments: Optional[Type]) -> NDArray[integer[Any]]:
    """
    Constructs a multi-dimensional connection graph representing the connections between the leaves of a map with the given dimensions.
    Also called a Cartesian product decomposition or dimensional product mapping.

    Parameters:
        listDimensions: A sequence of integers representing the dimensions of the map.
    Returns:
        connectionGraph: A 3D numpy array with shape of (dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1).
    """
    datatype = keywordArguments.get('datatype', dtypeDefault)
    mapShape = validateListDimensions(listDimensions)
    leavesTotal = getLeavesTotal(mapShape)
    arrayDimensions = numpy.array(mapShape, dtype=datatype)
    dimensionsTotal = len(arrayDimensions)

    # Step 1: find the cumulative product of the map's dimensions
    cumulativeProduct = numpy.multiply.accumulate([1] + mapShape, dtype=datatype)

    # Step 2: create a coordinate system
    coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=datatype)

    for dimension1ndex in range(1, dimensionsTotal + 1):
        for leaf1ndex in range(1, leavesTotal + 1):
            coordinateSystem[dimension1ndex, leaf1ndex] = (
                ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) %
                arrayDimensions[dimension1ndex - 1] + 1
            )

    # Step 3: create and fill the connection graph
    connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=datatype)

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

def makeDataContainer(shape, datatype: Optional[Type] = None):
    """Create a container, probably numpy.ndarray, with the given shape and datatype."""
    if datatype is None:
        datatype = dtypeDefault
    return numpy.zeros(shape, dtype=datatype)

def outfitFoldings(listDimensions: Sequence[int], computationDivisions: Optional[Union[int, str]] = None, CPUlimit: Optional[Union[bool, float, int]] = None, **keywordArguments: Optional[Type]) -> computationState:
    """
    Initializes and configures the computation state for map folding computations.

    Parameters
    ----------
    listDimensions:
        The dimensions of the map to be folded
    computationDivisions (None):
        Specifies how to divide the computation tasks
    CPUlimit (None):
        Limits the CPU usage for computations

    Returns
    -------
    computationState
        An initialized computation state containing:
        - connectionGraph: Graph representing connections in the map
        - foldsTotal: Array tracking total folds
        - mapShape: Validated and sorted dimensions of the map
        - my: Array for internal state tracking
        - gapsWhere: Array tracking gap positions
        - the: Static settings and metadata
        - track: Array for tracking computation progress
    """
    datatypeDefault = keywordArguments.get('datatypeDefault', dtypeDefault)
    datatypeLarge = keywordArguments.get('datatypeLarge', dtypeLarge)

    the = makeDataContainer(len(indexThe), datatypeDefault)

    mapShape = tuple(sorted(validateListDimensions(listDimensions)))
    the[indexThe.leavesTotal] = getLeavesTotal(mapShape)
    the[indexThe.dimensionsTotal] = len(mapShape)
    concurrencyLimit = setCPUlimit(CPUlimit)
    the[indexThe.taskDivisions] = getTaskDivisions(computationDivisions, concurrencyLimit, CPUlimit, listDimensions)
    
    stateInitialized = computationState(
        connectionGraph = makeConnectionGraph(mapShape, datatype=datatypeDefault),
        foldsTotal = makeDataContainer(the[indexThe.leavesTotal], datatypeLarge),
        mapShape = mapShape,
        my = makeDataContainer(len(indexMy), datatypeLarge),
        gapsWhere = makeDataContainer(int(the[indexThe.leavesTotal]) * int(the[indexThe.leavesTotal]) + 1, datatypeDefault),
        the = the,
        track = makeDataContainer((len(indexTrack), the[indexThe.leavesTotal] + 1), datatypeLarge)
        )

    stateInitialized['my'][indexMy.leaf1ndex.value] = 1

    return stateInitialized

def parseDimensions(dimensions: Sequence[int], parameterName: str = 'unnamed parameter') -> List[int]:
    """
    Parse and validate dimensions are non-negative integers.

    Parameters:
        dimensions: Sequence of integers representing dimensions
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

def setCPUlimit(CPUlimit: Union[bool, float, int, None]) -> int:
    """Sets CPU limit for Numba concurrent operations. Note that it can only affect Numba-jitted functions that have not yet been imported.

    Parameters:
        CPUlimit: whether and how to limit the CPU usage. See notes for details.
    Returns:
        concurrencyLimit: The actual concurrency limit that was set
    Raises:
        TypeError: If CPUlimit is not of the expected types

    Limits on CPU usage `CPUlimit`:
        - `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
        - `True`: Yes, limit the CPU usage; limits to 1 CPU.
        - Integer `>= 1`: Limits usage to the specified number of CPUs.
        - Decimal value (`float`) between 0 and 1: Fraction of total CPUs to use.
        - Decimal value (`float`) between -1 and 0: Fraction of CPUs to *not* use.
        - Integer `<= -1`: Subtract the absolute value from total CPUs.
    """
    if not (CPUlimit is None or isinstance(CPUlimit, (bool, int, float))):
        CPUlimit = oopsieKwargsie(CPUlimit)

    concurrencyLimit = defineConcurrencyLimit(CPUlimit)
    numba.set_num_threads(concurrencyLimit)

    return concurrencyLimit

def validateListDimensions(listDimensions: Sequence[int]) -> List[int]:
    """
    Validates and sorts a sequence of at least two positive dimensions.

    Parameters:
        listDimensions: A sequence of integer dimensions to be validated.

    Returns:
        dimensionsValidSorted: A list, with at least two elements, of only positive integers.

    Raises:
        ValueError: If the input listDimensions is None.
        NotImplementedError: If the resulting list of positive dimensions has fewer than two elements.
    """
    if not listDimensions:
        raise ValueError(f"listDimensions is a required parameter.")
    listNonNegative = parseDimensions(listDimensions, 'listDimensions')
    dimensionsValid = [dimension for dimension in listNonNegative if dimension > 0]
    if len(dimensionsValid) < 2:
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")
    return sorted(dimensionsValid)
