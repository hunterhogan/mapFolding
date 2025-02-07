"""A relatively stable API for oft-needed functionality."""
from operator import ge
from mapFolding import (
    computationState,
    getPathJobDEFAULT,
    hackSSOTdtype,
    indexMy,
    indexTrack,
    setDatatypeElephino,
    setDatatypeFoldsTotal,
    setDatatypeLeavesTotal,
)
from numpy import integer
from numpy.typing import NDArray, DTypeLike
from typing import Any, List, Optional, Sequence, Tuple, Type, Union
from Z0Z_tools import intInnit, defineConcurrencyLimit, oopsieKwargsie
import numba
import numpy
import os
import pathlib
import sys

def getFilenameFoldsTotal(mapShape: Union[Sequence[int], numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]]) -> str:
    """Generate a standardized filename string for storing map folding totals.

    The function takes a map shape sequence and converts it into a filename string
    representing the dimensions of the map, followed by '.foldsTotal' extension.
    For example, [3, 2] becomes 'p2x3.foldsTotal'.

    Parameters:
        mapShape: A sequence of integers representing the dimensions
            of the map (e.g., [3, 2] for a 3x2 map)

    Returns:
        A filename string in format 'pNxM.foldsTotal' where N,M are sorted dimensions
    """
    return 'p' + 'x'.join(str(dim) for dim in sorted(mapShape)) + '.foldsTotal'

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

def getPathFilenameFoldsTotal(mapShape: Union[Sequence[int], numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]], pathLikeWriteFoldsTotal: Optional[Union[str, os.PathLike[str]]] = None) -> pathlib.Path:
    """Get path for folds total file.

    This function determines the file path for storing fold totals. If a path is provided,
    it will use that path. If the path is a directory, it will append a default filename.
    The function ensures the parent directory exists by creating it if necessary.

    Parameters:
        mapShape (Sequence[int]): List of dimensions for the map folding problem.
        pathLikeWriteFoldsTotal (Union[str, os.PathLike[str]], optional): Path where to save
            the folds total. Can be a file path or directory path. If None, uses default path.
            Defaults to None.

    Returns:
        pathlib.Path: Complete path to the folds total file.
    """
    pathFilenameFoldsTotal = pathlib.Path(pathLikeWriteFoldsTotal) if pathLikeWriteFoldsTotal is not None else getPathJobDEFAULT()
    if pathFilenameFoldsTotal.is_dir():
        filenameFoldsTotalDEFAULT = getFilenameFoldsTotal(mapShape)
        pathFilenameFoldsTotal = pathFilenameFoldsTotal / filenameFoldsTotalDEFAULT
    pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
    return pathFilenameFoldsTotal

def getTaskDivisions(computationDivisions: Optional[Union[int, str]], concurrencyLimit: int, CPUlimit: Optional[Union[bool, float, int]], listDimensions: Sequence[int]) -> int:
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
    CPUlimit: for error reporting
    listDimensions: for error reporting

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
    taskDivisions = 0
    leavesTotal = getLeavesTotal(listDimensions)
    if not computationDivisions:
        pass
    elif isinstance(computationDivisions, int):
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

def makeConnectionGraph(listDimensions: Sequence[int], **keywordArguments: Optional[Type]) -> numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]:
    """
    Constructs a multi-dimensional connection graph representing the connections between the leaves of a map with the given dimensions.
    Also called a Cartesian product decomposition or dimensional product mapping.

    Parameters
        listDimensions: A sequence of integers representing the dimensions of the map.
        **keywordArguments: Datatype management.

    Returns
        connectionGraph: A 3D numpy array with shape of (dimensionsTotal, leavesTotal + 1, leavesTotal + 1).
    """
    if keywordArguments.get('datatype', None):
        setDatatypeLeavesTotal(keywordArguments['datatype']) # type: ignore
    datatype = hackSSOTdtype('connectionGraph')
    mapShape = validateListDimensions(listDimensions)
    leavesTotal = getLeavesTotal(mapShape)
    arrayDimensions = numpy.array(mapShape, dtype=datatype)
    dimensionsTotal = len(arrayDimensions)

    cumulativeProduct = numpy.multiply.accumulate([1] + mapShape, dtype=datatype)
    coordinateSystem = numpy.zeros((dimensionsTotal, leavesTotal + 1), dtype=datatype)
    for indexDimension in range(dimensionsTotal):
        for leaf1ndex in range(1, leavesTotal + 1):
            coordinateSystem[indexDimension, leaf1ndex] = ( ((leaf1ndex - 1) // cumulativeProduct[indexDimension]) % arrayDimensions[indexDimension] + 1 )

    connectionGraph = numpy.zeros((dimensionsTotal, leavesTotal + 1, leavesTotal + 1), dtype=datatype)
    for indexDimension in range(dimensionsTotal):
        for activeLeaf1ndex in range(1, leavesTotal + 1):
            for connectee1ndex in range(1, activeLeaf1ndex + 1):
                isFirstCoord = coordinateSystem[indexDimension, connectee1ndex] == 1
                isLastCoord = coordinateSystem[indexDimension, connectee1ndex] == arrayDimensions[indexDimension]
                exceedsActive = connectee1ndex + cumulativeProduct[indexDimension] > activeLeaf1ndex
                isEvenParity = (coordinateSystem[indexDimension, activeLeaf1ndex] & 1) == (coordinateSystem[indexDimension, connectee1ndex] & 1)

                if (isEvenParity and isFirstCoord) or (not isEvenParity and (isLastCoord or exceedsActive)):
                    connectionGraph[indexDimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex
                elif isEvenParity and not isFirstCoord:
                    connectionGraph[indexDimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex - cumulativeProduct[indexDimension]
                elif not isEvenParity and not (isLastCoord or exceedsActive):
                    connectionGraph[indexDimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex + cumulativeProduct[indexDimension]

    return connectionGraph

def makeDataContainer(shape: Union[int, Tuple[int, ...]], datatype: Optional[DTypeLike] = None) -> NDArray[integer[Any]]:
    """Create a zeroed-out `numpy.ndarray` with the given shape and datatype.

    Parameters:
        shape (Union[int, Tuple[int, ...]]): The shape of the array. Can be an integer for 1D arrays
            or a tuple of integers for multi-dimensional arrays.
        datatype (Optional[DTypeLike], optional): The desired data type for the array.
            If None, defaults to dtypeLargeDEFAULT. Defaults to None.

    Returns:
        numpy.ndarray: A new array of given shape and type, filled with zeros.
    """
    if datatype is None:
        datatype = hackSSOTdtype('dtypeFoldsTotal')
    return numpy.zeros(shape, dtype=datatype)

def outfitCountFolds(listDimensions: Sequence[int]
                        , computationDivisions: Optional[Union[int, str]] = None
                        , CPUlimit: Optional[Union[bool, float, int]] = None
                        , **keywordArguments: Optional[Union[str, bool]]) -> computationState:
    """
    Initializes and configures the computation state for map folding computations.

    Parameters
    ----------
    listDimensions:
        The dimensions of the map to be folded
    computationDivisions (None):
        Specifies how to divide computations:
        - None: no division of the computation into tasks; sets task divisions to 0
        - int: direct set the number of task divisions; cannot exceed the map's total leaves
        - "maximum": divides into `leavesTotal`-many `taskDivisions`
        - "cpu": limits the divisions to the number of available CPUs, i.e. `concurrencyLimit`
    CPUlimit (None):
        Whether and how to limit the CPU usage. See notes for details.
    **keywordArguments:
        Datatype management.

    Returns
    -------
    computationState
        An initialized computation state containing:
        - connectionGraph: Graph representing connections in the map
        - foldsSubTotals: Array tracking total folds
        - mapShape: Validated and sorted dimensions of the map
        - my: Array for internal state tracking
        - gapsWhere: Array tracking gap positions
        - the: Static settings and metadata
        - track: Array for tracking computation progress

    Limits on CPU usage `CPUlimit`:
        - `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
        - `True`: Yes, limit the CPU usage; limits to 1 CPU.
        - Integer `>= 1`: Limits usage to the specified number of CPUs.
        - Decimal value (`float`) between 0 and 1: Fraction of total CPUs to use.
        - Decimal value (`float`) between -1 and 0: Fraction of CPUs to *not* use.
        - Integer `<= -1`: Subtract the absolute value from total CPUs.
    """
    kwourGrapes = keywordArguments.get('sourGrapes', False)
    kwatatype = keywordArguments.get('datatypeElephino', None)
    if kwatatype: setDatatypeElephino(kwatatype, sourGrapes=kwourGrapes) # type: ignore
    kwatatype = keywordArguments.get('datatypeFoldsTotal', None)
    if kwatatype: setDatatypeFoldsTotal(kwatatype, sourGrapes=kwourGrapes) # type: ignore
    kwatatype = keywordArguments.get('datatypeLeavesTotal', None)
    if kwatatype: setDatatypeLeavesTotal(kwatatype, sourGrapes=kwourGrapes) # type: ignore

    my = makeDataContainer(len(indexMy), hackSSOTdtype('my'))

    mapShape = tuple(sorted(validateListDimensions(listDimensions)))
    concurrencyLimit = setCPUlimit(CPUlimit)
    my[indexMy.taskDivisions] = getTaskDivisions(computationDivisions, concurrencyLimit, CPUlimit, mapShape)

    foldGroups = makeDataContainer(max(my[indexMy.taskDivisions] + 1, 2), hackSSOTdtype('foldGroups'))
    leavesTotal = getLeavesTotal(mapShape)
    foldGroups[-1] = leavesTotal

    my[indexMy.dimensionsTotal] = len(mapShape)
    my[indexMy.leaf1ndex] = 1
    stateInitialized = computationState(
        connectionGraph = makeConnectionGraph(mapShape, datatype=hackSSOTdtype('connectionGraph')),
        foldGroups = foldGroups,
        mapShape = numpy.array(mapShape, dtype=hackSSOTdtype('mapShape')),
        my = my,
        gapsWhere = makeDataContainer(int(leavesTotal) * int(leavesTotal) + 1, hackSSOTdtype('gapsWhere')),
        track = makeDataContainer((len(indexTrack), leavesTotal + 1), hackSSOTdtype('track')),
        )

    return stateInitialized

def parseDimensions(dimensions: Sequence[int], parameterName: str = 'listDimensions') -> List[int]:
    """
    Parse and validate dimensions are non-negative integers.

    Parameters:
        dimensions: Sequence of integers representing dimensions
        parameterName ('listDimensions'): Name of the parameter for error messages. Defaults to 'listDimensions'
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

    return listNonNegative

def saveFoldsTotal(pathFilename: Union[str, os.PathLike[str]], foldsTotal: int) -> None:
    """
    Save foldsTotal with multiple fallback mechanisms.

    Parameters:
        pathFilename: Target save location
        foldsTotal: Critical computed value to save
    """
    try:
        pathFilenameFoldsTotal = pathlib.Path(pathFilename)
        pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
        pathFilenameFoldsTotal.write_text(str(foldsTotal))
    except Exception as ERRORmessage:
        try:
            print(f"\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n\n{foldsTotal=}\n\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n")
            print(ERRORmessage)
            print(f"\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n\n{foldsTotal=}\n\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n")
            randomnessPlanB = (int(str(foldsTotal).strip()[-1]) + 1) * ['YO_']
            filenameInfixUnique = ''.join(randomnessPlanB)
            pathFilenamePlanB = os.path.join(os.getcwd(), 'foldsTotal' + filenameInfixUnique + '.txt')
            writeStreamFallback = open(pathFilenamePlanB, 'w')
            writeStreamFallback.write(str(foldsTotal))
            writeStreamFallback.close()
            print(str(pathFilenamePlanB))
        except Exception:
            print(foldsTotal)

def setCPUlimit(CPUlimit: Optional[Any]) -> int:
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

    concurrencyLimit = int(defineConcurrencyLimit(CPUlimit))
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
        raise ValueError("listDimensions is a required parameter.")
    listNonNegative = parseDimensions(listDimensions, 'listDimensions')
    dimensionsValid = [dimension for dimension in listNonNegative if dimension > 0]
    if len(dimensionsValid) < 2:
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")
    return sorted(dimensionsValid)
