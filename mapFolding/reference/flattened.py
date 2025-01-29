"""The algorithm flattened into semantic sections.
This version is not maintained, so you may see differences from the current version."""
from numpy import integer
from numpy.typing import NDArray
from typing import List, Any, Final, Optional, Union, Sequence, Tuple, Type, TypedDict
import enum
import numpy
import sys

def countFolds(listDimensions: Sequence[int], computationDivisions = None, CPUlimit: Optional[Union[int, float, bool]] = None):
    def doWhile():

        while activeLeafGreaterThan0Condition():

            if activeLeafIsTheFirstLeafCondition() or leafBelowSentinelIs1Condition():

                if activeLeafGreaterThanLeavesTotalCondition():
                    foldsSubTotalsIncrement()

                else:

                    findGapsInitializeVariables()
                    while loopingTheDimensions():

                        if dimensionsUnconstrainedCondition():
                            dimensionsUnconstrainedIncrement()

                        else:

                            leafConnecteeInitialization()
                            while loopingLeavesConnectedToActiveLeaf():
                                if thereAreComputationDivisionsYouMightSkip():
                                    countGaps()
                                leafConnecteeUpdate()

                        dimension1ndexIncrement()

                    if allDimensionsAreUnconstrained():
                        insertUnconstrainedLeaf()

                    indexMiniGapInitialization()
                    while loopingToActiveGapCeiling():
                        filterCommonGaps()
                        indexMiniGapIncrement()

            while backtrackCondition():
                backtrack()

            if placeLeafCondition():
                placeLeaf()

    def activeGapIncrement():
        my[indexMy.gap1ndex] += 1

    def activeLeafGreaterThan0Condition():
        return my[indexMy.leaf1ndex] > 0

    def activeLeafGreaterThanLeavesTotalCondition():
        return my[indexMy.leaf1ndex] > the[indexThe.leavesTotal]

    def activeLeafIsTheFirstLeafCondition():
        return my[indexMy.leaf1ndex] <= 1

    def activeLeafNotEqualToTaskDivisionsCondition():
        return my[indexMy.leaf1ndex] != the[indexThe.taskDivisions]

    def allDimensionsAreUnconstrained():
        return my[indexMy.dimensionsUnconstrained] == the[indexThe.dimensionsTotal]

    def backtrack():
        my[indexMy.leaf1ndex] -= 1
        track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]] = track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]
        track[indexTrack.leafAbove, track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]] = track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]

    def backtrackCondition():
        return my[indexMy.leaf1ndex] > 0 and my[indexMy.gap1ndex] == track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex] - 1]

    def computationDivisionsCondition():
        return the[indexThe.taskDivisions] == int(False)

    def countGaps():
        gapsWhere[my[indexMy.gap1ndexCeiling]] = my[indexMy.leafConnectee]
        if track[indexTrack.countDimensionsGapped, my[indexMy.leafConnectee]] == 0:
            gap1ndexCeilingIncrement()
        track[indexTrack.countDimensionsGapped, my[indexMy.leafConnectee]] += 1

    def dimension1ndexIncrement():
        my[indexMy.dimension1ndex] += 1

    def dimensionsUnconstrainedCondition():
        return connectionGraph[my[indexMy.dimension1ndex], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]] == my[indexMy.leaf1ndex]

    def dimensionsUnconstrainedIncrement():
        my[indexMy.dimensionsUnconstrained] += 1

    def filterCommonGaps():
        gapsWhere[my[indexMy.gap1ndex]] = gapsWhere[my[indexMy.indexMiniGap]]
        if track[indexTrack.countDimensionsGapped, gapsWhere[my[indexMy.indexMiniGap]]] == the[indexThe.dimensionsTotal] - my[indexMy.dimensionsUnconstrained]:
            activeGapIncrement()
        track[indexTrack.countDimensionsGapped, gapsWhere[my[indexMy.indexMiniGap]]] = 0

    def findGapsInitializeVariables():
        my[indexMy.dimensionsUnconstrained] = 0
        my[indexMy.gap1ndexCeiling] = track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex] - 1]
        my[indexMy.dimension1ndex] = 1

    def foldsSubTotalsIncrement():
        foldsSubTotals[my[indexMy.taskIndex]] += the[indexThe.leavesTotal]

    def gap1ndexCeilingIncrement():
        my[indexMy.gap1ndexCeiling] += 1

    def indexMiniGapIncrement():
        my[indexMy.indexMiniGap] += 1

    def indexMiniGapInitialization():
        my[indexMy.indexMiniGap] = my[indexMy.gap1ndex]

    def insertUnconstrainedLeaf():
        my[indexMy.indexLeaf] = 0
        while my[indexMy.indexLeaf] < my[indexMy.leaf1ndex]:
            gapsWhere[my[indexMy.gap1ndexCeiling]] = my[indexMy.indexLeaf]
            my[indexMy.gap1ndexCeiling] += 1
            my[indexMy.indexLeaf] += 1

    def leafBelowSentinelIs1Condition():
        return track[indexTrack.leafBelow, 0] == 1

    def leafConnecteeInitialization():
        my[indexMy.leafConnectee] = connectionGraph[my[indexMy.dimension1ndex], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]]

    def leafConnecteeUpdate():
        my[indexMy.leafConnectee] = connectionGraph[my[indexMy.dimension1ndex], my[indexMy.leaf1ndex], track[indexTrack.leafBelow, my[indexMy.leafConnectee]]]

    def loopingLeavesConnectedToActiveLeaf():
        return my[indexMy.leafConnectee] != my[indexMy.leaf1ndex]

    def loopingTheDimensions():
        return my[indexMy.dimension1ndex] <= the[indexThe.dimensionsTotal]

    def loopingToActiveGapCeiling():
        return my[indexMy.indexMiniGap] < my[indexMy.gap1ndexCeiling]

    def placeLeaf():
        my[indexMy.gap1ndex] -= 1
        track[indexTrack.leafAbove, my[indexMy.leaf1ndex]] = gapsWhere[my[indexMy.gap1ndex]]
        track[indexTrack.leafBelow, my[indexMy.leaf1ndex]] = track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]]
        track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]] = my[indexMy.leaf1ndex]
        track[indexTrack.leafAbove, track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]] = my[indexMy.leaf1ndex]
        track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex]] = my[indexMy.gap1ndex]
        my[indexMy.leaf1ndex] += 1

    def placeLeafCondition():
        return my[indexMy.leaf1ndex] > 0

    def taskIndexCondition():
        return my[indexMy.leafConnectee] % the[indexThe.taskDivisions] == my[indexMy.taskIndex]

    def thereAreComputationDivisionsYouMightSkip():
        if computationDivisionsCondition():
            return True
        if activeLeafNotEqualToTaskDivisionsCondition():
            return True
        if taskIndexCondition():
            return True
        return False

    stateUniversal = outfitFoldings(listDimensions, computationDivisions=computationDivisions, CPUlimit=CPUlimit)
    connectionGraph: Final[numpy.ndarray] = stateUniversal['connectionGraph']
    foldsSubTotals = stateUniversal['foldsSubTotals']
    gapsWhere = stateUniversal['gapsWhere']
    my = stateUniversal['my']
    the: Final[numpy.ndarray] = stateUniversal['the']
    track = stateUniversal['track']

    if the[indexThe.taskDivisions] == int(False):
        doWhile()
    else:
        stateUniversal['my'] = my.copy()
        stateUniversal['gapsWhere'] = gapsWhere.copy()
        stateUniversal['track'] = track.copy()
        for indexSherpa in range(the[indexThe.taskDivisions]):
            my = stateUniversal['my'].copy()
            my[indexMy.taskIndex] = indexSherpa
            gapsWhere = stateUniversal['gapsWhere'].copy()
            track = stateUniversal['track'].copy()
            doWhile()

    return numpy.sum(foldsSubTotals).item()

@enum.verify(enum.CONTINUOUS, enum.UNIQUE) if sys.version_info >= (3, 11) else lambda x: x
class EnumIndices(enum.IntEnum):
    """Base class for index enums."""
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """0-indexed."""
        return count

    def __index__(self) -> int:
        """Adapt enum to the ultra-rare event of indexing a NumPy 'ndarray', which is not the
        same as `array.array`. See NumPy.org; I think it will be very popular someday."""
        return self

class indexMy(EnumIndices):
    """Indices for dynamic values."""
    dimension1ndex = enum.auto()
    dimensionsUnconstrained = enum.auto()
    gap1ndex = enum.auto()
    gap1ndexCeiling = enum.auto()
    indexLeaf = enum.auto()
    indexMiniGap = enum.auto()
    leaf1ndex = enum.auto()
    leafConnectee = enum.auto()
    taskIndex = enum.auto()

class indexThe(EnumIndices):
    """Indices for static values."""
    dimensionsTotal = enum.auto()
    leavesTotal = enum.auto()
    taskDivisions = enum.auto()

class indexTrack(EnumIndices):
    """Indices for state tracking array."""
    leafAbove = enum.auto()
    leafBelow = enum.auto()
    countDimensionsGapped = enum.auto()
    gapRangeStart = enum.auto()

class computationState(TypedDict):
    connectionGraph: NDArray[integer[Any]]
    foldsSubTotals: NDArray[integer[Any]]
    mapShape: Tuple[int, ...]
    my: NDArray[integer[Any]]
    gapsWhere: NDArray[integer[Any]]
    the: NDArray[integer[Any]]
    track: NDArray[integer[Any]]

dtypeLarge = numpy.int64
dtypeMedium = dtypeLarge

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
        raise ValueError("Not my problem.")

    if taskDivisions > leavesTotal:
        raise ValueError("What are you doing?")

    return taskDivisions

def makeConnectionGraph(listDimensions: Sequence[int], **keywordArguments: Optional[Type]) -> NDArray[integer[Any]]:
    datatype = keywordArguments.get('datatype', dtypeMedium)
    mapShape = validateListDimensions(listDimensions)
    leavesTotal = getLeavesTotal(mapShape)
    arrayDimensions = numpy.array(mapShape, dtype=datatype)
    dimensionsTotal = len(arrayDimensions)

    cumulativeProduct = numpy.multiply.accumulate([1] + mapShape, dtype=datatype)
    coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=datatype)
    for dimension1ndex in range(1, dimensionsTotal + 1):
        for leaf1ndex in range(1, leavesTotal + 1):
            coordinateSystem[dimension1ndex, leaf1ndex] = ( ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) % arrayDimensions[dimension1ndex - 1] + 1 )

    connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=datatype)
    for dimension1ndex in range(1, dimensionsTotal + 1):
        for activeLeaf1ndex in range(1, leavesTotal + 1):
            for connectee1ndex in range(1, activeLeaf1ndex + 1):
                isFirstCoord = coordinateSystem[dimension1ndex, connectee1ndex] == 1
                isLastCoord = coordinateSystem[dimension1ndex, connectee1ndex] == arrayDimensions[dimension1ndex - 1]
                exceedsActive = connectee1ndex + cumulativeProduct[dimension1ndex - 1] > activeLeaf1ndex
                isEvenParity = (coordinateSystem[dimension1ndex, activeLeaf1ndex] & 1) == (coordinateSystem[dimension1ndex, connectee1ndex] & 1)

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
    if datatype is None:
        datatype = dtypeMedium
    return numpy.zeros(shape, dtype=datatype)

def outfitFoldings(listDimensions: Sequence[int], computationDivisions: Optional[Union[int, str]] = None, CPUlimit: Optional[Union[bool, float, int]] = None, **keywordArguments: Optional[Type]) -> computationState:
    datatypeMedium = keywordArguments.get('datatypeMedium', dtypeMedium)
    datatypeLarge = keywordArguments.get('datatypeLarge', dtypeLarge)

    the = makeDataContainer(len(indexThe), datatypeMedium)

    mapShape = tuple(sorted(validateListDimensions(listDimensions)))
    the[indexThe.leavesTotal] = getLeavesTotal(mapShape)
    the[indexThe.dimensionsTotal] = len(mapShape)
    concurrencyLimit = setCPUlimit(CPUlimit)
    the[indexThe.taskDivisions] = getTaskDivisions(computationDivisions, concurrencyLimit, CPUlimit, listDimensions)

    stateInitialized = computationState(
        connectionGraph = makeConnectionGraph(mapShape, datatype=datatypeMedium),
        foldsSubTotals = makeDataContainer(the[indexThe.leavesTotal], datatypeLarge),
        mapShape = mapShape,
        my = makeDataContainer(len(indexMy), datatypeLarge),
        gapsWhere = makeDataContainer(int(the[indexThe.leavesTotal]) * int(the[indexThe.leavesTotal]) + 1, datatypeMedium),
        the = the,
        track = makeDataContainer((len(indexTrack), the[indexThe.leavesTotal] + 1), datatypeLarge)
        )

    stateInitialized['my'][indexMy.leaf1ndex] = 1
    return stateInitialized

def parseDimensions(dimensions: Sequence[int], parameterName: str = 'unnamed parameter') -> List[int]:
    # listValidated = intInnit(dimensions, parameterName)
    listNOTValidated = dimensions if isinstance(dimensions, (list, tuple)) else list(dimensions)
    listNonNegative = []
    for dimension in listNOTValidated:
        if dimension < 0:
            raise ValueError(f"Dimension {dimension} must be non-negative")
        listNonNegative.append(dimension)
    if not listNonNegative:
        raise ValueError("At least one dimension must be non-negative")
    return listNonNegative

def setCPUlimit(CPUlimit: Union[bool, float, int, None]) -> int:
    # if not (CPUlimit is None or isinstance(CPUlimit, (bool, int, float))):
    #     CPUlimit = oopsieKwargsie(CPUlimit)
    # concurrencyLimit = defineConcurrencyLimit(CPUlimit)
    # numba.set_num_threads(concurrencyLimit)
    concurrencyLimitHARDCODED = 1
    concurrencyLimit = concurrencyLimitHARDCODED
    return concurrencyLimit

def validateListDimensions(listDimensions: Sequence[int]) -> List[int]:
    if not listDimensions:
        raise ValueError(f"listDimensions is a required parameter.")
    listNonNegative = parseDimensions(listDimensions, 'listDimensions')
    dimensionsValid = [dimension for dimension in listNonNegative if dimension > 0]
    if len(dimensionsValid) < 2:
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")
    return sorted(dimensionsValid)
