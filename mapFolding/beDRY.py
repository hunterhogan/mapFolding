"""
Core utility functions implementing DRY (Don't Repeat Yourself) principles for the mapFolding package.

This module serves as the foundation for consistent data management and parameter validation across the entire
mapFolding computation assembly-line. It provides critical utility functions that:

1. Calculate and validate fundamental computational parameters such as leaves total and task divisions.
2. Generate specialized connection graphs that define the folding algorithm's constraints.
3. Provide centralized resource allocation and system limits management.
4. Construct and manage uniform data structures for the computation state.
5. Ensure parameter validation and safe type conversion.

The functions in this module maintain a clear separation between data initialization and algorithm implementation,
enabling the package to support multiple computational strategies (sequential, parallel, and JIT-compiled) while
ensuring consistent input handling and state management.

These utilities form a stable internal API that other modules depend on, particularly theSSOT (Single Source of Truth),
theDao (core algorithm), and the synthetic module generators that produce optimized implementations.
"""
from collections.abc import Sequence
from mapFolding import Array1DElephino, Array1DFoldsTotal, Array1DLeavesTotal, Array3D, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal, NumPyIntegerType
from numpy import dtype as numpy_dtype, int64 as numpy_int64, ndarray
from sys import maxsize as sysMaxsize
from typing import Any
from Z0Z_tools import defineConcurrencyLimit, intInnit, oopsieKwargsie
import numpy
import dataclasses

def getLeavesTotal(mapShape: tuple[int, ...]) -> int:
	"""
	Calculate the total number of leaves in a map with the given dimensions.

	The total number of leaves is the product of all dimensions in the map shape.

	Parameters
	----------
	mapShape
		A tuple of integers representing the dimensions of the map.

	Returns
	-------
	leavesTotal
		The total number of leaves in the map, calculated as the product of all dimensions.

	Raises
	------
	OverflowError
		If the product of dimensions would exceed the system's maximum integer size. This check prevents silent numeric
		overflow issues that could lead to incorrect results.
	"""
	productDimensions = 1
	for dimension in mapShape:
		# NOTE this check is one-degree short of absurd, but three lines of early absurdity is better than invalid output later. I'd add more checks if I could think of more.
		if dimension > sysMaxsize // productDimensions:
			raise OverflowError(f"I received `{dimension = }` in `{mapShape = }`, but the product of the dimensions exceeds the maximum size of an integer on this system.")
		productDimensions *= dimension
	return productDimensions

def getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, leavesTotal: int) -> int:
	"""
	Determines whether to divide the computation into tasks and how many divisions.

	Parameters
	----------
	computationDivisions: None
		Specifies how to divide computations: Please see the documentation in `countFolds` for details. I know it is
		annoying, but I want to be sure you have the most accurate information.
	concurrencyLimit
		Maximum number of concurrent tasks allowed.

	Returns
	-------
	taskDivisions
		How many tasks must finish before the job can compute the total number of folds; `0` means no tasks, only job.

	Raises
	------
	ValueError
		If `computationDivisions` is an unsupported type or if resulting task divisions exceed total leaves.

	Notes
	-----
	Task divisions should not exceed total leaves or the folds will be over-counted.
	"""
	taskDivisions = 0
	match computationDivisions:
		case None | 0 | False:
			pass
		case int() as intComputationDivisions:
			taskDivisions = intComputationDivisions
		case str() as strComputationDivisions:
			strComputationDivisions = strComputationDivisions.lower()
			match strComputationDivisions:
				case 'maximum':
					taskDivisions = leavesTotal
				case 'cpu':
					taskDivisions = min(concurrencyLimit, leavesTotal)
				case _:
					raise ValueError(f"I received '{strComputationDivisions}' for the parameter, `computationDivisions`, but the string value is not supported.")
		case _:
			raise ValueError(f"I received {computationDivisions} for the parameter, `computationDivisions`, but the type {type(computationDivisions).__name__} is not supported.")

	if taskDivisions > leavesTotal:
		raise ValueError(f"Problem: `{taskDivisions = }`, is greater than `{leavesTotal = }`, which will cause duplicate counting of the folds.\n\nChallenge: you cannot directly set `taskDivisions` or `leavesTotal`: they are derived from parameters that may or may not be named `computationDivisions`, `CPUlimit` , and `listDimensions` and from my dubious-quality Python code.")
	return int(max(0, taskDivisions))

def _makeConnectionGraph(mapShape: tuple[int, ...], leavesTotal: int) -> ndarray[tuple[int, int, int], numpy_dtype[numpy_int64]]:
	"""
	Implementation of connection graph generation for map folding.

	This is the internal implementation that calculates all possible connections between leaves in a map folding problem
	based on Lunnon's algorithm. The function constructs a three-dimensional array representing which leaves can be
	connected to each other for each dimension of the map.

	Parameters
	----------
	mapShape
		A tuple of integers representing the dimensions of the map.
	leavesTotal
		The total number of leaves in the map.

	Returns
	-------
	connectionGraph
		A 3D NumPy array with shape (`dimensionsTotal`, `leavesTotal`+1, `leavesTotal`+1) where each entry [d,i,j]
		represents the leaf that would be connected to leaf j when inserting leaf i in dimension d.

	Notes
	-----
	This is an implementation detail and shouldn't be called directly by external code. Use `getConnectionGraph`
	instead, which applies proper typing.

	The algorithm calculates a coordinate system first, then determines connections based on parity rules, boundary
	conditions, and dimensional constraints.
	"""
	dimensionsTotal = len(mapShape)
	cumulativeProduct = numpy.multiply.accumulate([1] + list(mapShape), dtype=numpy_int64)
	arrayDimensions = numpy.array(mapShape, dtype=numpy_int64)
	coordinateSystem = numpy.zeros((dimensionsTotal, leavesTotal + 1), dtype=numpy_int64)
	for indexDimension in range(dimensionsTotal):
		for leaf1ndex in range(1, leavesTotal + 1):
			coordinateSystem[indexDimension, leaf1ndex] = (((leaf1ndex - 1) // cumulativeProduct[indexDimension]) % arrayDimensions[indexDimension] + 1)

	connectionGraph = numpy.zeros((dimensionsTotal, leavesTotal + 1, leavesTotal + 1), dtype=numpy_int64)
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

def getConnectionGraph(mapShape: tuple[int, ...], leavesTotal: int, datatype: type[NumPyIntegerType]) -> ndarray[tuple[int, int, int], numpy_dtype[NumPyIntegerType]]:
	"""
	Create a properly typed connection graph for the map folding algorithm.

	This function serves as a typed wrapper around the internal implementation that generates connection graphs. It
	provides the correct type information for the returned array, ensuring consistency throughout the computation
	assembly-line.

	Parameters
	----------
	mapShape
		A tuple of integers representing the dimensions of the map.
	leavesTotal
		The total number of leaves in the map.
	datatype
		The NumPy integer type to use for the array elements, ensuring proper memory usage and compatibility with the
		computation state.

	Returns
	-------
	connectionGraph
		A 3D NumPy array with shape (`dimensionsTotal`, `leavesTotal`+1, `leavesTotal`+1) with the specified `datatype`,
		representing all possible connections between leaves.
	"""
	connectionGraph = _makeConnectionGraph(mapShape, leavesTotal)
	connectionGraph = connectionGraph.astype(datatype)
	return connectionGraph

def makeDataContainer(shape: int | tuple[int, ...], datatype: type[NumPyIntegerType]) -> ndarray[Any, numpy_dtype[NumPyIntegerType]]:
	"""
	Create a typed NumPy array container with initialized values.

	This function centralizes the creation of data containers used throughout the computation assembly-line, enabling
	easy switching between different container types or implementation strategies if needed in the future.

	Parameters
	----------
	shape
		Either an integer (for 1D arrays) or a tuple of integers (for multi-dimensional arrays) specifying the
		dimensions of the array.
	datatype
		The NumPy integer type to use for the array elements, ensuring proper type consistency and memory efficiency.

	Returns
	-------
	container
		A NumPy array of zeros with the specified shape and `datatype`.
	"""
	return numpy.zeros(shape, dtype=datatype)


@dataclasses.dataclass
class ComputationState:
	"""
	Represents the complete state of a map folding computation.

	This dataclass encapsulates all the information required to compute the number of possible ways to fold a map,
	including the map dimensions, leaf connections, computation progress, and fold counting. It serves as the central
	data structure that flows through the entire computational algorithm.

	Fields are categorized into:
	1. Input parameters (`mapShape`, `leavesTotal`, etc.).
	2. Core computational structures (`connectionGraph`, etc.).
	3. Tracking variables for the folding algorithm state.
	4. Result accumulation fields (`foldsTotal`, `groupsOfFolds`).
	"""
	# NOTE Python is anti-DRY, again, `DatatypeLeavesTotal` metadata needs to match the type
	mapShape: tuple[DatatypeLeavesTotal, ...] = dataclasses.field(init=True, metadata={'elementConstructor': 'DatatypeLeavesTotal'})
	"""Dimensions of the map to be folded, as a tuple of integers."""

	leavesTotal: DatatypeLeavesTotal
	"""Total number of leaves (unit squares) in the map, equal to the product of all dimensions."""

	taskDivisions: DatatypeLeavesTotal
	"""Number of parallel tasks to divide the computation into. Zero means sequential computation."""

	concurrencyLimit: DatatypeElephino
	"""Maximum number of concurrent processes to use during computation."""

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'dtype': Array3D.__args__[1].__args__[0]}) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
	"""3D array encoding the connections between leaves in all dimensions."""

	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)
	"""Total number of dimensions in the map shape."""

	# I am using `dataclasses.field` metadata and `typeAlias.__args__[1].__args__[0]` to make the code more DRY. https://github.com/hunterhogan/mapFolding/issues/9
	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Tracks how many dimensions are gapped for each leaf."""

	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Number of dimensions that are not constrained in the current folding state."""

	gapRangeStart: Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DElephino.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Starting index for the gap range for each leaf."""

	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Tracks where gaps occur in the folding pattern."""

	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""For each leaf, stores the index of the leaf above it in the folding pattern."""

	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""For each leaf, stores the index of the leaf below it in the folding pattern."""

	foldGroups: Array1DFoldsTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DFoldsTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Accumulator for fold groups across parallel tasks."""

	foldsTotal: DatatypeFoldsTotal = DatatypeFoldsTotal(0)
	"""The final computed total number of distinct folding patterns."""

	gap1ndex: DatatypeElephino = DatatypeElephino(0)
	"""Current index into gaps array during algorithm execution."""

	gap1ndexCeiling: DatatypeElephino = DatatypeElephino(0)
	"""Upper limit for gap index during the current algorithm phase."""

	groupsOfFolds: DatatypeFoldsTotal = dataclasses.field(default=DatatypeFoldsTotal(0), metadata={'theCountingIdentifier': True})
	"""Accumulator for the number of fold groups found during computation."""

	indexDimension: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""Current dimension being processed during algorithm execution."""

	indexLeaf: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""Current leaf index during iteration."""

	indexMiniGap: DatatypeElephino = DatatypeElephino(0)
	"""Index used when filtering common gaps."""

	leaf1ndex: DatatypeLeavesTotal = DatatypeLeavesTotal(1)
	"""Active leaf being processed in the folding algorithm. Starts at 1, not 0."""

	leafConnectee: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""Leaf that is being connected to the active leaf."""

	taskIndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""Index of the current parallel task when using task divisions."""

	def __post_init__(self) -> None:
		from mapFolding.beDRY import getConnectionGraph, makeDataContainer
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		leavesTotalAsInt = int(self.leavesTotal)
		self.connectionGraph = getConnectionGraph(self.mapShape, leavesTotalAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype'])

		if self.dimensionsUnconstrained is None: self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal)) # pyright: ignore[reportUnnecessaryComparison]

		if self.foldGroups is None: # pyright: ignore[reportUnnecessaryComparison]
			self.foldGroups = makeDataContainer(max(2, int(self.taskDivisions) + 1), self.__dataclass_fields__['foldGroups'].metadata['dtype'])
			self.foldGroups[-1] = self.leavesTotal

		# Dataclasses, Default factories, and arguments in `ComputationState` https://github.com/hunterhogan/mapFolding/issues/12
		if self.gapsWhere is None: self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]

		if self.countDimensionsGapped is None: self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.gapRangeStart is None: self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.leafAbove is None: self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.leafBelow is None: self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]

	# Automatic, or not, calculation in dataclass `ComputationState` https://github.com/hunterhogan/mapFolding/issues/14
	def getFoldsTotal(self) -> None:
		self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)

def outfitCountFolds(mapShape: tuple[int, ...], computationDivisions: int | str | None = None, concurrencyLimit: int = 1) -> ComputationState:
	"""
	Initialize a `ComputationState` with validated parameters for map folding calculation.

	This function serves as the central initialization point for creating a properly configured `ComputationState`
	object, ensuring consistent calculation of the fundamental parameters (`leavesTotal` and `taskDivisions`) across the
	entire package.

	Parameters
	----------
	mapShape
		A tuple of integers representing the dimensions of the map.
	computationDivisions: None
		Controls how to divide the computation into parallel tasks. I know it is annoying, but please see
		`countFolds` for details, so that you and I both know you have the most accurate information.
	concurrencyLimit: 1
		Maximum number of concurrent processes to use during computation.

	Returns
	-------
	computationStateInitialized
		A fully initialized `ComputationState` object that's ready for computation.

	Notes
	-----
	This function maintains the Single Source of Truth principle for `leavesTotal` and `taskDivisions` calculation,
	ensuring these values are derived consistently throughout the package.
	"""
	leavesTotal = getLeavesTotal(mapShape)
	taskDivisions = getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
	computationStateInitialized = ComputationState(mapShape, leavesTotal, taskDivisions, concurrencyLimit)
	return computationStateInitialized

def setProcessorLimit(CPUlimit: Any | None, concurrencyPackage: str | None = None) -> int:
	"""
	Whether and how to limit the CPU usage.

	Parameters
	----------
	CPUlimit: None
		Please see the documentation for in `countFolds` for details. I know it is annoying, but I want to be sure you
		have the most accurate information.
	concurrencyPackage: None
		Specifies which concurrency package to use:
		- `None` or `'multiprocessing'`: Uses standard `multiprocessing`.
		- `'numba'`: Uses Numba's threading system.

	Returns
	-------
	concurrencyLimit
		The actual concurrency limit that was set.

	Raises
	------
	TypeError
		If `CPUlimit` is not of the expected types.
	NotImplementedError
		If `concurrencyPackage` is not supported.

	Notes
	-----
	If using `'numba'` as the concurrency package, the maximum number of processors is retrieved from
	`numba.get_num_threads()` rather than by polling the hardware. If Numba environment variables limit available
	processors, that will affect this function.

	When using Numba, this function must be called before importing any Numba-jitted function for this processor limit
	to affect the Numba-jitted function.
	"""
	if not (CPUlimit is None or isinstance(CPUlimit, (bool, int, float))):
		CPUlimit = oopsieKwargsie(CPUlimit)

	match concurrencyPackage:
		case 'multiprocessing' | None:
			# When to use multiprocessing.set_start_method https://github.com/hunterhogan/mapFolding/issues/6
			concurrencyLimit: int = defineConcurrencyLimit(CPUlimit)
		case 'numba':
			from numba import get_num_threads, set_num_threads
			concurrencyLimit = defineConcurrencyLimit(CPUlimit, get_num_threads())
			set_num_threads(concurrencyLimit)
			concurrencyLimit = get_num_threads()
		case _:
			raise NotImplementedError(f"I received `{concurrencyPackage = }` but I don't know what to do with that.")
	return concurrencyLimit

def validateListDimensions(listDimensions: Sequence[int]) -> tuple[int, ...]:
	"""
	Validate and normalize dimensions for a map folding problem.

	This function serves as the gatekeeper for dimension inputs, ensuring that all map dimensions provided to the
	package meet the requirements for valid computation. It performs multiple validation steps and normalizes the
	dimensions into a consistent format.

	Parameters
	----------
	listDimensions
		A sequence of integers representing the dimensions of the map.

	Returns
	-------
	mapShape
		An _unsorted_ tuple of positive integers representing the validated dimensions.

	Raises
	------
	ValueError
		If the input is empty or contains negative values.
	NotImplementedError
		If fewer than two positive dimensions are provided.
	"""
	if not listDimensions:
		raise ValueError("`listDimensions` is a required parameter.")
	listOFint: list[int] = intInnit(listDimensions, 'listDimensions')
	mapDimensions: list[int] = []
	for dimension in listOFint:
		if dimension <= 0:
			raise ValueError(f"I received `{dimension = }` in `{listDimensions = }`, but all dimensions must be a non-negative integer.")
		mapDimensions.append(dimension)
	if len(mapDimensions) < 2:
		raise NotImplementedError(f"This function requires `{listDimensions = }` to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")

	"""
	I previously sorted the dimensions for a few reasons that may or may not be valid:
		1. After empirical testing, I believe that (2,10), for example, computes significantly faster than (10,2).
		2. Standardization, generally.
		3. If I recall correctly, after empirical testing, I concluded that sorted dimensions always leads to
		non-negative values in the connection graph, but if the dimensions are not in ascending order of magnitude,
		the connection graph might have negative values, which as far as I know, is not an inherent problem, but the
		negative values propagate into other data structures, which requires the datatypes to hold negative values,
		which means I cannot optimize the bit-widths of the datatypes as easily. (And optimized bit-widths helps with
		performance.)

	Furthermore, now that the package includes OEIS A000136, 1 x N stamps/maps, sorting could distort results.
	"""
	# NOTE Do NOT sort the dimensions.
	return tuple(mapDimensions)
