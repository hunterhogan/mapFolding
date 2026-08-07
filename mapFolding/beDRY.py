"""Oft-needed computations or actions, especially for multi-dimensional map folding."""
from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from hunterMakesPy import inclusive
from hunterMakesPy.parseParameters import defineConcurrencyLimit, intInnit
from numpy import int64 as numpy_int64
from sys import maxsize as sysMaxsize
from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from mapFolding import Array1DLeavesTotal, Array2DLeavesTotal, Array3DLeavesTotal, NumPyIntegerType
	from numpy import dtype as numpy_dtype, ndarray
	from typing import Any

#======== Parse parameters ======================================

def defineProcessorLimit(CPUlimit: Limitation, concurrencyPackage: str | None = None) -> int:
	"""Compute the CPU usage limit for concurrent operations; for `numba` managed concurrency, set the global limit.

	Parameters
	----------
	CPUlimit : bool | float | int | None
		Please see the documentation in `countFolds` for details. I know it is annoying, but I want to
		be sure you have the most accurate information.
	concurrencyPackage : str | None = None
		Specifies which concurrency package to use. - `None` or `'multiprocessing'`: Uses standard
		`multiprocessing`. - `'numba'`: Uses Numba's threading system.

	Returns
	-------
	concurrencyLimit : int
		The actual concurrency limit that was set.

	Numba
	-----
	If using `'numba'` as the concurrency package, the maximum number of processors is retrieved from
	`numba.get_num_threads()` rather than by polling the hardware. If Numba environment variables
	limit available processors, that will affect this function.

	When using Numba, this function must be called before importing any Numba-jitted function for this
	processor limit to affect the Numba-jitted function.
	"""
	match concurrencyPackage:
		case 'numba':
			#=Sin= `numba` is optional.
			# ruff: ignore[import-outside-top-level]
			from mapFolding._optionalNumba import defineProcessorLimitNumba
			concurrencyLimit: int = defineProcessorLimitNumba(CPUlimit)
		case 'multiprocessing' | _:
			concurrencyLimit = defineConcurrencyLimit(limit=CPUlimit)
	return concurrencyLimit

def getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, leavesTotal: int) -> int:
	"""Determine whether to divide the computation into tasks and how many divisions.

	Parameters
	----------
	computationDivisions : int | str | None
		Specifies how to divide computations. Please see the documentation in `countFolds` for
		details. I know it is annoying, but I want to be sure you have the most accurate information.
	concurrencyLimit : int
		Maximum number of concurrent tasks allowed.
	leavesTotal : int
		Total number of leaves in the map.

	Returns
	-------
	taskDivisions : int
		How many tasks must finish before the job can compute the total number of folds. `0` means no
		tasks, only job.

	Raises
	------
	ValueError
		If `computationDivisions` is an unsupported type or if resulting task divisions exceed total
		leaves.

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
					taskDivisions: int = leavesTotal
				case 'cpu':
					taskDivisions = min(concurrencyLimit, leavesTotal)
				case _:
					message: str = f"I received '{strComputationDivisions}' for the parameter, `computationDivisions`, but the string value is not supported."
					raise ValueError(message)
		case _:
			message = f"I received {computationDivisions} for the parameter, `computationDivisions`, but the type {type(computationDivisions).__name__} is not supported."
			raise ValueError(message)

	if taskDivisions > leavesTotal:
		message = (
			f"I derived `{taskDivisions = }`, which is greater than `{leavesTotal = }`, but task divisions cannot exceed the map's "
			"total leaves because that would count folds more than once."
		)
		raise ValueError(message)
	return int(max(0, taskDivisions))

def validateMapShape(mapShape: Sequence[int]) -> tuple[int, ...]:
	"""Validate and normalize a map shape for a map-folding problem.

	(AI generated docstring)

	This function serves as the gatekeeper for dimension inputs, ensuring that all map dimensions
	provided to the package meet the requirements for valid computation. It performs multiple
	validation steps and normalizes the dimensions into a consistent format.

	Parameters
	----------
	mapShape : Sequence[int]
		A sequence of integers representing the dimensions of the map.

	Returns
	-------
	mapShape : tuple[int, ...]
		An _unsorted_ tuple of positive integers representing the validated dimensions.

	Raises
	------
	ValueError
		If the input is empty or contains non-positive values.
	"""
	mapShapeAsList: list[int] = intInnit(mapShape, 'mapShape', Sequence[int])
	if not mapShapeAsList or any(map((0).__gt__, mapShapeAsList)):
		message: str = f"I received `{mapShape = }`, but I need at least one positive integer dimension."
		raise ValueError(message)

	#=EndNotes##sortingDimensions=
	# Do NOT sort the dimensions.
	return tuple(mapShapeAsList)

#======== map folding ===================================

def getConnectionGraph(mapShape: tuple[int, ...], leavesTotal: int, datatype: type[NumPyIntegerType]) -> ndarray[tuple[int, int, int], numpy_dtype[NumPyIntegerType]]:
	"""Create a properly typed connection graph for the map folding algorithm.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.
	leavesTotal : int
		The total number of leaves in the map.
	datatype : type[NumPyIntegerType]
		The NumPy integer type to use for the array elements, ensuring proper memory usage and
		compatibility with the computation state.

	Returns
	-------
	connectionGraph : ndarray[tuple[int, int, int], numpy_dtype[NumPyIntegerType]]
		A 3D NumPy array with shape (`dimensionsTotal`, `leavesTotal`+1, `leavesTotal`+1) with the
		specified `datatype`, representing all possible connections between leaves.
	"""
	connectionGraph: Array3DLeavesTotal = _makeConnectionGraph(mapShape, leavesTotal)
	return connectionGraph.astype(datatype)

@cache
def getLeavesTotal(mapShape: tuple[int, ...]) -> int:
	"""The definitive calculation of the total number of leaves in a map with the given dimensions.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers with the length of each dimension of the map.

	Returns
	-------
	leavesTotal : int
		The definitive total number of leaves in the map.

	Raises
	------
	OverflowError
		If the product of dimensions would exceed the system's maximum integer size. This check
		prevents silent numeric overflow issues that could lead to incorrect results.

	Notes
	-----
	It is impossible to overstate the importance of `leavesTotal` in every algorithm for counting
	folds. Therefore, in this package, this function is the ***only*** permissible way to compute
	`leavesTotal`.

	The total number of leaves is the product of all dimensions in `mapShape`.
	"""
	productDimensions = 1
	for dimension in mapShape:
		#=EndNotes##absurd=
		if dimension > sysMaxsize // productDimensions:
			message: str = f"I received `{dimension = }` in `{mapShape = }`, but the product of the dimensions exceeds the maximum size of an integer on this system."
			raise OverflowError(message)
		productDimensions *= dimension
	return productDimensions

def _makeConnectionGraph(mapShape: tuple[int, ...], leavesTotal: int) -> ndarray[tuple[int, int, int], numpy_dtype[numpy_int64]]:
	"""Implement connection graph generation for map folding.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.
	leavesTotal : int
		The total number of leaves in the map.

	Returns
	-------
	connectionGraph : ndarray[tuple[int, int, int], numpy_dtype[numpy_int64]]
		A 3D NumPy array with shape (`dimensionsTotal`, `leavesTotal`+1, `leavesTotal`+1) where each
		entry [d,i,j] represents the leaf that would be connected to leaf j when inserting leaf i in
		dimension d.

	Notes
	-----
	This is an implementation detail and shouldn't be called directly by external code. Use
	`getConnectionGraph` instead, which applies proper typing.

	The algorithm calculates a coordinate system first, then determines connections based on parity
	rules, boundary conditions, and dimensional constraints.
	"""
	dimensionsTotal: int = len(mapShape)
	cumulativeProduct: Array1DLeavesTotal = numpy.multiply.accumulate([1, *list(mapShape)], dtype=numpy_int64)
	arrayDimensions: Array1DLeavesTotal = numpy.array(mapShape, dtype=numpy_int64)
	coordinateSystem: Array2DLeavesTotal = numpy.zeros((dimensionsTotal, leavesTotal + 1), dtype=numpy_int64)
	for indexDimension in range(dimensionsTotal):
		for leaf1ndex in range(1, leavesTotal + inclusive):
			coordinateSystem[indexDimension, leaf1ndex] = (((leaf1ndex - 1) // cumulativeProduct[indexDimension]) % arrayDimensions[indexDimension] + 1)

	connectionGraph: Array3DLeavesTotal = numpy.zeros((dimensionsTotal, leavesTotal + 1, leavesTotal + 1), dtype=numpy_int64)
	for indexDimension in range(dimensionsTotal):
		for activeLeaf1ndex in range(1, leavesTotal + inclusive):
			for connectee1ndex in range(1, activeLeaf1ndex + inclusive):
				isFirstCoord: bool = coordinateSystem[indexDimension, connectee1ndex] == 1
				isLastCoord: bool = coordinateSystem[indexDimension, connectee1ndex] == arrayDimensions[indexDimension]
				exceedsActive: bool = connectee1ndex + cumulativeProduct[indexDimension] > activeLeaf1ndex
				isEvenParity: bool = (coordinateSystem[indexDimension, activeLeaf1ndex] & 1) == (coordinateSystem[indexDimension, connectee1ndex] & 1)

				if (isEvenParity and isFirstCoord) or (not isEvenParity and (isLastCoord or exceedsActive)):
					connectionGraph[indexDimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex
				elif isEvenParity and not isFirstCoord:
					connectionGraph[indexDimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex - cumulativeProduct[indexDimension]
				elif not isEvenParity and not (isLastCoord or exceedsActive):
					connectionGraph[indexDimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex + cumulativeProduct[indexDimension]
	return connectionGraph

def makeDataContainer(shape: int | tuple[Any, ...], datatype: type[NumPyIntegerType]) -> ndarray[Any, numpy_dtype[NumPyIntegerType]]:
	"""Create any data container as long as it is a `numpy.ndarray` full of zeroes of type `numpy.integer`.

	By centralizing data container creation, you can more easily make global changes.

	Parameters
	----------
	shape : int | tuple[Any, ...]
		The array shape, either as a single axis length or a tuple of axes lengths.
	datatype : type[NumPyIntegerType]
		The `numpy.integer` type for the array elements.

	Returns
	-------
	container : ndarray[Any, numpy_dtype[NumPyIntegerType]]
		A zero-filled `ndarray` with the specified `shape` and `datatype`.

	"""
	return numpy.zeros(shape, dtype=datatype)

def mapShapeIs2上nDimensions(mapShape: tuple[int, ...], *, youMustBeDimensionsTallToRideThis: int = 3) -> bool:
	"""Test whether `mapShape` is a sufficiently sized 2ⁿ-dimensional map.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.
	youMustBeDimensionsTallToRideThis : int = 3
		Minimum number of required dimensions.

	Returns
	-------
	is2上nDimensions : bool
		`True` when `mapShape` is a 2ⁿ-dimensional map with the required minimum dimension count.
	"""
	return (youMustBeDimensionsTallToRideThis <= len(mapShape)) and all(map((2).__eq__, mapShape))
