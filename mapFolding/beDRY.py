"""
Utility functions for maintaining DRY (Don't Repeat Yourself) principles in the mapFolding package.

This module provides a collection of helper functions that abstract common operations needed
throughout the package, preventing code duplication and ensuring consistency. The functions
manage core aspects of the computation process, including:

1. Resource allocation and system limits management
2. Data structure initialization and manipulation
3. Parameter validation and interpretation
4. Construction of specialized arrays and matrices for the folding algorithm

The functions in this module serve as a relatively stable API for other modules to use,
particularly for initializing computation state, validating inputs, and creating data
structures needed by the folding algorithms.
"""
from collections.abc import Sequence
from mapFolding.theSSOT import Array3D, ComputationState, getDatatypePackage, getNumpyDtypeDefault
from sys import maxsize as sysMaxsize
from typing import Any
from Z0Z_tools import defineConcurrencyLimit, intInnit, oopsieKwargsie
import numpy

def getLeavesTotal(mapShape: tuple[int, ...]) -> int:
	productDimensions = 1
	for dimension in mapShape:
		if dimension > sysMaxsize // productDimensions:
			raise OverflowError(f"I received {dimension=} in {mapShape=}, but the product of the dimensions exceeds the maximum size of an integer on this system.")
		productDimensions *= dimension
	return productDimensions

def getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, leavesTotal: int) -> int:
	"""
	Determines whether to divide the computation into tasks and how many divisions.

	Parameters
	----------
	computationDivisions (None)
		Specifies how to divide computations:
		- `None`: no division of the computation into tasks; sets task divisions to 0.
		- int: direct set the number of task divisions; cannot exceed the map's total leaves.
		- `'maximum'`: divides into `leavesTotal`-many `taskDivisions`.
		- `'cpu'`: limits the divisions to the number of available CPUs, i.e. `concurrencyLimit`.
	concurrencyLimit
		Maximum number of concurrent tasks allowed.
	CPUlimit
		for error reporting.
	listDimensions
		for error reporting.

	Returns
	-------
	taskDivisions
		How many tasks must finish before the job can compute the total number of folds; `0` means no tasks, only job.

	Raises
	------
	ValueError
		If computationDivisions is an unsupported type or if resulting task divisions exceed total leaves.

	Notes
	-----
	Task divisions should not exceed total leaves or the folds will be over-counted.
	"""
	taskDivisions = 0
	if not computationDivisions:
		pass
	elif isinstance(computationDivisions, int):
		taskDivisions = computationDivisions
	elif isinstance(computationDivisions, str): # type: ignore
		# 'Unnecessary isinstance call; "str" is always an instance of "str", so sayeth Pylance'. Yeah, well "User is not always an instance of "correct input" so sayeth the programmer.
		computationDivisions = computationDivisions.lower()
		if computationDivisions == 'maximum':
			taskDivisions = leavesTotal
		elif computationDivisions == 'cpu':
			taskDivisions = min(concurrencyLimit, leavesTotal)
	else:
		raise ValueError(f"I received {computationDivisions} for the parameter, `computationDivisions`, but the so-called programmer didn't implement code for that.")

	if taskDivisions > leavesTotal:
		raise ValueError(f"Problem: `taskDivisions`, ({taskDivisions}), is greater than `leavesTotal`, ({leavesTotal}), which will cause duplicate counting of the folds.\n\nChallenge: you cannot directly set `taskDivisions` or `leavesTotal`. They are derived from parameters that may or may not still be named `computationDivisions`, `CPUlimit` , and `listDimensions` and from dubious-quality Python code.")
	return int(max(0, taskDivisions))

def interpretParameter_datatype(datatype: type[numpy.signedinteger[Any]] | None = None) -> type[numpy.signedinteger[Any]]:
	"""An imperfect way to reduce code duplication."""
	if 'numpy' == getDatatypePackage():
		numpyDtype = datatype or getNumpyDtypeDefault()
	else:
		raise NotImplementedError("Somebody done broke it.")
	return numpyDtype

def makeConnectionGraph(mapShape: tuple[int, ...], leavesTotal: int, datatype: type[numpy.signedinteger[Any]] | None = None) -> Array3D:
	numpyDtype = interpretParameter_datatype(datatype)
	dimensionsTotal = len(mapShape)
	cumulativeProduct = numpy.multiply.accumulate([1] + list(mapShape), dtype=numpyDtype)
	arrayDimensions = numpy.array(mapShape, dtype=numpyDtype)
	coordinateSystem = numpy.zeros((dimensionsTotal, leavesTotal + 1), dtype=numpyDtype)
	for indexDimension in range(dimensionsTotal):
		for leaf1ndex in range(1, leavesTotal + 1):
			coordinateSystem[indexDimension, leaf1ndex] = (((leaf1ndex - 1) // cumulativeProduct[indexDimension]) % arrayDimensions[indexDimension] + 1)

	connectionGraph = numpy.zeros((dimensionsTotal, leavesTotal + 1, leavesTotal + 1), dtype=numpyDtype)
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

def makeDataContainer(shape: int | tuple[int, ...], datatype: type[numpy.signedinteger[Any]] | None = None) -> numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[Any]]]:
	numpyDtype = interpretParameter_datatype(datatype)
	return numpy.zeros(shape, dtype=numpyDtype)

def outfitCountFolds(mapShape: tuple[int, ...], computationDivisions: int | str | None = None, concurrencyLimit: int = 1) -> ComputationState:
	leavesTotal = getLeavesTotal(mapShape)
	taskDivisions = getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
	computationStateInitialized = ComputationState(mapShape, leavesTotal, taskDivisions, concurrencyLimit)
	return computationStateInitialized

def setCPUlimit(CPUlimit: Any | None) -> int:
	"""Sets CPU limit for concurrent operations.

	If the concurrency is managed by `numba`, the maximum number of CPUs is retrieved from `numba.get_num_threads()` and not by polling the hardware. Therefore, if there are
	numba environment variables limiting the number of available CPUs, that will effect this function. That _should_ be a good thing: you control the number of CPUs available
	to numba. But if you're not aware of that, you might be surprised by the results.

	If you are designing custom modules that use numba, note that you must call `numba.set_num_threads()` (i.e., this function) before executing an `import` statement
	on a Numba-jitted function. Otherwise, the `numba.set_num_threads()` call will have no effect on the imported function.

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

	from mapFolding.theSSOT import concurrencyPackage
	if concurrencyPackage == 'numba':
		from numba import get_num_threads, set_num_threads
		concurrencyLimit: int = defineConcurrencyLimit(CPUlimit, get_num_threads())
		set_num_threads(concurrencyLimit)
		concurrencyLimit = get_num_threads()
	elif concurrencyPackage == 'multiprocessing':
		# When to use multiprocessing.set_start_method https://github.com/hunterhogan/mapFolding/issues/6
		concurrencyLimit = defineConcurrencyLimit(CPUlimit)
	else:
		raise NotImplementedError(f"I received {concurrencyPackage=} but I don't know what to do with that.")
	return concurrencyLimit

def validateListDimensions(listDimensions: Sequence[int]) -> tuple[int, ...]:
	if not listDimensions:
		raise ValueError("listDimensions is a required parameter.")
	listValidated: list[int] = intInnit(listDimensions, 'listDimensions')
	listNonNegative: list[int] = []
	for dimension in listValidated:
		if dimension < 0:
			raise ValueError(f"Dimension {dimension} must be non-negative")
		listNonNegative.append(dimension)
	dimensionsValid = [dimension for dimension in listNonNegative if dimension > 0]
	if len(dimensionsValid) < 2:
		raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")
	return tuple(sorted(dimensionsValid))
