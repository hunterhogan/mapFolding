"""A relatively stable API for oft-needed functionality."""
from mapFolding import (
	computationState,
	getDatatypeModule,
	getPathJobRootDEFAULT,
	hackSSOTdatatype,
	hackSSOTdtype,
	indexMy,
	indexTrack,
	setDatatypeElephino,
	setDatatypeFoldsTotal,
	setDatatypeLeavesTotal,
	setDatatypeModule,
)
from collections.abc import Sequence
from numba import get_num_threads, set_num_threads
from numpy import dtype, integer, ndarray
from numpy.typing import DTypeLike, NDArray
from pathlib import Path
from sys import maxsize as sysMaxsize
from typing import Any
from Z0Z_tools import defineConcurrencyLimit, intInnit, oopsieKwargsie
import numpy
import os

def getFilenameFoldsTotal(mapShape: Sequence[int] | ndarray[tuple[int], dtype[integer[Any]]]) -> str:
	"""Imagine your computer has been counting folds for 70 hours, and when it tries to save your newly discovered value,
	the filename is invalid. I bet you think this function is more important after that thought experiment.

	Make a standardized filename for the computed value `foldsTotal`.

	The filename takes into account
		- the dimensions of the map, aka `mapShape`, aka `listDimensions`
		- no spaces in the filename
		- safe filesystem characters
		- unique extension
		- Python-safe strings:
			- no starting with a number
			- no reserved words
			- no dashes or other special characters
			- uh, I can't remember, but I found some other frustrating limitations
		- if 'p' is still the first character of the filename, I picked that because it was the original identifier for the map shape in Lunnan's code

	Parameters:
		mapShape: A sequence of integers representing the dimensions of the map.

	Returns:
		filenameFoldsTotal: A filename string in format 'pNxM.foldsTotal' where N,M are sorted dimensions
	"""
	return 'p' + 'x'.join(str(dimension) for dimension in sorted(mapShape)) + '.foldsTotal'

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
			if dimension > sysMaxsize // productDimensions:
				raise OverflowError(f"I received {dimension=} in {listDimensions=}, but the product of the dimensions exceeds the maximum size of an integer on this system.")
			productDimensions *= dimension

		return productDimensions

def getPathFilenameFoldsTotal(mapShape: Sequence[int] | ndarray[tuple[int], dtype[integer[Any]]], pathLikeWriteFoldsTotal: str | os.PathLike[str] | None = None) -> Path:
	"""Get a standardized path and filename for the computed value `foldsTotal`.

	If you provide a directory, the function will append a standardized filename. If you provide a filename
	or a relative path and filename, the function will prepend the default path.

	Parameters:
		mapShape: List of dimensions for the map folding problem.
		pathLikeWriteFoldsTotal (pathJobRootDEFAULT): Path, filename, or relative path and filename. If None, uses default path.
			Defaults to None.

	Returns:
		pathFilenameFoldsTotal: Absolute path and filename.
	"""
	pathLikeSherpa = Path(pathLikeWriteFoldsTotal) if pathLikeWriteFoldsTotal is not None else None
	if not pathLikeSherpa:
		pathLikeSherpa = getPathJobRootDEFAULT()
	if pathLikeSherpa.is_dir():
		pathFilenameFoldsTotal = pathLikeSherpa / getFilenameFoldsTotal(mapShape)
	elif pathLikeSherpa.is_absolute():
		pathFilenameFoldsTotal = pathLikeSherpa
	else:
		pathFilenameFoldsTotal = getPathJobRootDEFAULT() / pathLikeSherpa

	pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
	return pathFilenameFoldsTotal

def getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, CPUlimit: bool | float | int | None, listDimensions: Sequence[int]) -> int:
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
	leavesTotal = getLeavesTotal(listDimensions)
	if not computationDivisions:
		pass
	elif isinstance(computationDivisions, int):
		taskDivisions = computationDivisions
	elif isinstance(computationDivisions, str):
		computationDivisions = computationDivisions.lower()
		if computationDivisions == 'maximum':
			taskDivisions = leavesTotal
		elif computationDivisions == 'cpu':
			taskDivisions = min(concurrencyLimit, leavesTotal)
	else:
		raise ValueError(f"I received {computationDivisions} for the parameter, `computationDivisions`, but the so-called programmer didn't implement code for that.")

	if taskDivisions > leavesTotal:
		raise ValueError(f"Problem: `taskDivisions`, ({taskDivisions}), is greater than `leavesTotal`, ({leavesTotal}), which will cause duplicate counting of the folds.\n\nChallenge: you cannot directly set `taskDivisions` or `leavesTotal`. They are derived from parameters that may or may not still be named `computationDivisions`, `CPUlimit` , and `listDimensions` and from dubious-quality Python code.\n\nFor those parameters, I received {computationDivisions=}, {CPUlimit=}, and {listDimensions=}.\n\nPotential solutions: get a different hobby or set `computationDivisions` to a different value.")

	return taskDivisions

def makeConnectionGraph(listDimensions: Sequence[int], **keywordArguments: str | None) -> ndarray[tuple[int, int, int], dtype[integer[Any]]]:
	"""
	Constructs a multi-dimensional connection graph representing the connections between the leaves of a map with the given dimensions.
	Also called a Cartesian product decomposition or dimensional product mapping.

	Parameters
		listDimensions: A sequence of integers representing the dimensions of the map.
		**keywordArguments: Datatype management.

	Returns
		connectionGraph: A 3D numpy array with shape of (dimensionsTotal, leavesTotal + 1, leavesTotal + 1).
	"""
	ImaSetTheDatatype = keywordArguments.get('datatype', None)
	if ImaSetTheDatatype:
		setDatatypeLeavesTotal(ImaSetTheDatatype)
	dtype = hackSSOTdtype('connectionGraph')
	mapShape = validateListDimensions(listDimensions)
	leavesTotal = getLeavesTotal(mapShape)
	arrayDimensions = numpy.array(mapShape, dtype=dtype)
	dimensionsTotal = len(arrayDimensions)

	cumulativeProduct = numpy.multiply.accumulate([1] + mapShape, dtype=dtype)
	coordinateSystem = numpy.zeros((dimensionsTotal, leavesTotal + 1), dtype=dtype)
	for indexDimension in range(dimensionsTotal):
		for leaf1ndex in range(1, leavesTotal + 1):
			coordinateSystem[indexDimension, leaf1ndex] = ( ((leaf1ndex - 1) // cumulativeProduct[indexDimension]) % arrayDimensions[indexDimension] + 1 )

	connectionGraph: ndarray[tuple[int, int, int], numpy.dtype[integer[Any]]] = numpy.zeros((dimensionsTotal, leavesTotal + 1, leavesTotal + 1), dtype=dtype)
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

def makeDataContainer(shape: int | tuple[int, ...], datatype: DTypeLike | None = None) -> NDArray[integer[Any]]:
	"""Create a zeroed-out `ndarray` with the given shape and datatype.

	Parameters:
		shape: The shape of the array. Can be an integer for 1D arrays
			or a tuple of integers for multi-dimensional arrays.
		datatype ('dtypeFoldsTotal'): The desired data type for the array.
			If `None`, defaults to 'dtypeFoldsTotal'. Defaults to None.

	Returns:
		dataContainer: A new array of given shape and type, filled with zeros.

	Notes:
		If a version of the algorithm were to use something other than numpy, such as JAX or CUDA, because other
		functions use this function, it would be much easier to change the datatype "ecosystem".
	"""
	numpyDtype = datatype or hackSSOTdtype('dtypeFoldsTotal')
	if 'numpy' == getDatatypeModule():
		return numpy.zeros(shape, dtype=numpyDtype)
	else:
		raise NotImplementedError("Somebody done broke it.")

def outfitCountFolds(listDimensions: Sequence[int], computationDivisions: int | str | None = None, CPUlimit: bool | float | int | None = None, **keywordArguments: str | bool | None) -> computationState:
	"""
	Initializes and configures the computation state for map folding computations.

	Parameters:
		listDimensions: The dimensions of the map to be folded
		computationDivisions (None): see `getTaskDivisions`
		CPUlimit (None): see `setCPUlimit`
		**keywordArguments: Datatype management, it's complicated: see the code below.

	Returns:
		stateInitialized: The initialized computation state
	"""
	# keywordArguments START
	kwourGrapes = keywordArguments.get('sourGrapes', None)
	if kwourGrapes:
		sourGrapes = True
	else:
		sourGrapes = False

	ImaSetTheDatatype = keywordArguments.get('datatypeElephino', None)
	if ImaSetTheDatatype:
		ImaSetTheDatatype = str(ImaSetTheDatatype)
		setDatatypeElephino(ImaSetTheDatatype, sourGrapes)

	ImaSetTheDatatype = keywordArguments.get('datatypeFoldsTotal', None)
	if ImaSetTheDatatype:
		ImaSetTheDatatype = str(ImaSetTheDatatype)
		setDatatypeFoldsTotal(ImaSetTheDatatype, sourGrapes)

	ImaSetTheDatatype = keywordArguments.get('datatypeLeavesTotal', None)
	if ImaSetTheDatatype:
		ImaSetTheDatatype = str(ImaSetTheDatatype)
		setDatatypeLeavesTotal(ImaSetTheDatatype, sourGrapes)

	# NOTE well: this might be only hypothetical because as of this writing, `makeDataContainer` only makes numpy.zeros. But it's here in case things change.
	ImaSetTheDatatype = keywordArguments.get('datatypeModule', None)
	if ImaSetTheDatatype:
		ImaSetTheDatatype = str(ImaSetTheDatatype)
		setDatatypeModule(ImaSetTheDatatype, sourGrapes)
	# keywordArguments END

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
		connectionGraph = makeConnectionGraph(mapShape, datatype=hackSSOTdatatype('connectionGraph')),
		foldGroups = foldGroups,
		mapShape = numpy.array(mapShape, dtype=hackSSOTdtype('mapShape')),
		my = my,
		gapsWhere = makeDataContainer(int(leavesTotal) * int(leavesTotal) + 1, hackSSOTdtype('gapsWhere')),
		track = makeDataContainer((len(indexTrack), leavesTotal + 1), hackSSOTdtype('track')),
		)

	return stateInitialized

def parseDimensions(dimensions: Sequence[int], parameterName: str = 'listDimensions') -> list[int]:
	"""
	Parse and validate the dimensions are non-negative integers.

	Parameters:
		dimensions: Sequence of integers representing dimensions.
		parameterName ('listDimensions'): Name of the parameter for error messages. Defaults to 'listDimensions'.
	Returns:
		listNonNegative: List of validated non-negative integers.
	Raises:
		ValueError: If any dimension is negative or if the list is empty.
		TypeError: If any element cannot be converted to integer (raised by `intInnit`).
	"""
	listValidated = intInnit(dimensions, parameterName)
	listNonNegative = []
	for dimension in listValidated:
		if dimension < 0:
			raise ValueError(f"Dimension {dimension} must be non-negative")
		listNonNegative.append(dimension)

	return listNonNegative

def saveFoldsTotal(pathFilename: str | os.PathLike[str], foldsTotal: int) -> None:
	"""
	Save foldsTotal with multiple fallback mechanisms.

	Parameters:
		pathFilename: Target save location
		foldsTotal: Critical computed value to save
	"""
	try:
		pathFilenameFoldsTotal = Path(pathFilename)
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

def setCPUlimit(CPUlimit: Any | None) -> int:
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
	set_num_threads(concurrencyLimit)
	concurrencyLimit = get_num_threads()

	return concurrencyLimit

def validateListDimensions(listDimensions: Sequence[int]) -> list[int]:
	"""
	Validates and sorts a sequence of at least two positive dimensions.

	Parameters:
		listDimensions: A sequence of integer dimensions to be validated.

	Returns:
		dimensionsValidSorted: A list, with at least two elements, of only positive integers.

	Raises:
		ValueError: If the input listDimensions is empty.
		NotImplementedError: If the resulting list of positive dimensions has fewer than two elements.
	"""
	if not listDimensions:
		raise ValueError("listDimensions is a required parameter.")
	listNonNegative = parseDimensions(listDimensions, 'listDimensions')
	dimensionsValid = [dimension for dimension in listNonNegative if dimension > 0]
	if len(dimensionsValid) < 2:
		raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")
	return sorted(dimensionsValid)
