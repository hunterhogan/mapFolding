"""A relatively stable API for oft-needed functionality."""
from mapFolding import (
	Array1DElephino,
	Array1DFoldsTotal,
	Array1DLeavesTotal,
	Array3D,
	DatatypeElephino,
	DatatypeFoldsTotal,
	DatatypeLeavesTotal,
	numpyElephino,
	numpyFoldsTotal,
	numpyLeavesTotal,
)
from mapFolding import getDatatypeModule, getNumpyDtypeDefault
from Z0Z_tools import defineConcurrencyLimit, intInnit, oopsieKwargsie
from collections.abc import Sequence
from numba import get_num_threads, set_num_threads
from numpy import dtype, integer, ndarray
from numpy.typing import DTypeLike
from pathlib import Path
from sys import maxsize as sysMaxsize
from typing import Any
import dataclasses
import numpy
import numpy.typing
import os

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

def getLeavesTotal(mapShape: tuple[int, ...]) -> int:
	productDimensions = 1
	for dimension in mapShape:
		if dimension > sysMaxsize // productDimensions:
			raise OverflowError(f"I received {dimension=} in {mapShape=}, but the product of the dimensions exceeds the maximum size of an integer on this system.")
		productDimensions *= dimension
	return productDimensions

def makeConnectionGraph(mapShape: tuple[int, ...], leavesTotal: int, datatype: DTypeLike | None = None):
	if 'numpy' == getDatatypeModule():
		numpyDtype = datatype or getNumpyDtypeDefault()
	else:
		raise NotImplementedError("Somebody done broke it.")
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

def makeDataContainer(shape: int | tuple[int, ...], datatype: DTypeLike | None = None):
	# ChatGPT (4o reasoning?): "Tip: Create them with functions like np.empty(...) or np.zeros(...) to ensure contiguous memory layout."
	if 'numpy' == getDatatypeModule():
		numpyDtype = datatype or getNumpyDtypeDefault()
		return numpy.zeros(shape, dtype=numpyDtype)
	else:
		raise NotImplementedError("Somebody done broke it.")

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

	concurrencyLimit: int = int(defineConcurrencyLimit(CPUlimit))
	from mapFolding import concurrencyPackage
	if concurrencyPackage == 'numba':
		set_num_threads(concurrencyLimit)
		concurrencyLimit = get_num_threads()
	else:
		raise NotImplementedError("This function only supports the 'numba' concurrency package.")

	return concurrencyLimit

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
	elif isinstance(computationDivisions, str): # type: ignore 'Unnecessary isinstance call; "str" is always an instance of "str", so sayeth Pylance'. Yeah, well "User is not always an instance of "correct input" so sayeth the programmer.
		computationDivisions = computationDivisions.lower()
		if computationDivisions == 'maximum':
			taskDivisions = leavesTotal
		elif computationDivisions == 'cpu':
			taskDivisions = min(concurrencyLimit, leavesTotal)
	else:
		raise ValueError(f"I received {computationDivisions} for the parameter, `computationDivisions`, but the so-called programmer didn't implement code for that.")

	if taskDivisions > leavesTotal:
		raise ValueError(f"Problem: `taskDivisions`, ({taskDivisions}), is greater than `leavesTotal`, ({leavesTotal}), which will cause duplicate counting of the folds.\n\nChallenge: you cannot directly set `taskDivisions` or `leavesTotal`. They are derived from parameters that may or may not still be named `computationDivisions`, `CPUlimit` , and `listDimensions` and from dubious-quality Python code.")
	return taskDivisions

@dataclasses.dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
class ComputationState:
	mapShape: tuple[DatatypeLeavesTotal, ...]
	leavesTotal: DatatypeLeavesTotal
	taskDivisions: DatatypeLeavesTotal

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'description': 'A 3D array representing the connection graph of the map.'})
	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(init=False)
	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)
	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(init=False)
	foldGroups: Array1DFoldsTotal = dataclasses.field(init=False)
	foldsTotal: DatatypeFoldsTotal = DatatypeFoldsTotal(0)
	gap1ndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	gap1ndexCeiling: DatatypeElephino = DatatypeElephino(0)
	gapRangeStart: Array1DElephino = dataclasses.field(init=False)
	gapsWhere: Array1DLeavesTotal = dataclasses.field(init=False)
	groupsOfFolds: DatatypeFoldsTotal = DatatypeFoldsTotal(0)
	indexDimension: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexLeaf: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexMiniGap: DatatypeElephino = DatatypeElephino(0)
	leaf1ndex: DatatypeElephino = DatatypeElephino(1)
	leafAbove: Array1DLeavesTotal = dataclasses.field(init=False)
	leafBelow: Array1DLeavesTotal = dataclasses.field(init=False)
	leafConnectee: DatatypeElephino = DatatypeElephino(0)
	taskIndex: DatatypeLeavesTotal = dataclasses.field(default=DatatypeLeavesTotal(0), metadata={'myType': DatatypeLeavesTotal})
	# taskIndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)

	def __post_init__(self):
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal))
		self.connectionGraph = makeConnectionGraph(self.mapShape, self.leavesTotal, numpyLeavesTotal)

		# the dtype is defined above
		self.foldGroups = makeDataContainer(max(2, int(self.taskDivisions) + 1), numpyFoldsTotal)
		self.foldGroups[-1] = self.leavesTotal

		leavesTotalAsInt = int(self.leavesTotal)
		self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, numpyElephino)
		self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)
		self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, numpyLeavesTotal)
		self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)
		self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)

	def getFoldsTotal(self):
		self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)

	# factory? constructor?
	# state.taskIndex = state.taskIndex.type(indexSherpa)
	# self.fieldName = self.fieldName.fieldType(indexSherpa)
	# state.taskIndex.toMyType(indexSherpa)

def outfitCountFolds(mapShape: tuple[int, ...], computationDivisions: int | str | None = None, concurrencyLimit: int = 1) -> ComputationState:
	leavesTotal = getLeavesTotal(mapShape)
	taskDivisions = getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
	computationStateInitialized = ComputationState(mapShape, leavesTotal, taskDivisions)
	return computationStateInitialized

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
	return None

def getFilenameFoldsTotal(mapShape: Sequence[int] | ndarray[tuple[int], dtype[integer[Any]]]) -> str:
	"""Imagine your computer has been counting folds for 9 days, and when it tries to save your newly discovered value,
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
		filenameFoldsTotal: A filename string in format 'pMxN.foldsTotal' where M,N are sorted dimensions
	"""
	return 'p' + 'x'.join(str(dimension) for dimension in sorted(mapShape)) + '.foldsTotal'

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
	from mapFolding import getPathJobRootDEFAULT
	pathLikeSherpa = Path(pathLikeWriteFoldsTotal) if pathLikeWriteFoldsTotal is not None else None
	if not pathLikeSherpa:
		pathLikeSherpaIsNotNone: Path = getPathJobRootDEFAULT()
	else:
		pathLikeSherpaIsNotNone: Path = pathLikeSherpa
	if pathLikeSherpaIsNotNone.is_dir():
		pathFilenameFoldsTotal = pathLikeSherpaIsNotNone / getFilenameFoldsTotal(mapShape)
	elif pathLikeSherpaIsNotNone.is_absolute():
		pathFilenameFoldsTotal = pathLikeSherpaIsNotNone
	else:
		pathFilenameFoldsTotal = getPathJobRootDEFAULT() / pathLikeSherpaIsNotNone

	pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
	return pathFilenameFoldsTotal
