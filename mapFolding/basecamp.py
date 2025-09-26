"""Unified interface for map folding computation."""

from collections.abc import Sequence
from functools import cache
from mapFolding import (
	getPathFilenameFoldsTotal, packageSettings, saveFoldsTotal, saveFoldsTotalFAILearly, validateListDimensions)
from os import PathLike
from pathlib import PurePath
from typing import Literal

# ruff: noqa: PLC0415
"""TODO new flow paradigm, incomplete

I don't want to FORCE people to use the meaningless OEIS ids without providing the definition of the ID at the same time.

On the other hand, I don't have any evidence that anyone is using this package except me.

algorithms directory
	manually coded algorithms or formulas
	`countFolds` will be a stable interface for multidimensional map folding, including synthetic modules
		This has special treatment because people may want to call mapShape not defined in OEIS
	`countMeanders` will be a stable interface for meanders
		This has special treatment because people may want to call meanders not defined in OEIS
	an enhanced version of `oeisIDfor_n` will be a stable interface for calling by ID and n

General flow structure
	doTheNeedful
		specific to that version of that algorithm
		abstracts the API for that algorithm, so that algorithm (such as multidimensional map folding) has a stable interface
		The last place to do defensive programming

- Incomplete: how to count
	- currently in parameters computationDivisions, CPUlimit, and flow

- Flow in count______
	- DEFENSIVE PROGRAMMING
	- FAIL EARLY
	- Implement "common foundational logic".
		- IDK what the correct technical term is, but I'm sure other people have researched excellent ways to do this.
		- Example: in `countFolds`, every possible flow path needs `mapShape`. Therefore, `mapShape` is foundational logic that
			all flow paths have in common: "common foundational logic".
		- Example: in `countFolds`, some flow paths have more than one "task division" (i.e., the computation is divided into
			multiple tasks), while other flow paths only have one task division. One reasonable perspective is that computing task
			divisions is NOT "common foundational logic". My perspective for this example: to compute whether or not there are
			task divisions and if so, how many task divisions is identical for all flow paths. Therefore, I handle computing task
			divisions as "common foundational logic".
		- Incomplete
	- Initialize memorialization instructions, if asked
	- MORE DEFENSIVE PROGRAMMING
	- FAIL EARLIER THAN EARLY
	- Incomplete
	- DEFENSIVE PROGRAMMING ON BEHALF of downstream modules and functions
	- FAIL SO EARLY IT IS BEFORE THE USER INSTALLS THE APP
	- Incomplete
	- REPEAT MANY OR ALL OF THE DEFENSIVE PROGRAMMING YOU HAVE ALREADY DONE

	- Incomplete
	- Pass control to the correct `doTheNeedful`
	- I don't know how to "elegantly" pass control without putting `doTheNeedful` over `count______` in the stack, therefore,
		control will come back here.
	- DO NOT, for the love of puppies and cookies, DO NOT use defensive programming here. Defensive programming AFTER a
		four-week-long computation is a tacit admission of incompetent programming.
	- Follow memorialization instructions: which means pass control to a function will tenaciously follow the instructions.
	- return "a(n)" (as OEIS calls it), such as foldsTotal

"""

# Parameters
	# What you want to compute
	# Memorialization
	# Concurrency
	# How you want to compute it
# Interpretation of parameters
	# Input data

def countFolds(listDimensions: Sequence[int] | None = None
				, pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
				, computationDivisions: int | str | None = None
				# , * # TODO improve `standardizedEqualToCallableReturn` so it will work with keyword arguments
				, CPUlimit: bool | float | int | None = None  # noqa: FBT001
				, mapShape: tuple[int, ...] | None = None
				, flow: str | None = None
				) -> int:
	"""
	Count the total number of distinct ways to fold a map.

	Mathematicians also describe this as folding a strip of stamps, and they usually call the total "number of distinct ways to
	fold" a map the map's "foldings."

	Parameters
	----------
	listDimensions : Sequence[int] | None = None
		List of integers representing the dimensions of the map to be folded.
	pathLikeWriteFoldsTotal : PathLike[str] | PurePath | None = None
		A filename, a path of only directories, or a path with directories and a filename to which `countFolds` will write the
		value of `foldsTotal`. If `pathLikeWriteFoldsTotal` is a path of only directories, `countFolds` creates a filename based
		on the map dimensions.
	computationDivisions : int | str | None = None
		Whether and how to divide the computational work.
		- `None`: no division of the computation into tasks.
		- `int`: into how many tasks `countFolds` will divide the computation. The values 0 or 1 are identical to `None`. It is
		mathematically impossible to divide the computation into more tasks than the map's total leaves.
		- 'maximum': divides the computation into `leavesTotal`-many tasks.
		- 'cpu': divides the computation into the number of available CPUs.
	CPUlimit : bool | float | int | None = None
		If relevant, whether and how to limit the number of processors `countFolds` will use.
		- `False`, `None`, or `0`: No limits on processor usage; uses all available processors. All other values will
		potentially limit processor usage.
		- `True`: Yes, limit the processor usage; limits to 1 processor.
		- `int >= 1`: The maximum number of available processors to use.
		- `0 < float < 1`: The maximum number of processors to use expressed as a fraction of available processors.
		- `-1 < float < 0`: The number of processors to *not* use expressed as a fraction of available processors.
		- `int <= -1`: The number of available processors to *not* use.
		- If the value of `CPUlimit` is a `float` greater than 1 or less than -1, `countFolds` truncates the value to an `int`
		with the same sign as the `float`.
	mapShape : tuple[int, ...] | None = None
		Tuple of integers representing the dimensions of the map to be folded. Mathematicians almost always use the term
		"dimensions", such as in the seminal paper, "Multi-dimensional map-folding". Nevertheless, in contemporary Python
		programming, in the context of these algorithms, the term "shape" makes it much easier to align the mathematics with the
		syntax of the programming language.
	flow : str | None = None
		My stupid way of selecting the version of the algorithm to use in the computation. There are certainly better ways to do
		this, but I have not yet solved this issue. As of 2025 Aug 14, these values will work:
		- 'daoOfMapFolding'
		- 'numba'
		- 'theorem2'
		- 'theorem2Numba'
		- 'theorem2Trimmed'

	Returns
	-------
	foldsTotal: Total number of distinct ways to fold a map of the given dimensions.

	Note well
	---------
	You probably do not want to divide your computation into tasks.

	If you want to compute a large `foldsTotal`, dividing the computation into tasks is usually a bad idea. Dividing the
	algorithm into tasks is inherently inefficient: efficient division into tasks means there would be no overlap in the
	work performed by each task. When dividing this algorithm, the amount of overlap is between 50% and 90% by all
	tasks: at least 50% of the work done by every task must be done by each task. If you improve the computation time,
	it will only change by -10 to -50% depending on (at the very least) the ratio of the map dimensions and the number
	of leaves. If an undivided computation would take 10 hours on your computer, for example, the computation will still
	take at least 5 hours but you might reduce the time to 9 hours. Most of the time, however, you will increase the
	computation time. If logicalCores >= `leavesTotal`, it will probably be faster. If logicalCores <= 2 * `leavesTotal`, it
	will almost certainly be slower for all map dimensions.
	"""
	# mapShape ---------------------------------------------------------------------

	if mapShape:
		pass
	elif listDimensions:
		mapShape = validateListDimensions(listDimensions)

	if mapShape is None:
		message = (f"""I received these values:
	`{listDimensions = }` and `{mapShape = }`,
	but I was unable to select a map for which to count the folds."""
		)
		raise ValueError(message)

	# task division instructions -----------------------------------------------------

	if computationDivisions:
		from mapFolding.beDRY import getLeavesTotal, getTaskDivisions, setProcessorLimit
		concurrencyLimit: int = setProcessorLimit(CPUlimit, packageSettings.concurrencyPackage)
		leavesTotal: int = getLeavesTotal(mapShape)
		taskDivisions = getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
		del leavesTotal
	else:
		concurrencyLimit = 1
		taskDivisions = 0

	# memorialization instructions ---------------------------------------------

	if pathLikeWriteFoldsTotal is not None:
		pathFilenameFoldsTotal = getPathFilenameFoldsTotal(mapShape, pathLikeWriteFoldsTotal)
		saveFoldsTotalFAILearly(pathFilenameFoldsTotal)
	else:
		pathFilenameFoldsTotal = None

	# Flow control until I can figure out a good way ---------------------------------

	if flow == 'daoOfMapFolding':
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.algorithms.daoOfMapFolding import doTheNeedful
		mapFoldingState = doTheNeedful(mapFoldingState)
		foldsTotal = mapFoldingState.foldsTotal

	elif flow == 'numba':
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.syntheticModules.daoOfMapFoldingNumba import doTheNeedful
		mapFoldingState = doTheNeedful(mapFoldingState)
		foldsTotal = mapFoldingState.foldsTotal

	elif flow == 'theorem2' and any(dimension > 2 for dimension in mapShape):
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
		mapFoldingState = transitionOnGroupsOfFolds(mapFoldingState)

		from mapFolding.syntheticModules.theorem2 import count
		mapFoldingState = count(mapFoldingState)

		foldsTotal = mapFoldingState.foldsTotal

	elif (flow == 'theorem2Numba' or taskDivisions == 0) and any(dimension > 2 for dimension in mapShape):
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
		mapFoldingState = transitionOnGroupsOfFolds(mapFoldingState)

		from mapFolding.syntheticModules.dataPacking import sequential
		mapFoldingState = sequential(mapFoldingState)

		foldsTotal = mapFoldingState.foldsTotal

	elif flow == 'theorem2Trimmed' and any(dimension > 2 for dimension in mapShape):
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
		mapFoldingState = transitionOnGroupsOfFolds(mapFoldingState)

		from mapFolding.syntheticModules.theorem2Trimmed import count
		mapFoldingState = count(mapFoldingState)

		foldsTotal = mapFoldingState.foldsTotal

	elif taskDivisions > 1:
		from mapFolding.dataBaskets import ParallelMapFoldingState
		mapFoldingParallelState: ParallelMapFoldingState = ParallelMapFoldingState(mapShape, taskDivisions=taskDivisions)

		from mapFolding.syntheticModules.countParallelNumba import doTheNeedful

		# `listStatesParallel` exists so you can research the parallel computation.
		foldsTotal, listStatesParallel = doTheNeedful(mapFoldingParallelState, concurrencyLimit) # pyright: ignore[reportUnusedVariable]  # noqa: RUF059

	else:
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.algorithms.daoOfMapFolding import doTheNeedful
		mapFoldingState = doTheNeedful(mapFoldingState)
		foldsTotal = mapFoldingState.foldsTotal

	# Follow memorialization instructions ---------------------------------------------

	if pathFilenameFoldsTotal is not None:
		saveFoldsTotal(pathFilenameFoldsTotal, foldsTotal)

	return foldsTotal

@cache
def A000682(n: int, flow: str | None = None) -> int:
	"""Compute A000682(n)."""
	match flow:
		case 'matrixNumPy':
			from mapFolding.algorithms.matrixMeandersNumPy import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersNumPyState as State
		case 'matrixPandas':
			from mapFolding.algorithms.matrixMeandersPandas import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersNumPyState as State
		case _:
			from mapFolding.algorithms.matrixMeanders import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersState as State

	oeisID = 'A000682'

	kOfMatrix: int = n - 1

	if n == 1:
		return 1
	elif n & 0b1:
		arcCode: int = 5
	else:
		arcCode = 1
	listCurveLocations: list[int] = [(arcCode << 1) | arcCode]

	MAXIMUMarcCode: int = 1 << (2 * kOfMatrix + 4)
	while listCurveLocations[-1] < MAXIMUMarcCode:
		arcCode = (arcCode << 4) | 0b101 # == arcCode * 2**4 + 5
		listCurveLocations.append((arcCode << 1) | arcCode)

	dictionaryMeanders=dict.fromkeys(listCurveLocations, 1)

	state = State(n, oeisID, kOfMatrix, dictionaryMeanders)

	return doTheNeedful(state) # pyright: ignore[reportArgumentType]

@cache
def A005316(n: int, flow: str | None = None) -> int:
	"""Compute A005316(n)."""
	match flow:
		case 'matrixNumPy':
			from mapFolding.algorithms.matrixMeandersNumPy import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersNumPyState as State
		case 'matrixPandas':
			from mapFolding.algorithms.matrixMeandersPandas import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersNumPyState as State
		case _:
			from mapFolding.algorithms.matrixMeanders import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersState as State

	oeisID = 'A005316'

	kOfMatrix: int = n - 1

	if n & 0b1:
		dictionaryMeanders: dict[int, int] = {15: 1}
	else:
		dictionaryMeanders = {22: 1}

	state = State(n, oeisID, kOfMatrix, dictionaryMeanders)

	return doTheNeedful(state) # pyright: ignore[reportArgumentType]

def NOTcountingFolds(oeisID: str, oeis_n: int, flow: str | None = None
		# , pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None  # noqa: ERA001
		, CPUlimit: bool | float | int | None = None  # noqa: FBT001
		) -> int:
	"""Do stuff."""
	countTotal: int = -1

	match oeisID:
		case 'A000136':
			from mapFolding.algorithms.oeisIDbyFormula import A000136
			countTotal = A000136(oeis_n)
		case 'A000560':
			from mapFolding.algorithms.oeisIDbyFormula import A000560
			countTotal = A000560(oeis_n)
		case 'A000682':
			countTotal = A000682(oeis_n, flow)
		case 'A001010':
			from mapFolding.algorithms.oeisIDbyFormula import A001010
			countTotal = A001010(oeis_n)
		case 'A001011':
			from mapFolding.algorithms.oeisIDbyFormula import A001011
			countTotal = A001011(oeis_n)
		case 'A005315':
			from mapFolding.algorithms.oeisIDbyFormula import A005315
			countTotal = A005315(oeis_n)
		case 'A005316':
			countTotal = A005316(oeis_n, flow)
		case 'A007822':
			mapShape: tuple[Literal[1], int] = (1, 2 * oeis_n)
			match flow:
				case 'asynchronous':
					from mapFolding.dataBaskets import MapFoldingState
					mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

					from mapFolding import setProcessorLimit
					concurrencyLimit = setProcessorLimit(CPUlimit)

					from mapFolding.syntheticModules.A007822.asynchronous import doTheNeedful
					mapFoldingState = doTheNeedful(mapFoldingState)

				case 'asynchronousNumba':
					from mapFolding.dataBaskets import MapFoldingState
					mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

					from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds
					mapFoldingState = transitionOnGroupsOfFolds(mapFoldingState)

					from mapFolding import setProcessorLimit
					concurrencyLimit = setProcessorLimit(CPUlimit)

					from mapFolding.syntheticModules.A007822.asynchronousAnnex import initializeConcurrencyManager
					initializeConcurrencyManager(maxWorkers=concurrencyLimit, groupsOfFolds=mapFoldingState.groupsOfFolds)
					mapFoldingState.groupsOfFolds = 0

					from mapFolding.syntheticModules.A007822.asynchronousNumba import count
					mapFoldingState = count(mapFoldingState)

				case 'asynchronousTrimmed':
					from mapFolding.dataBaskets import MapFoldingState
					mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

					from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds
					mapFoldingState = transitionOnGroupsOfFolds(mapFoldingState)

					from mapFolding import setProcessorLimit
					concurrencyLimit = setProcessorLimit(CPUlimit)

					from mapFolding.syntheticModules.A007822.asynchronousAnnex import initializeConcurrencyManager
					initializeConcurrencyManager(maxWorkers=concurrencyLimit, groupsOfFolds=mapFoldingState.groupsOfFolds)
					mapFoldingState.groupsOfFolds = 0

					from mapFolding.syntheticModules.A007822.asynchronousTrimmed import count
					mapFoldingState = count(mapFoldingState)

				case 'numba':
					from mapFolding.dataBaskets import MapFoldingState
					mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

					from mapFolding.syntheticModules.A007822.algorithmNumba import doTheNeedful
					mapFoldingState = doTheNeedful(mapFoldingState)

				case 'theorem2':
					from mapFolding.dataBaskets import MapFoldingState
					mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

					from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds
					mapFoldingState = transitionOnGroupsOfFolds(mapFoldingState)

					from mapFolding.syntheticModules.A007822.theorem2 import count
					mapFoldingState = count(mapFoldingState)

				case 'theorem2Numba':
					from mapFolding.dataBaskets import MapFoldingState
					mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

					from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds
					mapFoldingState = transitionOnGroupsOfFolds(mapFoldingState)

					from mapFolding.syntheticModules.A007822.theorem2Numba import count
					mapFoldingState = count(mapFoldingState)

				case 'theorem2Trimmed':
					from mapFolding.dataBaskets import MapFoldingState
					mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

					from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds
					mapFoldingState = transitionOnGroupsOfFolds(mapFoldingState)

					from mapFolding.syntheticModules.A007822.theorem2Trimmed import count
					mapFoldingState = count(mapFoldingState)

				case _:
					from mapFolding.dataBaskets import MapFoldingState
					mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

					from mapFolding.syntheticModules.A007822.algorithm import doTheNeedful
					mapFoldingState = doTheNeedful(mapFoldingState)

			countTotal = mapFoldingState.groupsOfFolds
		case 'A060206':
			from mapFolding.algorithms.oeisIDbyFormula import A060206
			countTotal = A060206(oeis_n)
		case 'A077460':
			from mapFolding.algorithms.oeisIDbyFormula import A077460
			countTotal = A077460(oeis_n)
		case 'A078591':
			from mapFolding.algorithms.oeisIDbyFormula import A078591
			countTotal = A078591(oeis_n)
		case 'A178961':
			from mapFolding.algorithms.oeisIDbyFormula import A178961
			countTotal = A178961(oeis_n)
		case 'A223094':
			from mapFolding.algorithms.oeisIDbyFormula import A223094
			countTotal = A223094(oeis_n)
		case 'A259702':
			from mapFolding.algorithms.oeisIDbyFormula import A259702
			countTotal = A259702(oeis_n)
		case 'A301620':
			from mapFolding.algorithms.oeisIDbyFormula import A301620
			countTotal = A301620(oeis_n)

		case _:
			pass

	return countTotal

