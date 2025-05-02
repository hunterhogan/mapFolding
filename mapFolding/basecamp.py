"""
Public API for the map folding algorithm with simplified interface.

This module provides the main entry point for users of the mapFolding package, abstracting away the complexities of the
computational algorithm. It offers a high-level interface to count the total number of possible ways to fold a
rectangular map of specified dimensions, with options for customizing the computation process and saving results.

The primary function is countFolds, which handles parameter validation, computation state management, dispatching to the
appropriate algorithm implementation, and optional persistence of results.
"""

from collections.abc import Sequence
from mapFolding import (
	getPathFilenameFoldsTotal,
	saveFoldsTotal,
	saveFoldsTotalFAILearly,
	setProcessorLimit,
	The,
	validateListDimensions,
)
from os import PathLike
from pathlib import PurePath

def countFolds(listDimensions: Sequence[int] | None = None
				, pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
				, computationDivisions: int | str | None = None
				, CPUlimit: int | float | bool | None = None
				# , * I need to improve `standardizedEqualToCallableReturn` so it will work with keyword arguments
				, mapShape: tuple[int, ...] | None = None
				, oeisID: str | None = None
				, oeis_n: int | None = None
				, flow: str | None = None
				) -> int:
	"""
	Count the total number of possible foldings for a given map dimensions.

	This function serves as the main public interface to the map folding algorithm, handling all parameter validation,
	computation state management, and result persistence in a user-friendly way.

	Parameters
	----------
	listDimensions
		List of integers representing the dimensions of the map to be folded.
	pathLikeWriteFoldsTotal: None
		Path, filename, or pathFilename to write the total fold count to. If a directory is provided, creates a file
		with a default name based on map dimensions.
	computationDivisions: None
		Whether and how to divide the computational work.
		- `None`: no division of the computation into tasks; sets task divisions to 0.
		- int: directly set the number of task divisions; cannot exceed the map's total leaves.
		- `'maximum'`: divides into `leavesTotal`-many `taskDivisions`.
		- `'cpu'`: limits the divisions to the number of available CPUs: i.e., `concurrencyLimit`.
	CPUlimit: None
		This is only relevant if there are `computationDivisions`: whether and how to limit the CPU usage.
		- `False`, `None`, or `0`: No limits on processor usage; uses all available processors. All other values will
		potentially limit processor usage.
		- `True`: Yes, limit the processor usage; limits to 1 processor.
		- Integer `>= 1`: Limits usage to the specified number of processors.
		- Decimal value (`float`) between 0 and 1: Fraction of total processors to use.
		- Decimal value (`float`) between -1 and 0: Fraction of processors to _not_ use.
		- Integer `<= -1`: Subtract the absolute value from total processors.

	Returns
	-------
	foldsTotal: Total number of distinct ways to fold a map of the given dimensions.

	Note well
	---------
	You probably do not want to divide your computation into tasks.

	If you want to compute a large `foldsTotal`, dividing the computation into tasks is usually a bad idea. Dividing the
	algorithm into tasks is inherently inefficient: efficient division into tasks means there would be no overlap in the
	work performed by each task. When dividing this algorithm, the amount of overlap is between 50% and 90% by all
	tasks: at least 50% of the work done by every task must be done by _all_ tasks. If you improve the computation time,
	it will only change by -10 to -50% depending on (at the very least) the ratio of the map dimensions and the number
	of leaves. If an undivided computation would take 10 hours on your computer, for example, the computation will still
	take at least 5 hours but you might reduce the time to 9 hours. Most of the time, however, you will increase the
	computation time. If logicalCores >= `leavesTotal`, it will probably be faster. If logicalCores <= 2 * `leavesTotal`, it
	will almost certainly be slower for all map dimensions.
	"""
	# mapShape =====================================================================

	if mapShape:
		pass
	else:
		if oeisID and oeis_n:
			from mapFolding.oeis import settingsOEIS
			try:
				mapShape = settingsOEIS[oeisID]['getMapShape'](oeis_n)
			except KeyError:
				pass
		if not mapShape and listDimensions:
			mapShape = validateListDimensions(listDimensions)

	if mapShape is None:
		raise ValueError(f"""I received these values:
	`{listDimensions = }`,
	`{mapShape = }`,
	`{oeisID = }` and `{oeis_n = }`,
	but I was unable to select a map for which to count the folds.""")

	# task division instructions ===============================================

	if computationDivisions:
		# NOTE `The.concurrencyPackage`
		concurrencyLimit: int = setProcessorLimit(CPUlimit, The.concurrencyPackage)
		from mapFolding.beDRY import getLeavesTotal, getTaskDivisions
		leavesTotal: int = getLeavesTotal(mapShape)
		taskDivisions = getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
		del leavesTotal
	else:
		concurrencyLimit = 1
		taskDivisions = 0

	# memorialization instructions ===========================================

	if pathLikeWriteFoldsTotal is not None:
		pathFilenameFoldsTotal = getPathFilenameFoldsTotal(mapShape, pathLikeWriteFoldsTotal)
		saveFoldsTotalFAILearly(pathFilenameFoldsTotal)
	else:
		pathFilenameFoldsTotal = None

	# Flow control until I can figure out a good way ===============================

	if flow == 'daoOfMapFolding':
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.daoOfMapFolding import doTheNeedful
		mapFoldingState = doTheNeedful(mapFoldingState)
		foldsTotal = mapFoldingState.foldsTotal

	elif flow == 'theorem2' and any((dimension > 2 for dimension in mapShape)):
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.syntheticModules.initializeCount import initializeGroupsOfFolds
		mapFoldingState = initializeGroupsOfFolds(mapFoldingState)

		from mapFolding.syntheticModules.theorem2 import count
		mapFoldingState = count(mapFoldingState)

		foldsTotal = mapFoldingState.foldsTotal

	elif flow == 'theorem2Trimmed' and any((dimension > 2 for dimension in mapShape)):
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.syntheticModules.initializeCount import initializeGroupsOfFolds
		mapFoldingState = initializeGroupsOfFolds(mapFoldingState)

		from mapFolding.syntheticModules.theorem2Trimmed import count
		mapFoldingState = count(mapFoldingState)

		foldsTotal = mapFoldingState.foldsTotal

	elif (flow == 'theorem2Numba' or taskDivisions == 0) and any((dimension > 2 for dimension in mapShape)):
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.syntheticModules.initializeCount import initializeGroupsOfFolds
		mapFoldingState = initializeGroupsOfFolds(mapFoldingState)

		from mapFolding.syntheticModules.dataPacking import sequential
		mapFoldingState = sequential(mapFoldingState)

		foldsTotal = mapFoldingState.foldsTotal

	elif taskDivisions > 1:
		from mapFolding.dataBaskets import ParallelMapFoldingState
		parallelMapFoldingState: ParallelMapFoldingState = ParallelMapFoldingState(mapShape, taskDivisions=taskDivisions)

		from mapFolding.syntheticModules.countParallel import doTheNeedful
		foldsTotal, listStatesParallel = doTheNeedful(parallelMapFoldingState, concurrencyLimit)

	else:
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.syntheticModules.daoOfMapFolding import doTheNeedful
		mapFoldingState = doTheNeedful(mapFoldingState)
		foldsTotal = mapFoldingState.foldsTotal

	# Follow memorialization instructions ===========================================

	if pathFilenameFoldsTotal is not None:
		saveFoldsTotal(pathFilenameFoldsTotal, foldsTotal)

	return foldsTotal
