"""
Public API for the map folding algorithm with simplified interface.

This module provides the main entry point for users of the mapFolding package,
abstracting away the complexities of the computational algorithm. It offers
a high-level interface to count the total number of possible ways to fold
a rectangular map of specified dimensions, with options for customizing the
computation process and saving results.

The primary function is countFolds, which handles parameter validation,
computation state management, dispatching to the appropriate algorithm
implementation, and optional persistence of results.
"""

from collections.abc import Sequence
from mapFolding import (
	ComputationState,
	getPathFilenameFoldsTotal,
	outfitCountFolds,
	saveFoldsTotal,
	saveFoldsTotalFAILearly,
	setProcessorLimit,
	The,
	validateListDimensions,
)
from os import PathLike
from pathlib import PurePath

def countFolds(listDimensions: Sequence[int]
				, pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
				, computationDivisions: int | str | None = None
				, CPUlimit: int | float | bool | None = None
				) -> int:
	"""
	Count the total number of possible foldings for a given map dimensions.

	This function serves as the main public interface to the map folding algorithm,
	handling all parameter validation, computation state management, and result
	persistence in a user-friendly way.

	Parameters
	----------
	listDimensions: List of integers representing the dimensions of the map to be folded.
	pathLikeWriteFoldsTotal (None): Path, filename, or pathFilename to write the total fold count to.
		If a directory is provided, creates a file with a default name based on map dimensions.
	computationDivisions (None):
		Whether and how to divide the computational work. See notes for details.
	CPUlimit (None): This is only relevant if there are `computationDivisions`: whether and how to limit the CPU usage. See notes for details.
	Returns
	-------
	foldsTotal: Total number of distinct ways to fold a map of the given dimensions.

	Notes
	-----
	Computation divisions:
		- None: no division of the computation into tasks; sets task divisions to 0
		- int: direct set the number of task divisions; cannot exceed the map's total leaves
		- "maximum": divides into `leavesTotal`-many `taskDivisions`
		- "cpu": limits the divisions to the number of available CPUs, i.e. `concurrencyLimit`

	Limits on CPU usage `CPUlimit`:
		- `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
		- `True`: Yes, limit the CPU usage; limits to 1 CPU.
		- Integer `>= 1`: Limits usage to the specified number of CPUs.
		- Decimal value (`float`) between 0 and 1: Fraction of total CPUs to use.
		- Decimal value (`float`) between -1 and 0: Fraction of CPUs to *not* use.
		- Integer `<= -1`: Subtract the absolute value from total CPUs.

	N.B.: You probably don't want to divide the computation into tasks.

	If you want to compute a large `foldsTotal`, dividing the computation into tasks is usually a bad idea. Dividing the algorithm into tasks is inherently inefficient: efficient division into tasks means there would be no overlap in the work performed by each task. When dividing this algorithm, the amount of overlap is between 50% and 90% by all tasks: at least 50% of the work done by every task must be done by _all_ tasks. If you improve the computation time, it will only change by -10 to -50% depending on (at the very least) the ratio of the map dimensions and the number of leaves. If an undivided computation would take 10 hours on your computer, for example, the computation will still take at least 5 hours but you might reduce the time to 9 hours. Most of the time, however, you will increase the computation time. If logicalCores >= leavesTotal, it will probably be faster. If logicalCores <= 2 * leavesTotal, it will almost certainly be slower for all map dimensions.
	"""
	mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
	concurrencyLimit: int = setProcessorLimit(CPUlimit, The.concurrencyPackage)
	computationStateInitialized: ComputationState = outfitCountFolds(mapShape, computationDivisions, concurrencyLimit)

	if pathLikeWriteFoldsTotal is not None:
		pathFilenameFoldsTotal = getPathFilenameFoldsTotal(computationStateInitialized.mapShape, pathLikeWriteFoldsTotal)
		saveFoldsTotalFAILearly(pathFilenameFoldsTotal)
	else:
		pathFilenameFoldsTotal = None

	computationStateComplete: ComputationState = The.dispatcher(computationStateInitialized)

	computationStateComplete.getFoldsTotal()

	if pathFilenameFoldsTotal is not None:
		saveFoldsTotal(pathFilenameFoldsTotal, computationStateComplete.foldsTotal)

	return computationStateComplete.foldsTotal
