# ruff: noqa: E701
from mapFolding import (
	defineProcessorLimit, getPathFilenameFoldsTotal, packageSettings, saveFoldsTotal, saveFoldsTotalFAILearly)
from mapFolding._e import mapShapeIs2上nDimensions
from mapFolding._e.dataBaskets import EliminationState
from os import PathLike
from pathlib import Path, PurePath

def eliminateFolds(mapShape: tuple[int, ...] | None = None
				, state: EliminationState | None = None
				, pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
				# , * # TODO improve `standardizedEqualToCallableReturn` so it will work with keyword arguments
				, CPUlimit: bool | float | int | None = None  # noqa: FBT001
				, flow: str | None = None
				) -> int:
	"""
	Compute foldsTotal by elimination.

	Parameters
	----------
	mapShape : tuple[int, ...] | None = None
		Tuple of integers representing the dimensions of the map to be folded. Mathematicians almost always use the term
		"dimensions", such as in the seminal paper, "Multi-dimensional map-folding". Nevertheless, in contemporary Python
		programming, in the context of these algorithms, the term "shape" makes it much easier to align the mathematics with the
		syntax of the programming language.
	pathLikeWriteFoldsTotal : PathLike[str] | PurePath | None = None
		A filename, a path of only directories, or a path with directories and a filename to which `countFolds` will write the
		value of `foldsTotal`. If `pathLikeWriteFoldsTotal` is a path of only directories, `countFolds` creates a filename based
		on the map dimensions.
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
	flow : str | None = None
		My stupid way of selecting the version of the algorithm to use in the computation.

	Returns
	-------
	foldsTotal : int
		Number of distinct ways to fold a map of the given dimensions.
	"""
#-------- state ---------------------------------------------------------------------
	if not state:
		if not mapShape:
			message = (f"""I received these values:
	`{mapShape = }` and `{state = }`,
	but I was unable to select a map of which to count the folds."""
			)
			raise ValueError(message)
		state = EliminationState(mapShape)

#-------- concurrency limit -----------------------------------------------------

	concurrencyLimit: int = defineProcessorLimit(CPUlimit, packageSettings.concurrencyPackage)

#-------- memorialization instructions ---------------------------------------------

	if pathLikeWriteFoldsTotal is not None:
		pathFilenameFoldsTotal: Path | None = getPathFilenameFoldsTotal(state.mapShape, pathLikeWriteFoldsTotal)
		saveFoldsTotalFAILearly(pathFilenameFoldsTotal)
	else:
		pathFilenameFoldsTotal = None

#-------- Algorithm version -----------------------------------------------------
	match flow:
		case 'constraintPropagation': from mapFolding._e.algorithms.constraintPropagation import doTheNeedful  # noqa: PLC0415
		case 'crease':
			if mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=4):
				from mapFolding._e.algorithms.eliminationCrease import doTheNeedful  # noqa: PLC0415
			else:
				message: str = "As of 25 December 2025, this algorithm only works on mapShape = (2,) * n, n >= 4. Did I forget to update this barrier?"
				raise NotImplementedError(message)
		case 'elimination' | _: from mapFolding._e.algorithms.elimination import doTheNeedful  # noqa: PLC0415

	state = doTheNeedful(state, concurrencyLimit)

#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameFoldsTotal is not None:
		saveFoldsTotal(pathFilenameFoldsTotal, state.foldsTotal)

	return state.foldsTotal
