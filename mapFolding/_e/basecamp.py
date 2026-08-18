#=Sin= Import `doTheNeedful` from the selected algorithm: a flow control technique.
# ruff: file-ignore[import-outside-top-level]
from __future__ import annotations

from mapFolding._e.dataBaskets import StateElimination
from mapFolding.beDRY import defineProcessorLimit, mapShapeIs2上nDimensions
from mapFolding.kitFilesystem import makePathFilenameFolds, saveTotal, saveTotalFAILearly
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike
	from pathlib import Path

def eliminateFolds(
	mapShape: tuple[int, ...] | None = None
	, state: StateElimination | None = None
	, pathLikeWrite: PathLike[str] | None = None
	, *
	, CPUlimit: Limitation = None
	, flow: str | None = None
	, suffix: str = '.totalFolds'
) -> int:
	"""
	Compute totalFolds by elimination.

	Parameters
	----------
	mapShape : tuple[int, ...] | None = None
		Tuple of integers representing the dimensions of the map to be folded. Mathematicians almost always use the term
		"dimensions", such as in the seminal paper, "Multi-dimensional map-folding". Nevertheless, in contemporary Python
		programming, in the context of these algorithms, the term "shape" makes it much easier to align the mathematics with the
		syntax of the programming language.
	pathLikeWrite : PathLike[str] | None = None
		A filename, a path of only directories, or a path with directories and a filename to which `countFolds` will write the
		value of `totalFolds`. If `pathLikeWrite` is a path of only directories, `countFolds` creates a filename based
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
	suffix : str = ".totalFolds"
		The filename suffix for the saved count.

	Returns
	-------
	totalFolds : int
		Number of distinct ways to fold a map of the given dimensions.

	Raises
	------
	ValueError
		If `mapShape` is `None` and `state` is `None`, then `eliminateFolds` raises a `ValueError`
		because it cannot determine the map shape to compute the number of folds for.
	NotImplementedError
		If `flow` is set to "crease" and `mapShape` is not of the form `(2,) * n` for `n >= 4`, then
		`eliminateFolds` raises a `NotImplementedError` because the crease algorithm is only
		implemented for maps of that
	"""
	if not state:
		if not mapShape:
			message: str = f'I received `{mapShape = }` and `{state = }`, and I was unable to select a `mapShape`.'
			raise ValueError(message)
		state = StateElimination(mapShape)

	concurrencyLimit: int = defineProcessorLimit(CPUlimit, settingsPackage.concurrencyPackage)

	#-------- Memorialization instructions ---------------------------------------------

	if pathLikeWrite is None:
		pathFilenameTotalFolds: Path | None = None
	else:
		pathFilenameTotalFolds = makePathFilenameFolds(state.mapShape, pathLikeWrite, suffix=suffix)
		saveTotalFAILearly(pathFilenameTotalFolds)

	#-------- Algorithm version -----------------------------------------------------

	if 0 in state.mapShape:
		totalFolds: int = 1
	else:
		match flow:
			case 'constraintPropagation':
				from mapFolding._e.algorithms.constraintPropagation import doTheNeedful
			case 'crease':
				if mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToRideThis=4):
					from mapFolding._e.algorithms.eliminationCrease import doTheNeedful
				else:
					message: str = f'`{flow = }` is restricted to `mapShape` = (2,) * n, 4 <= n. Did I forget to update this check?'
					raise NotImplementedError(message)
			case 'elimination' | _:
				from mapFolding._e.algorithms.elimination import doTheNeedful

		totalFolds = doTheNeedful(state, concurrencyLimit).totalFolds

	#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameTotalFolds is not None:
		saveTotal(pathFilenameTotalFolds, totalFolds)

	return totalFolds
