"""Count map foldings with the selected elimination algorithm.

(AI generated docstring)

You can run elimination computations and optionally save the resulting count.

Contents
--------
Functions
	eliminateFolds
		Count distinct map foldings by eliminating invalid leaf arrangements.
"""

#=Sin= Import `doTheNeedful` from the selected algorithm: a flow control technique.
# ruff: file-ignore[import-outside-top-level]
from __future__ import annotations

from mapFolding._e.dataBaskets import StateElimination
from mapFolding.beDRY import defineProcessorLimit, mapShapeIs2上nDimensions
from mapFolding.kitFilesystem import makePathFilenameFolds, saveTotal, saveTotalFAILearly
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit
	from os import PathLike
	from pathlib import Path

def eliminateFolds(
	mapShape: tuple[int, ...] | None = None
	, state: StateElimination | None = None
	, pathLikeWrite: PathLike[str] | None = None
	, *
	, CPUlimit: ConcurrencyLimit = None
	, flow: str | None = None
	, suffix: str = '.totalFolds'
) -> int:
	"""Count distinct map foldings by eliminating invalid leaf arrangements.

	You can run the selected algorithm for `mapShape` or the supplied `state`. When you supply `state`,
	you use its shape and existing search space. You can also save the count to `pathLikeWrite`.

	Parameters
	----------
	mapShape : tuple[int, ...] | None = None
		Nonempty tuple of nonnegative dimension lengths. Required when `state` is absent.
	state : StateElimination | None = None
		Existing elimination state [1]. The selected algorithm can mutate `state`.
	pathLikeWrite : PathLike[str] | None = None
		Filename or destination directory for the count. `None` disables writing. A directory receives a
		filename derived from the map shape [2].
	CPUlimit : ConcurrencyLimit = None
		Processor limit interpreted by `mapFolding.beDRY.defineProcessorLimit` [3]. `None`, `False`, and `0`
		use available processors; `True` selects one. Positive integers cap usage, negative integers reserve
		processors, and fractions specify the proportion to use or reserve.
	flow : str | None = None
		Algorithm selector. `None`, `'elimination'`, and unrecognized strings select elimination [4].
		`'constraintPropagation'` selects the constraint solver [5]. `'crease'` selects the specialized
		crease algorithm [6].
	suffix : str = '.totalFolds'
		Suffix used when creating the output filename.

	Returns
	-------
	totalFolds : int
		Count after applying the selected algorithm's symmetry multipliers. A shape containing zero returns
		`1`.

	Raises
	------
	ValueError
		If neither `state` nor a nonempty `mapShape` is supplied, or shape validation fails.
	NotImplementedError
		If `flow` is `'crease'` and the nonzero shape is not `(2,) * n` for `n >= 4`.

	Examples
	--------
	The computation script passes both input and output options.

		```python
		computed = eliminateFolds(
			mapShape=mapShape, state=state, pathLikeWrite=pathLikeWrite, CPUlimit=CPUlimit, flow=flow
		)
		```

	References
	----------
	[1] `mapFolding._e.dataBaskets.StateElimination`

	[2] `mapFolding.kitFilesystem.makePathFilenameFolds`

	[3] `mapFolding.beDRY.defineProcessorLimit`

	[4] `mapFolding._e.algorithms.elimination.doTheNeedful`

	[5] `mapFolding._e.algorithms.constraintPropagation.doTheNeedful`

	[6] `mapFolding._e.algorithms.eliminationCrease.doTheNeedful`
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
