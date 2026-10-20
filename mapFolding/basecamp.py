#=Sin= Import `doTheNeedful` from the selected algorithm: a flow control technique.
#ruff: file-ignore[import-outside-top-level]
"""Select and run map-folding, stamp-folding, and meander algorithms.

(AI generated docstring)

You can use this module as the entry point for counting map foldings, rotationally symmetric
foldings, and meander sequence terms, or enumerate stamp-folding and meander permutations. Each
entry point selects an algorithm, prepares computation state, and optionally saves a result.

Contents
--------
Functions
	countFolds
		Count distinct foldings of a map with the supplied dimensions.
	countFoldsSymmetric
		Count foldings constrained by rotational symmetry.
	countMeanders
		Compute a meander term and return the completed computation state.
	enumerateSequences
		Generate stamp-folding or meander permutations, or return a bilateral count.
"""
from __future__ import annotations

from mapFolding.beDRY import defineProcessorLimit, getTaskDivisions, getTotalLeaves, mapShapeIs2上nDimensions, validateMapShape
from mapFolding.dataBaskets import StateMapFolding, StateMapFoldingParallel, StateMapFoldingSymmetric, StateMeanders
from mapFolding.kitFilesystem import makePathFilenameCount, makePathFilenameFolds, saveTotal, saveTotalFAILearly, writeAlbum
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import Folding
	from os import PathLike
	from pathlib import Path
	from typing import Literal, LiteralString

def countFolds(mapShape: Sequence[int]
				, flow: Literal['daoOfMapFolding', 'daoOfMapFoldingNumba', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'] | LiteralString | None = None
				, pathLikeWrite: PathLike[str] | None = None
				, *
				, CPUlimit: ConcurrencyLimit = None
				, computationDivisions: int | str | None = None
				, suffix: str = ".totalFolds"
				) -> int:
	"""Count distinct foldings of the map described by `mapShape`.

	You can use this function to count ways of folding a map or stamp strip. This function normalizes
	the dimension lengths, selects a counting algorithm, and returns the count. Providing
	`pathLikeWrite` also requests an output file.

	Task Division
	-------------
	A task count greater than one selects the generated parallel implementation regardless of
	`flow`. Tasks can repeat preparatory work, so more tasks do not guarantee a faster computation.
	Truthy task requests calculate the leaf count before the zero-dimension special case.

	Output Files
	------------
	`saveTotalFAILearly` [1] writes and reads a test value before counting. The test file remains
	in place until the count is saved. `saveTotal` [2] attempts fallback output if the final write
	fails.

	Parameters
	----------
	mapShape : Sequence[int]
		A sequence containing the positive integer length of each dimension of the map to be folded.
	pathLikeWrite : PathLike[str] | None = None
		A filename, a path of only directories, or a path with directories and a filename to which
		`countFolds` will write the total. If `pathLikeWrite` is a path of only directories,
		`countFolds` creates a filename based on the map dimensions.
	computationDivisions : int | str | None = None
		Whether and how to divide the computational work.
		- `None`: no division of the computation into tasks.
		- `int`: into how many tasks `countFolds` will divide the computation. The values 0 or 1 are
		identical to `None`. It is mathematically impossible to divide the computation into more tasks
		than the map's total leaves.
		- 'maximum': divides the computation into `totalLeaves`-many tasks.
		- 'cpu': divides the computation into the number of available CPUs.
	CPUlimit : bool | float | int | None = None
		If relevant, whether and how to limit the number of processors `countFolds` will use.
		- `False`, `None`, or `0`: No limits on processor usage; uses all available processors. All
		other values will potentially limit processor usage.
		- `True`: Yes, limit the processor usage; limits to 1 processor.
		- `int >= 1`: The maximum number of available processors to use.
		- `0 < float < 1`: The maximum number of processors to use expressed as a fraction of
		available processors.
		- `-1 < float < 0`: The number of processors to *not* use expressed as a fraction of available
		processors.
		- `int <= -1`: The number of available processors to *not* use.
		- If the value of `CPUlimit` is a `float` greater than 1 or less than -1, `countFolds`
		truncates the value to an `int` with the same sign as the `float`.
	flow : Literal['daoOfMapFolding', 'numba', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'] | LiteralString | None = None
		My stupid way of selecting the version of the algorithm to use in the computation. There are
		certainly better ways to do this, but I have not yet solved this issue.

	Returns
	-------
	totalFolds : int
		Number of distinct ways to fold a map of the given dimensions.

	Using `computationDivisions`
	----------------------------
	You probably do not want to divide your computation into tasks.

	Dividing the algorithm into tasks is inherently inefficient: between 50% and 90% of the work done
	by every task must be done by each task. In some cases, you will increase the total computation
	time, but if you improve the computation time, it will only change by -10 to -50% depending on (at
	the very least) the ratio of the map dimensions, the number of leaves, and the ratio of
	`totalLeaves` to logicalCores.

	If an undivided computation would take 10 hours on your computer, for example, the computation
	will take at least 5 hours, you might reduce the time to 9 hours, but most of the time, you will
	increase the computation time. If logicalCores >= `totalLeaves`, the computation will probably be
	faster. If logicalCores <= 2 * `totalLeaves`, it will almost certainly be slower.
	"""
	mapShape = validateMapShape(mapShape)

#-------- task division instructions -----------------------------------------------------

	if computationDivisions:
		concurrencyLimit: int = defineProcessorLimit(CPUlimit, settingsPackage.concurrencyPackage)
		taskDivisions: int = getTaskDivisions(computationDivisions, concurrencyLimit, getTotalLeaves(mapShape))
	else:
		concurrencyLimit = 1
		taskDivisions = 0

#-------- memorialization instructions ---------------------------------------------

	if pathLikeWrite is None:
		pathFilenameTotalFolds: Path | None = None
	else:
		pathFilenameTotalFolds = saveTotalFAILearly(makePathFilenameFolds(mapShape, pathLikeWrite, suffix=suffix))

#-------- Algorithm version -----------------------------------------------------

	if 0 in mapShape:
		""" I don't think 0 is a valid number of panels in a dimension. I think some offset values on OEIS are wrong.
		| oeisID  | offset | mapShape                      | Equivalencies to offset 1            |
		| ------- | ------ | ----------------------------- | ------------------------------------ |
		| A000136 | 1      | (1,) or (1, 1)                |                                      |
		| A001415 | 0      | (2, 0)                        | a(1) = A000136(2), a(2) = A001418(2) |
		| A001416 | 0      | (3, 0)                        | a(1) = A000136(3), a(3) = A001418(3) |
		| A001417 | 0      | Let p ∈ {2}. ∴ (p,) * 0 or () | a(1) = A000136(2), a(2) = A001418(2) |
		| A001418 | 1      | (1, 1)                        |                                      |
		| A195646 | 0      | Let p ∈ {3}. ∴ (p,) * 0 or () | a(1) = A000136(3), a(2) = A001418(3) |
		"""
		totalFolds = 1
	elif 1 < taskDivisions:
		from mapFolding.synthesized.countParallelNumba import doTheNeedful

		mapFoldingParallelState: StateMapFoldingParallel = StateMapFoldingParallel(mapShape, taskDivisions=taskDivisions)

		# IMPROVEMENT Figure out a better place to document this. `boxOfStatesParallel` exists so you can research the parallel computation.
		totalFolds, _boxOfStatesParallel = doTheNeedful(mapFoldingParallelState, concurrencyLimit)

	else:
		if flow == 'daoOfMapFolding':
			from mapFolding.algorithms.daoOfMapFolding import doTheNeedful
		elif flow == 'daoOfMapFoldingNumba':
			from mapFolding.synthesized.daoOfMapFoldingNumba import doTheNeedful
		elif flow == 'numba':
			from mapFolding.synthesized.inlineNumba import doTheNeedful
		elif any(map((2).__lt__, mapShape)) or mapShapeIs2上nDimensions(mapShape, youMustBeDimensionsTallToRideThis=2):
			match flow:
				case 'theorem2Numba':
					from mapFolding.synthesized.theorem2Numba import doTheNeedful
				case 'theorem2Trimmed':
					from mapFolding.synthesized.theorem2Trimmed import doTheNeedful
				case 'theorem2' | _:
					from mapFolding.synthesized.theorem2 import doTheNeedful
		else:
			from mapFolding.algorithms.daoOfMapFolding import doTheNeedful

		mapFoldingState: StateMapFolding = StateMapFolding(mapShape)
		mapFoldingState = doTheNeedful(mapFoldingState)
		totalFolds = mapFoldingState.totalFolds

#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameTotalFolds is not None:
		saveTotal(pathFilenameTotalFolds, totalFolds)

	return totalFolds

def countFoldsSymmetric(mapShape: tuple[int, ...], flow: LiteralString | Literal['algorithm', 'algorithmNumba', 'asynchronous', 'theorem2', 'theorem2Numba', 'theorem2Trimmed', ''] = '', pathLikeWrite: PathLike[str] | None = None, *, CPUlimit: ConcurrencyLimit = None, suffix: str = ".totalFolds") -> int:
	"""Count foldings constrained by rotational symmetry.

	(AI generated docstring)

	You can use this function to count the symmetric foldings described by OEIS A007822 [1].
	For sequence index `n`, supply strip dimensions `(1, 2 * n)`. The algorithm uses this even strip
	shape although the sequence describes symmetric foldings of an odd-length strip.

	Output Files
	------------
	`mapFolding.kitFilesystem.saveTotalFAILearly` [2] leaves a test file before counting.
	`mapFolding.kitFilesystem.saveTotal` [3] replaces the test value with the completed count and
	attempts fallback output if the final write fails.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Strip dimensions for the symmetric-folding calculation.
	flow : LiteralString | Literal['algorithm', 'algorithmNumba', 'asynchronous', 'theorem2', 'theorem2Numba', 'theorem2Trimmed', ''] = ''
		Algorithm preference. `'asynchronous'` enables concurrent execution. Each named variant
		selects the corresponding generated module. `'algorithm'`, the empty string, and
		unrecognized values select the standard algorithm.
	pathLikeWrite : PathLike[str] | None = None
		Optional output file or existing directory. `None` disables saving.
	CPUlimit : ConcurrencyLimit = None
		Processor preference interpreted by `mapFolding.beDRY.defineProcessorLimit` [4] only for
		`'asynchronous'`.
	suffix : str = ".totalFolds"
		Filename suffix when saving is enabled.

	Returns
	-------
	symmetricFolds : int
		Number of foldings satisfying the rotational-symmetry constraint.

	Examples
	--------
	The runner in `easyRun/countFoldsSymmetric.py` supplies algorithm and processor preferences.

		```python
		totalFolds = countFoldsSymmetric(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit)
		```

	References
	----------
	[1] OEIS A007822, number of symmetric foldings of an odd-length stamp strip.
		https://oeis.org/A007822
	[2] `mapFolding.kitFilesystem.saveTotalFAILearly`

	[3] `mapFolding.kitFilesystem.saveTotal`

	[4] `mapFolding.beDRY.defineProcessorLimit`

	"""
	mapShape = validateMapShape(mapShape)

#-------- memorialization instructions ---------------------------------------------

	if pathLikeWrite is None:
		pathFilenameTotalFolds: Path | None = None
	else:
		pathFilenameTotalFolds = saveTotalFAILearly(makePathFilenameFolds(mapShape, pathLikeWrite, suffix=suffix))

#-------- Algorithm version -----------------------------------------------------

	if flow == 'asynchronous':
		from mapFolding.synthesized.mapFoldingSymmetric.asynchronous import doTheNeedful
		totalFolds = doTheNeedful(StateMapFoldingSymmetric(mapShape), defineProcessorLimit(CPUlimit)).symmetricFolds
	else:
		match flow:
			case 'algorithmNumba':
				from mapFolding.synthesized.mapFoldingSymmetric.algorithmNumba import doTheNeedful
			case 'theorem2':
				from mapFolding.synthesized.mapFoldingSymmetric.theorem2 import doTheNeedful
			case 'theorem2Numba':
				from mapFolding.synthesized.mapFoldingSymmetric.theorem2Numba import doTheNeedful
			case 'theorem2Trimmed':
				from mapFolding.synthesized.mapFoldingSymmetric.theorem2Trimmed import doTheNeedful
			case 'algorithm' | _:
				from mapFolding.synthesized.mapFoldingSymmetric.algorithm import doTheNeedful

		symmetricState: StateMapFoldingSymmetric = StateMapFoldingSymmetric(mapShape)

		symmetricState = doTheNeedful(symmetricState)
		totalFolds = symmetricState.symmetricFolds

#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameTotalFolds is not None:
		saveTotal(pathFilenameTotalFolds, totalFolds)

	return totalFolds

#=Sin= `CPUlimit` is in the signature due to `KeywordArgumentsCount`.
#ruff: ignore[unused-function-argument]
def countMeanders(
	kind: Literal['closed', 'meanders', 'semi'] | LiteralString | None = None
	, n: int | None = None
	, flow: Literal['matrixMeanders', 'matrixNumPy', 'matrixPandas', 'prune', 'pruneNumPy', 'prunePandas'] | LiteralString | None = None
	, pathLikeWrite: PathLike[str] | None = None
	, *
	, state: StateMeanders | None = None
	, CPUlimit: ConcurrencyLimit = None
	, suffix: str = ".countTotal"
) -> StateMeanders:
	"""Compute a meander sequence term for `kind` and `n`, or continue `state`.

	(AI generated docstring)

	You can use this function to count meanders or semi-meanders and inspect the completed computation
	state. An existing `state` takes precedence over `kind` and `n`. This function returns
	`mapFolding.dataBaskets.StateMeanders` [1]; read the completed count from `state.totalMeanders`.

	State and Output
	----------------
	This function constructs `mapFolding.dataBaskets.StateMeanders` [1] when `state` is absent.
	The semi-meander case at index one returns the initialized state without running a counting
	algorithm. Providing a save path performs a test write with
	`mapFolding.kitFilesystem.saveTotalFAILearly` [2] before counting and writes the completed total
	through `mapFolding.kitFilesystem.saveTotal` [3].

	Parameters
	----------
	kind : Literal['closed', 'meanders', 'semi'] | LiteralString | None = None
		Meander family when `state` is absent. `'closed'` constructs a meander state at the
		transformed index `2 * n - 1`; other values are forwarded as the state family.
	n : int | None = None
		Sequence index when `state` is absent. This function does not validate the index range.
	flow : Literal['matrixMeanders', 'matrixNumPy', 'matrixPandas', 'prune', 'pruneNumPy', 'prunePandas'] | LiteralString | None = None
		Counting implementation. Pruning variants use known diagonal contributions. `None`,
		`'matrixMeanders'`, and unrecognized values select the source transfer algorithm.
	pathLikeWrite : PathLike[str] | None = None
		Optional output file or existing directory. `None` disables saving.
	state : StateMeanders | None = None
		Initialized computation state. When supplied, `state.kind` and `state.n` replace the
		corresponding arguments.
	CPUlimit : ConcurrencyLimit = None
		Accepted by the shared counting interface; this function does not use the value.
	suffix : str = ".countTotal"
		Filename suffix when saving is enabled.

	Returns
	-------
	stateCompleted : StateMeanders
		Completed or updated state. Read the total from `stateCompleted.totalMeanders`.

	Raises
	------
	TypeError
		If `state` is absent and either `kind` or `n` is `None`.

	Examples
	--------
	The runner in `easyRun/countMeanders.py` can continue an existing state.

		```python
		state = countMeanders(kind, n, flow, pathLikeWrite, state=state)
		```

	References
	----------
	[1] `mapFolding.dataBaskets.StateMeanders`

	[2] `mapFolding.kitFilesystem.saveTotalFAILearly`

	[3] `mapFolding.kitFilesystem.saveTotal`

	"""
#-------- memorialization instructions ---------------------------------------------

	if state is not None:
		kind = state.kind
		n = state.n
	else:
		if kind is None or n is None:
			message: str = 'kind and n are required when state is None.'
			raise TypeError(message)
		if kind == 'closed':
			state = StateMeanders(2 * n - 1, 'meanders')
		else:
			state = StateMeanders(n, kind)

	if pathLikeWrite is None:
		pathFilenameCountTotal: Path | None = None
	else:
		pathFilenameCountTotal = saveTotalFAILearly(makePathFilenameCount(pathLikeWrite, kind, str(n), suffix=suffix))

#-------- Algorithm selection and execution ---------------------------------------------

	if kind != 'semi' or n != 1:
		match flow:
			case 'prune':
				from mapFolding.synthesized.matrixMeanders.prune import doTheNeedful
			case 'pruneNumPy':
				from mapFolding.synthesized.matrixMeanders.pruneNumPy import doTheNeedful
			case 'prunePandas':
				from mapFolding.synthesized.matrixMeanders.prunePandas import doTheNeedful
			case 'matrixNumPy':
				from mapFolding.algorithms.matrixMeandersNumPy import doTheNeedful
			case 'matrixPandas':
				from mapFolding.algorithms.matrixMeandersPandas import doTheNeedful
			case 'matrixMeanders' | _:
				from mapFolding.algorithms.matrixMeanders import doTheNeedful

		state = doTheNeedful(state)
	countTotal: int = state.totalMeanders

#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameCountTotal is not None:
		saveTotal(pathFilenameCountTotal, countTotal)

	return state

def enumerateSequences(
	n: int
	, kind: LiteralString | Literal['stamps', 'stamps blank', 'meanders', 'meanders symmetric', 'semi', 'semi-meanders symmetric'] | None = None
	, flow: Literal['permutations', 'bilateral', 'bilateralConcurrent'] | None = None
	, pathLikeWrite: PathLike[str] | None = None
	, *
	, CPUlimit: ConcurrencyLimit = None
	, suffix: str = '.album'
) -> list[Folding] | int:
	"""Generate stamp-folding or meander permutations at sequence index `n`.

	(AI generated docstring)

	You can use this function to obtain the permutations for a selected sequence, or count
	semi-meanders without retaining the permutations. The permutation algorithm [1] returns a list.
	Either bilateral algorithm [2, 3] returns an integer count. Providing `pathLikeWrite` also saves
	the list or count to a new file.

	Parameters
	----------
	n : int
		Sequence index.
	kind : LiteralString | Literal['stamps', 'stamps blank', 'meanders', 'meanders symmetric', 'semi', 'semi-meanders symmetric'] | None = 'semi'
		Sequence family.
	flow : Literal['permutations', 'bilateral', 'bilateralConcurrent'] | None = 'permutations'
		`'permutations'` generates a list for any supported family. `'bilateral'` and
		`'bilateralConcurrent'` only count at the moment.
	pathLikeWrite : PathLike[str] | None = None
		New output filename or existing output directory. `None` skips all file output. An existing
		directory receives a filename formed from the canonical family name, `n`, and `suffix`.
		Relative file paths resolve under the default job directory [5].
	CPUlimit : ConcurrencyLimit = None
		Processor preference used only by `'bilateralConcurrent'`. `None`, `False`, and zero use the
		default processor count; `True` selects one worker. Positive integers request a worker
		count; negative integers reserve processors. Fractions between zero and one request a share
		of processors; fractions between minus one and zero reserve a share. Other floats truncate
		to integers. Resolution follows `mapFolding.beDRY.defineProcessorLimit` [6].
	suffix : str = '.album'
		Suffix for a generated filename. An explicit filename keeps its supplied suffix. This
		parameter does not select the output format.

	Returns
	-------
	album : list[Folding] | int
		List of leaf-label tuples in generation order for the permutation algorithm, or an integer
		count for either bilateral algorithm. Each tuple lists leaf labels by pile position. For
		`'meanders symmetric'`, each tuple has one more leaf than `n`.

	Output Files
	------------
	Path resolution creates missing parent directories [5]. Before computation,
	`mapFolding.kitFilesystem.saveTotalFAILearly` [7] rejects an existing output file with
	`FileExistsError`, then writes and reads a test integer for a new file. The test integer
	remains in the output file if subsequent computation fails. Successful permutation
	enumeration replaces the test integer with one comma-separated record per folding through
	`mapFolding.kitFilesystem.writeAlbum` [8]. Bilateral counting writes the integer count through
	`mapFolding.kitFilesystem.saveTotal` [9], even when the filename ends in `'.album'`.

	Examples
	--------
	The script `easyRun/permutations.py` supplies the computation choices and handles both return
	types with this call.

		```python
		album = enumerateSequences(n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit)
		total: int = album if isinstance(album, int) else len(album)
		```

	References
	----------
	[1] `mapFolding.algorithms.permutations.doTheNeedful`

	[2] `mapFolding.algorithms.permutationsBilateral.doTheNeedful`

	[3] `mapFolding.algorithms.permutationsBilateralConcurrent.doTheNeedful`

	[4] `mapFolding.oeis.getMetadata`

	[5] `mapFolding.kitFilesystem.makePathFilenameCount`

	[6] `mapFolding.beDRY.defineProcessorLimit`

	[7] `mapFolding.kitFilesystem.saveTotalFAILearly`

	[8] `mapFolding.kitFilesystem.writeAlbum`

	[9] `mapFolding.kitFilesystem.saveTotal`

	"""
	lookupParameterKind: dict[LiteralString | str | None, LiteralString | None] = {'a000136': 'stamps', 'A000136': 'stamps'
		, 'a000560': 'semi-meanders symmetric', 'A000560': 'semi-meanders symmetric', 'a000682': 'semi', 'A000682': 'semi'
		, 'a001011': 'stamps blank', 'A001011': 'stamps blank', 'a005316': 'meanders', 'A005316': 'meanders'
		, 'a077055': 'meanders symmetric', 'A077055': 'meanders symmetric', 'meanders symmetric': 'meanders symmetric'
		, 'meanders': 'meanders', 'semi-meanders symmetric': 'semi-meanders symmetric', 'semi-meanders': 'semi', 'semi': 'semi'
		, 'semimeanders symmetric': 'semi-meanders symmetric', 'semimeanders': 'semi', 'stamps blank': 'stamps blank', 'stamps': 'stamps'
		, None: None}
	kind = lookupParameterKind[kind]
	kind = kind or 'semi'

#-------- memorialization instructions ---------------------------------------------

	if pathLikeWrite is None:
		pathFilenameAlbum: Path | None = None
	else:
		pathFilenameAlbum = saveTotalFAILearly(makePathFilenameCount(pathLikeWrite, kind, str(n), suffix=suffix))

#-------- Algorithm selection and execution ---------------------------------------------

	if flow in {'bilateral', 'bilateralConcurrent'}:
		n = {'semi': n - 1, 'semi-meanders symmetric': n}[kind]
		symmetric: bool = kind == 'semi-meanders symmetric'
		if flow == 'bilateralConcurrent':
			from mapFolding.algorithms.permutationsBilateralConcurrent import doTheNeedful

			countTotal: int = doTheNeedful(n, symmetric=symmetric, concurrencyLimit=defineProcessorLimit(CPUlimit))
		else:
			from mapFolding.algorithms.permutationsBilateral import doTheNeedful

			countTotal = doTheNeedful(n, symmetric=symmetric)

		if pathFilenameAlbum is not None:
			saveTotal(pathFilenameAlbum, countTotal)
		return countTotal

	from mapFolding.algorithms.permutations import doTheNeedful

	album: list[Folding] = doTheNeedful(n, kind)

#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameAlbum is not None:
		writeAlbum(album, pathFilenameAlbum)

	return album
