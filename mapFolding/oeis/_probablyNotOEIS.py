# ruff:file-ignore[import-outside-top-level]
# TODO the following diagnostics suggest to me that there is a better paradigm for the flow control.
# pyright: reportPossiblyUnboundVariable=false
# pyright: reportArgumentType=false
# ty:ignore[invalid-argument-type]
from __future__ import annotations

from functools import cache
from hunterMakesPy import errorL33T
from itertools import chain
from mapFolding.kitFilesystem import getPathFilenameFoldsTotal, getPathRootJobDEFAULT, saveFoldsTotal, saveFoldsTotalFAILearly
from mapFolding.oeis._metadata import dictionaryOEISMapFolding
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike
	from pathlib import PurePath
	from typing import Literal

@cache
def getFoldsTotalKnown(mapShape: tuple[int, ...]) -> int:
	"""You can retrieve the known total number of distinct folding patterns for a given map shape.

	(AI generated docstring)

	This function queries the comprehensive dictionary of known folding totals constructed from OEIS
	sequence data. The function returns the total if the map shape matches a known value, or 0 if the
	shape is not found in the OEIS sequences.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.

	Returns
	-------
	foldingsTotal : int
		The known total number of distinct folding patterns for the given map shape, or 0 if the map
		shape does not match any known values in the OEIS sequences.

	Examples
	--------
	>>> from mapFolding.oeis import librarianLookupsFoldsTotalKnown
	>>> librarianLookupsFoldsTotalKnown((2, 3))
	10

	Implementation Details
	----------------------
	Map shapes are matched exactly as provided without internal sorting or normalization. The function
	uses `functools.cache` [1] for memoization to avoid reconstructing the lookup dictionary on
	repeated calls.

	See Also
	--------
	mapFolding.oeis.librarianConstructsDictionaryFoldsTotalKnown
		Construct the underlying lookup dictionary.

	References
	----------
	[1] functools.cache - Python standard library
		https://docs.python.org/3/library/functools.html#functools.cache
	"""
	lookupFoldsTotal: dict[tuple[int, ...], int] = makeDictionaryFoldsTotalKnown()
	return lookupFoldsTotal.get(tuple(mapShape), 0)

def makeDictionaryFoldsTotalKnown() -> dict[tuple[int, ...], int]:
	"""You can create a dictionary mapping map shapes to known folding totals from all OEIS sequences.

	(AI generated docstring)

	This function constructs a comprehensive lookup dictionary by extracting and transforming data
	from all map-folding OEIS sequences in `dictionaryOEISMapFolding`. The function applies each
	sequence's `getMapShape` function to its known indices to generate the corresponding map shapes,
	then pairs each shape with its folding total.

	Returns
	-------
	dictionaryFoldsTotalKnown : dict[tuple[int, ...], int]
		A dictionary where keys are tuple `mapShape` and values are the total number of distinct
		folding patterns for `mapShape`.

	Exclusions
	----------
	A007822 (symmetric foldings) is excluded from the dictionary because A007822 represents a
	constrained subset rather than the total count for each `mapShape`.

	See Also
	--------
	mapFolding.oeis.dictionaryOEISMapFolding
		Source metadata for map-folding OEIS sequences.
	"""
	return dict(chain.from_iterable(zip(map(oeisIDmetadata['getMapShape'], oeisIDmetadata['valuesKnown'].keys())
	, oeisIDmetadata['valuesKnown'].values(), strict=True) for oeisID, oeisIDmetadata in dictionaryOEISMapFolding.items() if oeisID != 'A007822'))

# TODO A long time ago, I had an explicit rule written in "oeis.py" that the module contained only OEIS stuff and ALL OEIS stuff.
# This function is fundamentally an OEIS function, but I have been trying to treat it the same as `countingFolds`. That mismatch
# is a major reason for the many problems I've had with semiotics and flow design. `oeisIDfor_n` _might_ be the correct identifier
# for this function. I created `oeisIDfor_n` a very very very long time ago, and I think my brain might have locked into it being
# a frontend for `countFolds`. I made `oeisIDfor_n` before I had a need for the `flow` parameter. Since creating the `flow`
# parameter, I have been trying to figure out how to put it into `oeisIDfor_n`. For a long time, `oeisIDfor_n` would call numba
# theorem2 because it was the fastest, but I made numba an optional dependency. All of these seemingly unrelated issues underscore
# the importance of the semiotics-first paradigm (for me).
def countingMeanders(oeisID: str, oeis_n: int, flow: str | None = None, pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None, *, CPUlimit: Limitation = None) -> int:
	"""Compute the n-th term of `oeisID`.

	(AI generated docstring)

	This function computes values for OEIS [1] sequences that require specialized algorithms (meanders
	[2], symmetric foldings) or closed-form formulas, as opposed to the general multidimensional
	map-folding algorithm accessible via `countFolds` [3]. The function dispatches to
	algorithm-specific implementations based on `oeisID` and `flow` parameters.

	The function name reflects that these computations are NOT standard map-folding counts: meanders
	use transfer matrix methods, symmetric foldings exploit symmetry constraints, and formula-based
	sequences compute directly without search.

	Parameters
	----------
	oeisID : str
		OEIS sequence identifier. Supported sequences fall into three categories:
		- Formula-based: A000136, A000560, A001010, A001011, A005315, A060206, A077460,
			A078591, A178961, A223094, A259702, A301620
		- Meanders: A000682 [5], A005316 [6]
		- Symmetric foldings: A007822
	oeis_n : int
		Sequence index (typically starting from 0 or 1, depending on OEIS sequence offset).
	flow : str | None = None
		Algorithm variant selector. Available values depend on `oeisID`:
		- For A000682, A005316: 'matrixMeanders' (default), 'matrixNumPy', 'matrixPandas'
		- For A007822: 'algorithm' (default), 'asynchronous', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'
		- For formula-based sequences: ignored (`flow` has no effect)
	CPUlimit : bool | float | int | None = None
		Processor usage limit for parallel algorithms (A007822 with certain `flow` values).
		Interpretation matches `countFolds.CPUlimit` [3]:
		- `False`, `None`, or `0`: use all available processors
		- `True`: limit to 1 processor
		- `int >= 1`: maximum number of processors
		- `0 < float < 1`: fraction of available processors
		- `-1 < float < 0`: fraction of processors to *not* use
		- `int <= -1`: number of processors to *not* use

	Returns
	-------
	countTotal : int
		The n-th term of the specified OEIS sequence.

	Raises
	------
	ValueError
		If `oeisID` matches A000682 or A005316 but receives an invalid internal state (programming
		error, not user error).

	Examples
	--------
	Formula-based sequence:

	>>> from mapFolding.basecamp import countingMeanders
	>>> countingMeanders('A000136', 3)
	8

	Meander computation with matrix algorithm:

	>>> countingMeanders('A000682', 5, flow='matrixMeanders')
	42

	Symmetric folding with Numba-optimized implementation:

	>>> countingMeanders('A007822', 6, flow='theorem2Numba')
	144

	See Also
	--------
	mapFolding.basecamp.countFolds
		General multidimensional map-folding computation.
	mapFolding.oeis.oeisIDfor_n
		Convenience function that routes to either `countFolds` or `countingMeanders` based on `oeisID`.

	Algorithm Details
	-----------------
	Meander Sequences (A000682, A005316)
		Meanders [2] represent configurations of non-intersecting curves crossing a line. The
		algorithms use transfer matrix methods to enumerate valid meander states. Initial arc codes
		(binary representations of curve crossings) are constructed based on sequence parity, then
		iterative transformations generate all valid meander states. See
		`mapFolding.reference.A000682facts` [7] and `mapFolding.reference.A005316facts` [8] for
		sequence-specific parameters.

	Symmetric Folding Sequence (A007822)
		A007822 counts foldings of 1×(2n) maps that are symmetric under 180-degree rotation.
		The algorithm exploits symmetry to reduce the search space. The `flow` parameter selects
		between serial, parallel (asynchronous), and optimized implementations (theorem2 variants).

	Formula-Based Sequences
		These sequences have closed-form definitions that compute values directly without search.
		Implementations reside in `mapFolding.algorithms.oeisIDbyFormula` [9]. Some formulas
		(A259702, A301620) are defined recursively in terms of A000682.

	References
	----------
	[1] OEIS - The On-Line Encyclopedia of Integer Sequences
		https://oeis.org/
	[2] Meanders - Wikipedia
		https://en.wikipedia.org/wiki/Meander_(mathematics)
	[3] mapFolding.basecamp.countFolds

	[5] OEIS A000682 - Semi-meanders: number of Folded meanders
		https://oeis.org/A000682
	[6] OEIS A005316 - Meanders
		https://oeis.org/A005316
	[7] mapFolding.reference.A000682facts
		Internal package reference (arc code patterns and bit-width data)
	[8] mapFolding.reference.A005316facts
		Internal package reference (boundary bucket distributions)
	[9] mapFolding.algorithms.oeisIDbyFormula
		Internal package reference (closed-form sequence formulas)

	"""
#-------- memorialization instructions ---------------------------------------------

	if pathLikeWriteFoldsTotal is not None:
		# For sequences without a natural mapShape, create filename based on oeisID and oeis_n
		if oeisID == 'A007822':
			# A007822 has a mapShape, so use the standard approach
			mapShapeForFilename: tuple[int, ...] = (1, 2 * oeis_n)
			pathFilenameFoldsTotal: Path | None = getPathFilenameFoldsTotal(mapShapeForFilename, pathLikeWriteFoldsTotal)
		else:
			# Other sequences don't have mapShape, so create filename directly
			filenameCountTotal: str = f"{oeisID}_n{oeis_n}.countTotal"
			pathLikeSherpa = Path(pathLikeWriteFoldsTotal)
			if pathLikeSherpa.is_dir():
				pathFilenameFoldsTotal = pathLikeSherpa / filenameCountTotal
			elif pathLikeSherpa.is_file() and pathLikeSherpa.is_absolute():
				pathFilenameFoldsTotal = pathLikeSherpa
			else:
				pathFilenameFoldsTotal = getPathRootJobDEFAULT() / pathLikeSherpa
			pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
		saveFoldsTotalFAILearly(pathFilenameFoldsTotal)
	else:
		pathFilenameFoldsTotal = None

#-------- Algorithm selection and execution ---------------------------------------------

	countTotal: int = -errorL33T
	matched_oeisID: bool = True

	match oeisID:
		case 'A000136':
			from mapFolding.algorithms.oeisIDbyFormula import A000136 as doTheNeedful
		case 'A000560':
			from mapFolding.algorithms.oeisIDbyFormula import A000560 as doTheNeedful
		case 'A001010':
			from mapFolding.algorithms.oeisIDbyFormula import A001010 as doTheNeedful
		case 'A001011':
			from mapFolding.algorithms.oeisIDbyFormula import A001011 as doTheNeedful
		case 'A005315':
			from mapFolding.algorithms.oeisIDbyFormula import A005315 as doTheNeedful
		case 'A060206':
			from mapFolding.algorithms.oeisIDbyFormula import A060206 as doTheNeedful
		case 'A077460':
			from mapFolding.algorithms.oeisIDbyFormula import A077460 as doTheNeedful
		case 'A078591':
			from mapFolding.algorithms.oeisIDbyFormula import A078591 as doTheNeedful
		case 'A178961':
			from mapFolding.algorithms.oeisIDbyFormula import A178961 as doTheNeedful
		case 'A223094':
			from mapFolding.algorithms.oeisIDbyFormula import A223094 as doTheNeedful
		case 'A259702':
			from mapFolding.algorithms.oeisIDbyFormula import A259702 as doTheNeedful
		case 'A301620':
			from mapFolding.algorithms.oeisIDbyFormula import A301620 as doTheNeedful
		case _:
			matched_oeisID = False
	if matched_oeisID:
		countTotal = doTheNeedful(oeis_n)
	else:
		matched_oeisID = True
		match oeisID:
			case 'A000682' | 'A005316':
				match flow:
					case 'matrixNumPy':
						from mapFolding.algorithms.matrixMeandersNumPyndas import doTheNeedful, MatrixMeandersNumPyState as State
					case 'matrixPandas':
						from mapFolding.algorithms.matrixMeandersNumPyndas import doTheNeedfulPandas as doTheNeedful, MatrixMeandersNumPyState as State
					case 'matrixMeanders' | _:
						from mapFolding.algorithms.matrixMeanders import doTheNeedful
						from mapFolding.dataBaskets import MatrixMeandersState as State

				boundary: int = oeis_n - 1

				if oeisID == 'A000682':
					if oeis_n == 1:
						return 1
					elif oeis_n & 0b1:
						arcCode: int = 0b101
					else:
						arcCode = 0b1
					listArcCodes: list[int] = [(arcCode << 1) | arcCode]
#															   0b1010 | 0b0101 is 0b1111, or 0xf
#															     0b10 |   0b01 is   0b11, or 0x3

					MAXIMUMarcCode: int = 1 << (2 * boundary + 4)
					while listArcCodes[-1] < MAXIMUMarcCode:
						arcCode = (arcCode << 4) | 0b0101  # e.g., 0b 10000 | 0b 0101 = 0b 10101
						listArcCodes.append((arcCode << 1) | arcCode)  # e.g., 0b 101010 | 0b 1010101 = 0b 111111 = 0x3f
						# Thereafter, append 0b1111 or 0xf, so, e.g., 0x3f, 0x3ff, 0x3fff, 0x3ffff, ...
						# See "mapFolding/reference/A000682facts.py"
					dictionaryMeanders: dict[int, int] = dict.fromkeys(listArcCodes, 1)

				elif oeisID == 'A005316':
					if oeis_n & 0b1:
						dictionaryMeanders: dict[int, int] = {0b1111: 1}  # 0xf
					else:
						dictionaryMeanders = {0b10110: 1}
				else:
					message: str = f"I received `{oeisID = }` for meander computation, but I only support 'A000682' and 'A005316' in this code path."
					raise ValueError(message)

				state = State(oeis_n, oeisID, boundary, dictionaryMeanders)
				countTotal = doTheNeedful(state)
			case 'A007822':
				mapShape: tuple[Literal[1], int] = (1, 2 * oeis_n)
				from mapFolding.beDRY import defineProcessorLimit
				concurrencyLimit: int = defineProcessorLimit(CPUlimit)

				from mapFolding.dataBaskets import SymmetricFoldsState
				symmetricState: SymmetricFoldsState = SymmetricFoldsState(mapShape)

				match flow:
					case 'asynchronous':
						from mapFolding.syntheticModules.A007822.asynchronous import doTheNeedful
						symmetricState = doTheNeedful(symmetricState, concurrencyLimit)
					case 'theorem2':
						from mapFolding.syntheticModules.A007822.theorem2 import doTheNeedful
						symmetricState = doTheNeedful(symmetricState)
					case 'theorem2Numba':
						from mapFolding.syntheticModules.A007822.theorem2Numba import doTheNeedful
						symmetricState = doTheNeedful(symmetricState)
					case 'theorem2Trimmed':
						from mapFolding.syntheticModules.A007822.theorem2Trimmed import doTheNeedful
						symmetricState = doTheNeedful(symmetricState)
					case _:
						from mapFolding.syntheticModules.A007822.algorithm import doTheNeedful
						symmetricState = doTheNeedful(symmetricState)

				countTotal = symmetricState.symmetricFolds
			case _:
				matched_oeisID = False

#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameFoldsTotal is not None:
		saveFoldsTotal(pathFilenameFoldsTotal, countTotal)

	return countTotal
