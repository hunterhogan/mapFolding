# ruff: file-ignore[unused-function-argument]
# ruff:file-ignore[import-outside-top-level]
# TODO the following diagnostics suggest to me that there is a better paradigm for the flow control.
# pyright: reportPossiblyUnboundVariable=false
# pyright: reportArgumentType=false
# ty:ignore[invalid-argument-type]
from __future__ import annotations

from mapFolding.kitFilesystem import getPathRootJobDEFAULT, saveFoldsTotal, saveFoldsTotalFAILearly
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike

def countMeanders(oeisID: str, oeis_n: int, flow: str = '', pathLikeWriteTotal: PathLike[str] | None = None, *, CPUlimit: Limitation = None) -> int:
	"""Compute a native meander sequence term.

	This entry point computes A000682 and A005316 with a matrix meander algorithm. Formula and
	symmetric-folding dispatch belongs to `oeisIDfor_n`.

	Parameters
	----------
	oeisID : str
		'A000682' or 'A005316'.
	oeis_n : int
		Sequence index.
	flow : str = ''
		'matrixMeanders' or '' for the native implementation, 'matrixNumPy', or 'matrixPandas'.
	pathLikeWriteTotal : PathLike[str] | None = None
		Optional output path for the computed total.
	CPUlimit : bool | float | int | None = None
		Processor limit from the shared counting entry-point contract.

	Returns
	-------
	countTotal : int
		The requested sequence term.

	Raises
	------
	ValueError
		If `oeisID` is not A000682 or A005316.

	Examples
	--------
	>>> from mapFolding.oeis import countMeanders
	>>> countMeanders('A000682', 5, flow='matrixMeanders')
	42

	See Also
	--------
	mapFolding.oeis.oeisIDfor_n
		Dispatch every implemented OEIS sequence.

	"""
#-------- memorialization instructions ---------------------------------------------

	if pathLikeWriteTotal is not None:
		filenameCountTotal: str = f"{oeisID}_n{oeis_n}.countTotal"
		pathLikeSherpa = Path(pathLikeWriteTotal)
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

	match flow:
		case 'matrixNumPy':
			from mapFolding.algorithms.matrixMeandersNumPy import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersNumPyState as State
		case 'matrixPandas':
			from mapFolding.algorithms.matrixMeandersPandas import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersNumPyState as State
		case 'matrixMeanders' | _:
			from mapFolding.algorithms.matrixMeanders import doTheNeedful
			from mapFolding.dataBaskets import MatrixMeandersState as State

	boundary: int = oeis_n - 1

	# TODO Consider: If A000682 is essentially A000136 * leavesTotal, then my graphs of A000136 are
	# _literal_ graphs of A000682. Since Theorem 2 applies to A000136, it must apply to A000682. Can I
	# use the graphs to find the midpoint of an A000682 computation using the matrix algorithm? The
	# problem with the matrix algorithm is memory usage. Unique signatures (buckets) grows
	# predictably. Cutting the count in half... In `doTheNeedful`, I used `while state.boundary > 0:`
	# and the ratio trick to find the midpoint: it didn't work.
	if oeisID == 'A000682':
		if oeis_n == 1:
			return 1
		elif oeis_n & 0b1:
			arcCode: int = 0b101
		else:
			arcCode = 0b1
		listArcCodes: list[int] = [(arcCode << 1) | arcCode]
#													   0b1010 | 0b0101 is 0b1111, or 0xf
#														 0b10 |   0b01 is   0b11, or 0x3

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
		message: str = f"I received `{oeisID = }` for meander computation, but I only support 'A000682' and 'A005316'."
		raise ValueError(message)

	state = State(oeis_n, oeisID, boundary, dictionaryMeanders)
	countTotal: int = doTheNeedful(state)

#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameFoldsTotal is not None:
		saveFoldsTotal(pathFilenameFoldsTotal, countTotal)

	return countTotal
