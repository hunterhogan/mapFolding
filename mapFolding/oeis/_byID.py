from __future__ import annotations

from mapFolding.basecamp import countFolds, countFoldsSymmetric
from mapFolding.oeis import byFormula
from mapFolding.oeis._meanders import countMeanders
from mapFolding.oeis._metadata import _formatOEISid, dictionaryOEISImplemented, dictionaryOEISMapFolding
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from mapFolding.oeis._dataBaskets import MetadataOEISid, MetadataOEISidMapFolding
	from os import PathLike
	from typing import Unpack

class KeywordArgumentsCount(TypedDict, total=False):
	flow: str
	pathLikeWriteTotal: PathLike[str] | None
	CPUlimit: Limitation

def oeisIDfor_n(oeisID: str, n: int, f: str = '', **keywordArguments: Unpack[KeywordArgumentsCount]) -> int:
	"""You can calculate the value a(n) for a specified OEIS ID and index.

	(AI generated docstring)

	This function computes every implemented OEIS sequence by dispatching to a map-folding, meander,
	symmetric-folding, or formula implementation. For small values or trivial cases, the function
	returns known values from cached OEIS data.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to evaluate.
	n : int
		A non-negative integer index for which to calculate the sequence value.
	f : str = ''
		The formula selector. A nonempty value selects a formula for sequences that also have a native
		counting implementation.
	**keywordArguments
		Keyword arguments for `countFolds`, `countMeanders`, or `countFoldsSymmetric`: `flow`,
		`pathLikeWriteTotal`, and `CPUlimit`. These arguments are not passed to formula functions.

	Returns
	-------
	a_of_n : int
		The value a(n) of the specified OEIS sequence.

	Raises
	------
	ValueError
		If `n` is not a non-negative integer.
	ArithmeticError
		If `n` is below the sequence's defined offset.

	Examples
	--------
	>>> from mapFolding.oeis import oeisIDfor_n
	>>> oeisIDfor_n('A001415', 19)
	87811001880539136

	See Also
	--------
	mapFolding.oeis.countMeanders
		Compute A000682 and A005316 with native meander algorithms.
	mapFolding.basecamp.countFolds
		General multidimensional map-folding computation.

	References
	----------
	[1] OEIS - The On-Line Encyclopedia of Integer Sequences
		https://oeis.org/
	[2] mapFolding.basecamp.countFolds
	"""
	oeisID = _formatOEISid(oeisID)

	if not isinstance(n, int) or n < 0:
		message: str = f"I received `{n = }` in the form of `{type(n) = }`, but it must be non-negative integer in the form of `{int}`."
		raise ValueError(message)

	metadataOEISid: MetadataOEISidMapFolding | MetadataOEISid = dictionaryOEISImplemented[oeisID]

	if n < metadataOEISid['offset']:
		message: str = f"I received `{n = }`, but OEIS sequence `{oeisID = }` is not defined for values below `offset = {metadataOEISid['offset']}`."
		raise ArithmeticError(message)

	if n <= 1:
		foldsTotal: int = metadataOEISid['valuesKnown'][n]
	else:
		match oeisID:
			case 'A000136' if f:
				foldsTotal = byFormula.A000136(n, f)
			case 'A000560':
				foldsTotal = byFormula.A000560(n, f)
			case 'A000682' if f:
				foldsTotal = byFormula.A000682(n, f)
			case 'A001010':
				foldsTotal = byFormula.A001010(n, f)
			case 'A001011':
				foldsTotal = byFormula.A001011(n, f)
			case 'A005315':
				foldsTotal = byFormula.A005315(n, f)
			case 'A005316' if f:
				foldsTotal = byFormula.A005316(n, f)
			case 'A060206':
				foldsTotal = byFormula.A060206(n, f)
			case 'A077014':
				foldsTotal = byFormula.A077014(n, f)
			case 'A077054':
				foldsTotal = byFormula.A077054(n, f)
			case 'A077460':
				foldsTotal = byFormula.A077460(n, f)
			case 'A078591':
				foldsTotal = byFormula.A078591(n, f)
			case 'A078592':
				foldsTotal = byFormula.A078592(n, f)
			case 'A085973':
				foldsTotal = byFormula.A085973(n, f)
			case 'A208357':
				foldsTotal = byFormula.A208357(n, f)
			case 'A217310':
				foldsTotal = byFormula.A217310(n, f)
			case 'A217318':
				foldsTotal = byFormula.A217318(n, f)
			case 'A223093':
				foldsTotal = byFormula.A223093(n, f)
			case 'A223094':
				foldsTotal = byFormula.A223094(n, f)
			case 'A223095':
				foldsTotal = byFormula.A223095(n, f)
			case 'A227167':
				foldsTotal = byFormula.A227167(n, f)
			case 'A259702':
				foldsTotal = byFormula.A259702(n, f)
			case 'A301620':
				foldsTotal = byFormula.A301620(n, f)
			case 'A330269':
				foldsTotal = byFormula.A330269(n, f)
			case 'A333971':
				foldsTotal = byFormula.A333971(n, f)
			case 'A334615':
				foldsTotal = byFormula.A334615(n, f)
			case 'A337581':
				foldsTotal = byFormula.A337581(n, f)
			case 'A000682' | 'A005316':
				foldsTotal = countMeanders(oeisID, n, **keywordArguments)
			case 'A007822':
				foldsTotal = countFoldsSymmetric((1, 2 * n), **keywordArguments)
			case _:
				foldsTotal = countFolds(
					mapShape=dictionaryOEISMapFolding[oeisID]['getMapShape'](n)
					, **keywordArguments
				)

	return foldsTotal
