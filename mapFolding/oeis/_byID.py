from __future__ import annotations

from mapFolding.basecamp import countFolds, countFoldsSymmetric, countMeanders
from mapFolding.oeis import byFormula, getMapShape
from mapFolding.oeis._metadata import _formatOEISid, dictionaryOEIS
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.oeis._dataBaskets import MetadataOEISid
	from mapFolding.theTypes import KeywordArgumentsCount
	from typing import Unpack

class _FormulaForN(Protocol):
	def __call__(self, n: int, f: str = ...) -> int: ...

# TODO Learn a better way to handle default values in this situation.
def _evaluateFormulaForN(formulaForN: _FormulaForN, n: int, f: str) -> int:
	if f:
		foldsTotal: int = formulaForN(n, f)
	else:
		foldsTotal = formulaForN(n)
	return foldsTotal

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

	metadataOEISid: MetadataOEISid = dictionaryOEIS[oeisID]

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
				foldsTotal = _evaluateFormulaForN(byFormula.A000560, n, f)
			case 'A000682' if f:
				foldsTotal = byFormula.A000682(n, f)
			case 'A001010':
				foldsTotal = _evaluateFormulaForN(byFormula.A001010, n, f)
			case 'A001011':
				foldsTotal = _evaluateFormulaForN(byFormula.A001011, n, f)
			case 'A005315':
				foldsTotal = _evaluateFormulaForN(byFormula.A005315, n, f)
			case 'A005316' if f:
				foldsTotal = byFormula.A005316(n, f)
			case 'A060206':
				foldsTotal = _evaluateFormulaForN(byFormula.A060206, n, f)
			case 'A077014':
				foldsTotal = _evaluateFormulaForN(byFormula.A077014, n, f)
			case 'A077054':
				foldsTotal = _evaluateFormulaForN(byFormula.A077054, n, f)
			case 'A077460':
				foldsTotal = _evaluateFormulaForN(byFormula.A077460, n, f)
			case 'A078591':
				foldsTotal = _evaluateFormulaForN(byFormula.A078591, n, f)
			case 'A078592':
				foldsTotal = _evaluateFormulaForN(byFormula.A078592, n, f)
			case 'A085973':
				foldsTotal = _evaluateFormulaForN(byFormula.A085973, n, f)
			case 'A208357':
				foldsTotal = _evaluateFormulaForN(byFormula.A208357, n, f)
			case 'A217310':
				foldsTotal = _evaluateFormulaForN(byFormula.A217310, n, f)
			case 'A217318':
				foldsTotal = _evaluateFormulaForN(byFormula.A217318, n, f)
			case 'A223093':
				foldsTotal = _evaluateFormulaForN(byFormula.A223093, n, f)
			case 'A223094':
				foldsTotal = _evaluateFormulaForN(byFormula.A223094, n, f)
			case 'A223095':
				foldsTotal = _evaluateFormulaForN(byFormula.A223095, n, f)
			case 'A227167':
				foldsTotal = _evaluateFormulaForN(byFormula.A227167, n, f)
			case 'A259702':
				foldsTotal = _evaluateFormulaForN(byFormula.A259702, n, f)
			case 'A301620':
				foldsTotal = _evaluateFormulaForN(byFormula.A301620, n, f)
			case 'A333971':
				foldsTotal = _evaluateFormulaForN(byFormula.A333971, n, f)
			case 'A334615':
				foldsTotal = _evaluateFormulaForN(byFormula.A334615, n, f)
			case 'A337581':
				foldsTotal = _evaluateFormulaForN(byFormula.A337581, n, f)
			case 'A000682' | 'A005316':
				foldsTotal = countMeanders(oeisID, n, **keywordArguments)
			case 'A007822':
				foldsTotal = countFoldsSymmetric(getMapShape(oeisID, n), **keywordArguments)
			case _:
				foldsTotal = countFolds(mapShape=getMapShape(oeisID, n), **keywordArguments)

	return foldsTotal
