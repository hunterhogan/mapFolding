from __future__ import annotations

from hunterMakesPy.parseParameters import intInnit
from mapFolding.basecamp import countFolds, countFoldsSymmetric, countMeanders
from mapFolding.oeis import byFormula, getMetadata, makeMapShape, oeisIDsMapFoldingImplemented
from mapFolding.oeis._beDRY import formatOEISid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.oeis._dataBaskets import MetadataOEISid
	from mapFolding.theTypes import KeywordArgumentsCount, OEISid
	from typing import LiteralString, Unpack

def oeisIDfor_n(oeisID: OEISid, n: int, f: LiteralString | None = None, **keywordArguments: Unpack[KeywordArgumentsCount]) -> int:
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
		`pathLikeWrite`, and `CPUlimit`. These arguments are not passed to formula functions.

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
	oeisID = formatOEISid(oeisID)

	# TODO =EndNotes= unambiguous but technically malformed user input: Be nice. Try to deal with
	# _unambiguous_ input that is malformed, such as an integer value passed as a `str` or `float`
	# type, instead of halting execution. `hunterMakesPy.parseParameters` and `datastructures` have
	# easy-to-implement, robust functions to do the hard work.
	if not isinstance(n, int):
		qq: list[int] = intInnit([n], 'n', int)
		if len(qq) == 1:
			n = qq.pop()

	if not isinstance(n, int) or n < 0:
		message: str = f"I received `{n = }` in the form of `{type(n) = }`, but I need a non-negative integer in the form of `{int}`."
		raise ValueError(message)

	metadata: MetadataOEISid = getMetadata(oeisID)

	if n < metadata['offset']:
		message: str = f"I received `{n = }`, but OEIS sequence `{oeisID = }` is not defined for values below `offset = {metadata['offset']}`."
		raise ArithmeticError(message)

	if n == metadata['offset']:
		foldsTotal: int = metadata['valuesKnown'][n]
	elif not f and oeisID in oeisIDsMapFoldingImplemented:
		foldsTotal = countFolds(mapShape=makeMapShape(oeisID, n), **keywordArguments)
	else:
		match oeisID:
			case 'A000682' if not f:
				foldsTotal = countMeanders('semi', n, **keywordArguments)
			case 'A005316' if not f:
				foldsTotal = countMeanders('meanders', n, **keywordArguments)
			case 'A007822' if not f:
				foldsTotal = countFoldsSymmetric(makeMapShape(oeisID, n), **keywordArguments)
			case 'A007822':
				foldsTotal = byFormula.A007822(n, f)
			case 'A000136':
				foldsTotal = byFormula.A000136(n, f)
			case 'A000560':
				foldsTotal = byFormula.A000560(n, f)
			case 'A000682':
				foldsTotal = byFormula.A000682(n, f)
			case 'A001010':
				foldsTotal = byFormula.A001010(n, f)
			case 'A001011':
				foldsTotal = byFormula.A001011(n, f)
			case 'A005315':
				foldsTotal = byFormula.A005315(n, f)
			case 'A005316':
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
			case 'A259689':
				foldsTotal = byFormula.A259689(n, f)
			case 'A259702':
				foldsTotal = byFormula.A259702(n, f)
			case 'A301620':
				foldsTotal = byFormula.A301620(n, f)
			case 'A333971':
				foldsTotal = byFormula.A333971(n, f)
			case 'A334615':
				foldsTotal = byFormula.A334615(n, f)
			case 'A337581':
				foldsTotal = byFormula.A337581(n, f)
			case _:
				message = f"I received `{oeisID = }`, but I couldn't find a formula for it."
				raise ValueError(message)

	return foldsTotal
