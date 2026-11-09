from __future__ import annotations

from hunterMakesPy.parseParameters import intInnit
from mapFolding.basecamp import countFolds, countFoldsSymmetric, countMeanders
from mapFolding.oeis import getMetadata, makeMapShape, settingsOEIS
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.oeis._symbolic import getValueSymbolic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.oeis._dataBaskets import MetadataOEISid
	from mapFolding.theTypes import OEISid, 形KeywordArgumentsCount
	from typing import LiteralString, Unpack

def oeisIDfor_n(oeisID: OEISid, n: int, f: LiteralString | None = None, **keywordArguments: Unpack[形KeywordArgumentsCount]) -> int:
	"""You can calculate the value a(n) for a specified OEIS ID and index.

	(AI generated docstring)

	This function computes every implemented OEIS sequence by dispatching to a map-folding, meander,
	symmetric-folding, or formula implementation. For small values or trivial cases, the function
	returns known values from cached OEIS data. Symbolic relationships derive values by chaining
	relations to basis sequences. Known values supply basis values when available, and recursive calls
	compute the others. Native counting implementations retain their default dispatch when `f` is
	`None`.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to evaluate.
	n : int
		A non-negative integer index for which to calculate the sequence value.
	f : str | None = None
		The formula selector. A nonempty value names the basis sequences of the formula, including for
		sequences that also have a native counting implementation.
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

	See Also
	--------
	`mapFolding.basecamp.countMeanders`
		Compute A000682 and A005316 with native meander algorithms.
	`mapFolding.basecamp.countFolds`
		General multidimensional map-folding computation.

	Examples
	--------
	>>> from mapFolding.oeis import oeisIDfor_n
	>>> oeisIDfor_n('A001415', 19)
	87811001880539136
	"""
	oeisID = formatOEISid(oeisID)

	#=EndNotes##unambiguousInput=
	# Normalize a clear integer value supplied as another type, then validate the index.
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

	if ((f or oeisID not in settingsOEIS.boxOfIDsCounting)
		and (valueSymbolic := getValueSymbolic(oeisID, n, f, oeisIDfor_n)) is not None):
		totalFolds: int = valueSymbolic
	elif n == metadata['offset']:
		totalFolds = metadata['valuesKnown'][n]
	elif not f and oeisID in settingsOEIS.boxOfIDsMapFolding:
		totalFolds = countFolds(mapShape=makeMapShape(oeisID, n), **keywordArguments)
	else:
		match oeisID:
			case 'A000682' if not f:
				state = countMeanders('semi', n, **keywordArguments)
				totalFolds = state.totalMeanders
			case 'A005316' if not f:
				state = countMeanders('meanders', n, **keywordArguments)
				totalFolds = state.totalMeanders
			case 'A007822' if not f:
				totalFolds = countFoldsSymmetric(makeMapShape(oeisID, n), **keywordArguments)
			case _:
				message = f"I received `{oeisID = }`, but I couldn't find a formula for it."
				raise ValueError(message)

	return totalFolds
