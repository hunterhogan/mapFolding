from __future__ import annotations

from mapFolding.basecamp import countFolds
from mapFolding.oeis._metadata import _formatOEISid, dictionaryOEISMapFolding

def oeisIDfor_n(oeisID: str, n: int) -> int:
	"""You can calculate the value a(n) for a specified OEIS ID and index.

	(AI generated docstring)

	This function computes OEIS [1] sequence values by dispatching to the appropriate algorithm based
	on the sequence ID. For small values or trivial cases, the function returns known values from
	cached OEIS data. For larger indices, the function invokes `countFolds` [2] with the map shape
	corresponding to the sequence index.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to evaluate.
	n : int
		A non-negative integer index for which to calculate the sequence value.

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
	mapFolding.oeis.countingMeanders
		Compute values for sequences requiring specialized algorithms.
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

	mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)

	if n <= 1 or len(mapShape) < 2:
		offset: int = dictionaryOEISMapFolding[oeisID]['offset']
		if n < offset:
			message: str = f"I received `{n = }`, but OEIS sequence `{oeisID = }` is not defined for values below `{offset = }`."
			raise ArithmeticError(message)
		foldsTotal: int = dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]
	else:
		foldsTotal = countFolds(mapShape=mapShape)

	return foldsTotal
