from __future__ import annotations

from functools import partial
from humpy_cytoolz import keymap, merge
from mapFolding.oeis._metadata import dictionaryOEIS, getValuesKnown
from mapFolding.oeis._theSSOT import settingsOEIS
from more_itertools import loops
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from typing import Literal, LiteralString

def getTotalFoldsKnown(mapShape: tuple[int, ...]) -> int | None:
	"""Retrieve the known folding total for an exact map shape.

	(AI generated docstring)

	You can use this function to compare a map shape with the known totals assembled by
	`makeLookupTotalFoldsKnown`. This function returns the stored total for `mapShape`, or
	`None` when the lookup contains no matching shape.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map dimensions in the order used by the sequence's shape conversion. Dimensions are not
		sorted and dimensions of length one are not removed.

	Returns
	-------
	foldingsTotal : int | None
		Known folding total for `mapShape`, or `None` when the shape is absent from the lookup.

	See Also
	--------
	`makeLookupTotalFoldsKnown`
		Retrieve the complete shape-to-total lookup instead of one total.

	Examples
	--------
	`mapFolding.kitAST.RecipeJob.fromMapShape` queries the known total before estimating a
	job's size.

		```python
		from mapFolding.oeis import getTotalFoldsKnown

		getTotalFoldsKnown(state.mapShape)
		```
	"""
	lookupTotalFolds: dict[tuple[int, ...], int] = makeLookupTotalFoldsKnown()
	return lookupTotalFolds.get(tuple(mapShape))

def makeLookupTotalFoldsKnown() -> dict[tuple[int, ...], int]:
	"""Build a map-shape lookup from known map-folding sequence values.

	(AI generated docstring)

	You can use this function to inspect known folding totals for every map shape represented by
	`mapFolding.oeis.settingsOEIS.boxOfIDsMapFolding`. This function retrieves known
	values with `mapFolding.oeis.getValuesKnown` and converts sequence indices to dimensions
	with `makeMapShape`.

	Returns
	-------
	lookupTotalFolds : dict[tuple[int, ...], int]
		New dictionary mapping exact dimension tuples to known folding totals. When sequences
		produce the same shape, the later sequence in the registered tuple supplies the value.

	Exclusions
	----------
	Only registered map-folding sequences contribute values. `A007822`, which counts symmetric
	foldings, is excluded. This function does not filter retrieval sentinels before shape conversion.
	"""
	return merge(*[keymap(partial(makeMapShape, oeisID), getValuesKnown(oeisID)) for oeisID in settingsOEIS.boxOfIDsMapFolding])

def makeMapShape(oeisID: LiteralString | Literal['A000136', 'A001415', 'A001416', 'A001417', 'A195646', 'A001418', 'A007822'], n: int) -> tuple[int, ...]:
	"""Convert a supported OEIS sequence identifier and index to map dimensions.

	(AI generated docstring)

	You can use this function to obtain the map shape corresponding to `n` for one of the supported
	folding sequences. This function checks the offset in `mapFolding.oeis._metadata.dictionaryOEIS`,
	then returns the sequence's dimension tuple.

	Shape Conversion
	----------------
	`A000136` produces `(1, n)`, `A001415` produces `(2, n)`, and `A001416` produces `(3, n)`.
	`A001417` produces `n` dimensions of length two, and `A195646` produces `n` dimensions of
	length three. `A001418` produces `(n, n)`, and `A007822` produces `(1, 2 * n)`.

	Parameters
	----------
	oeisID : LiteralString | Literal['A000136', 'A001415', 'A001416', 'A001417', 'A195646', 'A001418', 'A007822']
		Sequence identifier as written above. This function does not normalize letter case or
		surrounding whitespace. An identifier absent from the registry propagates `KeyError`.
	n : int
		Sequence index at or above the stored offset.

	Returns
	-------
	mapShape : tuple[int, ...]
		Dimensions of the map associated with the requested sequence term.

	Raises
	------
	ValueError
		If `n` is below the stored offset or a registered identifier has no shape conversion.

	Examples
	--------
	`easyRun/countFolds.py` obtains dimensions before counting the folds for a sequence term.

		```python
		from mapFolding.oeis import makeMapShape

		mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
		```
	"""
	if dictionaryOEIS[oeisID]['offset'] <= n:
		match oeisID:
			case 'A000136':
				mapShape: tuple[int, ...] = (1, n)
			case 'A001415':
				mapShape = (2, n)
			case 'A001416':
				mapShape = (3, n)
			case 'A001417':
				mapShape = tuple(2 for _dimension in loops(n))
			case 'A195646':
				mapShape = tuple(3 for _dimension in loops(n))
			case 'A001418':
				mapShape = (n, n)
			case 'A007822':
				mapShape = (1, 2 * n)
			case _:
				message: str = f"I received `{oeisID = }`, but it is not implemented in `makeMapShape`."
				raise ValueError(message)
	else:
		message = f"I received `{oeisID = }` and `{n = }`, but {oeisID} is not defined for n < {dictionaryOEIS[oeisID]['offset']}."
		raise ValueError(message)
	return mapShape
