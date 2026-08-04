# TODO Why is a literal str not a str? So annoying.
# pyright: reportArgumentType=false
# ty: ignore[invalid-argument-type]
from __future__ import annotations

from functools import partial
from humpy_cytoolz import keymap, merge
from mapFolding.oeis._metadata import dictionaryOEIS, getValuesKnown
from mapFolding.oeis._theSSOT import oeisIDsMapFolding
from more_itertools import loops
from typing import Literal

def getMapShape(oeisID: Literal['A000136', 'A001415', 'A001416', 'A001417', 'A195646', 'A001418', 'A007822'], n: int) -> tuple[int, ...]:
	"""Get the map shape for a given OEIS ID and index n."""
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
				message: str = f"I received `{oeisID = }`, but it is not implemented in `getMapShape`."
				raise ValueError(message)
	else:
		message = f"I received `{oeisID = }` and `{n = }`, but {oeisID} is not defined for n < {dictionaryOEIS[oeisID]['offset']}."
		raise ValueError(message)
	return mapShape

def makeDictionaryFoldsTotalKnown() -> dict[tuple[int, ...], int]:
	"""You can create a dictionary mapping map shapes to known folding totals from all OEIS sequences.

	(AI generated docstring)

	This function constructs a comprehensive lookup dictionary by extracting and transforming data
	from all map-folding OEIS sequences in `dictionaryOEIS`. The function applies `getMapShape` to
	each sequence's known indices to generate the corresponding map shapes,
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

	"""
	return merge(*[keymap(partial(getMapShape, oeisID), getValuesKnown(oeisID)) for oeisID in oeisIDsMapFolding])
