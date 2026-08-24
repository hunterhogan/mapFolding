"""Display available OEIS sequence information when executed as a module.

(AI generated docstring)

You can execute this module with `python -m mapFolding.oeis` to display the identifiers,
descriptions, and usage examples for implemented OEIS sequences. The module delegates output
generation to `getOEISids` [1].

Contents
--------
Functions
	getOEISids
		Display comprehensive information about all implemented OEIS sequences.

References
----------
[1] `mapFolding.oeis.getOEISids`
	Internal package reference for the OEIS sequence information display function.
"""
from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import keymap, merge
from hunterMakesPy import errorL33T
from mapFolding import ansiColorReset, ansiColors
from mapFolding.oeis import getValuesKnown
from mapFolding.oeis._metadata import dictionaryOEIS
from mapFolding.oeis._theSSOT import oeisIDsMapFoldingImplemented
from more_itertools import loops
from typing import TYPE_CHECKING
import sys
import time

if TYPE_CHECKING:
	from typing import Literal, LiteralString

@cache
def getTotalFoldsKnown(mapShape: tuple[int, ...]) -> int | None:
	"""You can retrieve the known total number of distinct folding patterns for a given map shape.

	(AI generated docstring)

	This function queries the comprehensive dictionary of known folding totals constructed from OEIS
	sequence data. The function returns the total if the map shape matches a known value, or None if
	the shape is not found in the OEIS sequences.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.

	Returns
	-------
	foldingsTotal : int | None
		The known total number of distinct folding patterns for the given map shape, or None if the
		map shape does not match any known values in the OEIS sequences.

	Examples
	--------
	>>> from mapFolding.oeis import getTotalFoldsKnown
	>>> getTotalFoldsKnown((2, 3))
	10

	Implementation Details
	----------------------
	Map shapes are matched exactly as provided without internal sorting or normalization. The function
	uses `functools.cache` [1] for memoization to avoid reconstructing the lookup dictionary on
	repeated calls.

	See Also
	--------
	mapFolding.oeis.makeLookupTotalFoldsKnown
		Construct the underlying lookup dictionary.

	References
	----------
	[1] functools.cache - Python standard library
		https://docs.python.org/3/library/functools.html#functools.cache
	"""
	lookupTotalFolds: dict[tuple[int, ...], int] = makeLookupTotalFoldsKnown()
	return lookupTotalFolds.get(tuple(mapShape))

def makeLookupTotalFoldsKnown() -> dict[tuple[int, ...], int]:
	"""You can create a dictionary mapping map shapes to known folding totals from all OEIS sequences.

	(AI generated docstring)

	This function constructs a comprehensive lookup dictionary by extracting and transforming data
	from all map-folding OEIS sequences in `dictionaryOEIS`. The function applies `makeMapShape` to
	each sequence's known indices to generate the corresponding map shapes,
	then pairs each shape with its folding total.

	Returns
	-------
	dictionaryTotalFoldsKnown : dict[tuple[int, ...], int]
		A dictionary where keys are tuple `mapShape` and values are the total number of distinct
		folding patterns for `mapShape`.

	Exclusions
	----------
	A007822 (symmetric foldings) is excluded from the dictionary because A007822 represents a
	constrained subset rather than the total count for each `mapShape`.

	"""
	return merge(*[keymap(partial(makeMapShape, oeisID), getValuesKnown(oeisID)) for oeisID in oeisIDsMapFoldingImplemented])

def makeMapShape(oeisID: LiteralString | Literal['A000136', 'A001415', 'A001416', 'A001417', 'A195646', 'A001418', 'A007822'], n: int) -> tuple[int, ...]:
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
				message: str = f"I received `{oeisID = }`, but it is not implemented in `makeMapShape`."
				raise ValueError(message)
	else:
		message = f"I received `{oeisID = }` and `{n = }`, but {oeisID} is not defined for n < {dictionaryOEIS[oeisID]['offset']}."
		raise ValueError(message)
	return mapShape

def printEasyRunBenchmark(oeisID: str, n: int, computed: int, timeStart: float, *, ratio: bool = False) -> None:
	"""Print a benchmark comparison line for an OEIS sequence value.

	Outputs a tab-separated line showing whether the computed value matches the known OEIS value, the
	index, both values, optionally their ratio, and elapsed time.

	Parameters
	----------
	oeisID : str
		The OEIS sequence identifier (e.g., 'A000136').
	n : int
		The index/term number in the sequence.
	computed : int
		The computed value to compare against the known OEIS value.
	timeStart : float
		The start time from `time.perf_counter()` for elapsed time calculation.
	ratio : bool = False
		If True and `computed` is non-zero, also print the ratio `known / computed`.

	Notes
	-----
	The output uses ANSI color codes: green for match, red for mismatch. The known value is retrieved
	from the OEIS data via `getValuesKnown`. If the sequence or term is not found, `known` will be a
	large negative sentinel value.
	"""
	known: int = getValuesKnown(oeisID).get(n, -errorL33T)
	match: bool = computed == known
	sys.stdout.write(
		f"{n:2}\t"
		f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}{match}{ansiColorReset}\t"
		f"{time.perf_counter() - timeStart:5.2f}\t"
		f"{computed}\t{known}\t"
	)
	if ratio and computed:
		integer = (known / computed).is_integer()
		sys.stdout.write(f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[integer]}{known / computed}{ansiColorReset}\t")
	sys.stdout.write(f"{ansiColorReset}\n")

def printEasyRunHeader(oeisID: str, flow: str) -> None:
	"""Print a colored header line for an easy run benchmark session.

	Outputs the OEIS ID and flow identifier in distinct colors based on their hash values, followed by
	a color reset.

	Parameters
	----------
	oeisID : str
		The OEIS sequence identifier (e.g., 'A000136').
	flow : str
		A flow identifier string (e.g., 'main', 'test', 'benchmark').

	Notes
	-----
	Colors are selected by converting the string to a base-36 integer and modulo the number of
	available ANSI colors. This provides consistent coloring for the same identifiers across runs.
	"""
	sys.stdout.write(f"{ansiColors[int(oeisID, 36) % len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow, 36) % len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + '\n')

if __name__ == "__main__":
	from mapFolding.oeis import getOEISids
	getOEISids()
