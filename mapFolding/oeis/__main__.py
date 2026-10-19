"""Inspect known OEIS values, map shapes, and triangle data.

(AI generated docstring)

You can use this module's functions to look up known folding totals, convert sequence indices
to map shapes, and print benchmark comparisons for individual terms or triangle rows.

Contents
--------
Functions
	getTotalFoldsKnown
		Retrieve a known folding total for an exact map shape.
	makeLookupTotalFoldsKnown
		Build a map-shape lookup from registered map-folding sequences.
	makeMapShape
		Convert a supported sequence identifier and index to map dimensions.
	printEasyRunBenchmark
		Print computed and known values with elapsed time and optional ratio.
	printEasyRunHeader
		Print two labels with reproducible terminal colors.
	printEasyRunTriangle
		Compare a computed triangle row with known cells and align differing rows.

Module Execution
----------------
Executing this module displays sequence identifiers, descriptions, and usage examples through
`mapFolding.oeis.getOEISids` [1]. The installed `getOEISids` console command provides the same
listing.

References
----------
[1] `mapFolding.oeis.getOEISids`

"""
from __future__ import annotations

from functools import partial
from humpy_cytoolz import keymap, merge
from hunterMakesPy import ansiColor, ansiColorReset, errorL33T
from itertools import chain, repeat
from mapFolding.oeis._beDRY import formatBenchmarkRatio
from mapFolding.oeis._metadata import dictionaryOEIS, getTriangleIndices, getValuesKnown
from mapFolding.oeis._theSSOT import oeisIDsMapFoldingImplemented
from more_itertools import loops
from operator import eq
from typing import TYPE_CHECKING
import anyascii
import sys
import time

if TYPE_CHECKING:
	from collections.abc import Sequence
	from mapFolding.theTypes import OEISid
	from typing import Literal, LiteralString

def getTotalFoldsKnown(mapShape: tuple[int, ...]) -> int | None:
	"""Retrieve the known folding total for an exact map shape.

	(AI generated docstring)

	You can use this function to compare a map shape with the known totals assembled by
	`makeLookupTotalFoldsKnown` [1]. This function returns the stored total for `mapShape`, or
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
	`mapFolding.kitAST.RecipeJob.fromMapShape` [2] queries the known total before estimating a
	job's size.

		```python
		from mapFolding.oeis import getTotalFoldsKnown

		getTotalFoldsKnown(state.mapShape)
		```

	References
	----------
	[1] `makeLookupTotalFoldsKnown`

	[2] `mapFolding.kitAST.RecipeJob.fromMapShape`

	"""
	lookupTotalFolds: dict[tuple[int, ...], int] = makeLookupTotalFoldsKnown()
	return lookupTotalFolds.get(tuple(mapShape))

def makeLookupTotalFoldsKnown() -> dict[tuple[int, ...], int]:
	"""Build a map-shape lookup from known map-folding sequence values.

	(AI generated docstring)

	You can use this function to inspect known folding totals for every map shape represented by
	`mapFolding.oeis._theSSOT.oeisIDsMapFoldingImplemented` [1]. This function retrieves known
	values with `mapFolding.oeis.getValuesKnown` [2] and converts sequence indices to dimensions
	with `makeMapShape` [3].

	Returns
	-------
	lookupTotalFolds : dict[tuple[int, ...], int]
		New dictionary mapping exact dimension tuples to known folding totals. When sequences
		produce the same shape, the later sequence in the registered tuple supplies the value.

	Exclusions
	----------
	Only registered map-folding sequences contribute values. `A007822`, which counts symmetric
	foldings, is excluded. This function does not filter retrieval sentinels before shape conversion.

	References
	----------
	[1] `mapFolding.oeis._theSSOT.oeisIDsMapFoldingImplemented`

	[2] `mapFolding.oeis.getValuesKnown`

	[3] `makeMapShape`

	"""
	return merge(*[keymap(partial(makeMapShape, oeisID), getValuesKnown(oeisID)) for oeisID in oeisIDsMapFoldingImplemented])

def makeMapShape(oeisID: LiteralString | Literal['A000136', 'A001415', 'A001416', 'A001417', 'A195646', 'A001418', 'A007822'], n: int) -> tuple[int, ...]:
	"""Convert a supported OEIS sequence identifier and index to map dimensions.

	You can use this function to obtain the map shape corresponding to `n` for one of the supported
	folding sequences. This function checks the offset in `mapFolding.oeis._metadata.dictionaryOEIS`
	[1], then returns the sequence's dimension tuple.

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

	References
	----------
	[1] `mapFolding.oeis._metadata.dictionaryOEIS`

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

def printEasyRunBenchmark(oeisID: str, n: int, computed: int, timeStart: float, *, ratio: int = 0) -> None:
	"""Print a benchmark comparison line for an OEIS sequence value.

	You can use this function to compare `computed` with the stored value at index `n` from
	`mapFolding.oeis.getValuesKnown` [1]. This function writes the index, a colored match flag,
	elapsed seconds, computed value, known value, and optional ratio to standard output.

	Parameters
	----------
	oeisID : str
		Sequence identifier, normalized by the known-value lookup.
	n : int
		Index of the sequence term to compare.
	computed : int
		The computed value to compare against the known OEIS value.
	timeStart : float
		Start time from `time.perf_counter` [2] for elapsed time calculation.
	ratio : int = 0
		Number of decimal places for the known-to-computed ratio. Zero hides the ratio.

	Output Fields
	-------------
	The index precedes a space; subsequent fields are separated by tabs. Matching values use
	`hunterMakesPy.ansiColor.GreenOnBlack` [3], and mismatches use `YellowOnRed`. An absent term
	uses `-errorL33T` as the known value. When requested, the ratio uses the match color if the
	float result is an integer and the mismatch color otherwise.

	Examples
	--------
	`easyRun/countFolds.py` prints a comparison after counting folds for a sequence term.

		```python
		from mapFolding.oeis import printEasyRunBenchmark

		printEasyRunBenchmark(oeisID, n, totalFolds, timeStart, ratio=0)
		```

	References
	----------
	[1] `mapFolding.oeis.getValuesKnown`

	[2] time.perf_counter - Python standard library.
		https://docs.python.org/3/library/time.html#time.perf_counter
	[3] hunterMakesPy - Official repository.
		https://github.com/hunterhogan/hunterMakesPy
	"""
	known: int = getValuesKnown(oeisID).get(n, -errorL33T)
	match: bool = computed == known
	sys.stdout.write(
		f"{n:2} "
		f"{(ansiColor.YellowOnRed, ansiColor.GreenOnBlack)[match]}{match}{ansiColorReset}\t"
		f"{time.perf_counter() - timeStart:5.2f}\t"
		f"{computed}\t{known}\t"
	)
	if ratio and computed:
		sys.stdout.write(formatBenchmarkRatio(known, computed, ratio) + '\t')
	sys.stdout.write(f"{ansiColorReset}\n")

def printEasyRunTriangle(oeisID: OEISid, n: int, computed: Sequence[int], timeStart: float, *, ratio: int = 0) -> None:
	"""Compare the computed triangle row `n` with known sequence cells.

	(AI generated docstring)

	You can use this function to inspect a row computation with its elapsed time. This function
	uses `mapFolding.oeis.getTriangleIndices` [1] to locate the complete row and
	`mapFolding.oeis.getValuesKnown` [2] to obtain the reference cells. The function prints the
	comparison and returns `None`.

	Parameters
	----------
	oeisID : OEISid
		Registered triangle sequence whose metadata defines the row layout.
	n : int
		Row number in the sequence's metadata layout.
	computed : Sequence[int]
		Computed cells in column order, starting with the first column.
	timeStart : float
		Starting value from `time.perf_counter` [3] for the elapsed-time display.
	ratio : int = 0
		Decimal places for known-to-computed ratios on a differing row. Zero hides ratios.

	Output Layout
	-------------
	The first line contains the row number, an exact-match flag, elapsed seconds, and `computed`.
	A match requires equal lengths and equal corresponding values. A differing row prints the
	known cells on a second aligned line. Missing positions in the shorter row display `'?'`.
	An unavailable reference value uses the negative `hunterMakesPy.errorL33T` sentinel [4].

	Ratios are printed only for differing rows and only for paired cells. Zero computed cells
	display `'-'`; integer ratios use the success color even when the ratio differs from
	one. This function does not calculate sequence terms or save results.

	See Also
	--------
	`printEasyRunBenchmark`
		Compare a single computed sequence term with its known value.

	References
	----------
	[1] `mapFolding.oeis.getTriangleIndices`

	[2] `mapFolding.oeis.getValuesKnown`

	[3] Python `time.perf_counter`.
		https://docs.python.org/3/library/time.html#time.perf_counter
	[4] hunterMakesPy.
		https://github.com/hunterhogan/hunterMakesPy
	"""
	#=SIN= Intermediate tuple: comparison, aligned output, and ratios consume the known cells separately.
	known: tuple[int, ...] = tuple(map(getValuesKnown(oeisID).get, getTriangleIndices(oeisID, n), repeat(-errorL33T)))
	match: bool = len(computed) == len(known) and all(map(eq, computed, known))
	timeElapsed: float = time.perf_counter() - timeStart

	sys.stdout.write(f"{n:2} {(ansiColor.YellowOnRed, ansiColor.GreenOnBlack)[match]}{match}{ansiColorReset}\t{timeElapsed:5.2f}\t")

	width: int = max(map(len, map(str, chain(computed, known))), default=1)
	sys.stdout.write(' '.join(map(str.rjust, chain(map(str, computed), repeat('?', max(0, len(known) - len(computed)))), repeat(width))))

	if not match:
		if ratio:
			sys.stdout.write('\t' + ' '.join(map(partial(formatBenchmarkRatio, ratio=ratio), known, computed)))
		sys.stdout.write('\n' + ' ' * len(f"{n:2} {match}\t{timeElapsed:5.2f}\t".expandtabs()))
		sys.stdout.write(' '.join(map(str.rjust, chain(map(str, known), repeat('?', max(0, len(computed) - len(known)))), repeat(width))))

	sys.stdout.write(f"{ansiColorReset}\n")

def printEasyRunHeader(title: str | int, subtitle: str | int) -> None:
	"""Print a title and subtitle with reproducible terminal colors.

	You can use this function to label a benchmark run with `title` and `subtitle`. This function
	writes the original labels to standard output and selects each color from the corresponding
	ASCII transliteration supplied by `anyascii` [1].

	Parameters
	----------
	title : str | int
		First label. Its ASCII transliteration must be accepted as a base-36 integer.
	subtitle : str | int
		Second label, subject to the same base-36 conversion requirement as `title`.

	Label Conversion
	----------------
	This function propagates `ValueError` if either label's ASCII transliteration cannot be
	converted to a base-36 integer.

	Color Selection
	---------------
	Each transliterated label is converted with `int` using base 36 [2]. The result modulo the
	number of entries in `hunterMakesPy.ansiColor` [3] selects the color. Labels can select the same
	color. This function resets terminal colors after printing both labels and a newline.

	Examples
	--------
	`easyRun/countFolds.py` labels each run with the sequence identifier and algorithm selector.

		```python
		from mapFolding.oeis import printEasyRunHeader

		printEasyRunHeader(oeisID, flow)
		```

	References
	----------
	[1] AnyAscii - Official repository.
		https://github.com/anyascii/anyascii
	[2] int - Python standard library.
		https://docs.python.org/3/library/functions.html#int
	[3] hunterMakesPy - Official repository.
		https://github.com/hunterhogan/hunterMakesPy
	"""
	sys.stdout.write(f"{ansiColor[int(anyascii.anyascii(str(title)), 36) % len(ansiColor)]}{title} ")
	sys.stdout.write(f"{ansiColor[int(anyascii.anyascii(str(subtitle)), 36) % len(ansiColor)]} {subtitle} ")
	sys.stdout.write(ansiColorReset + '\n')

if __name__ == "__main__":
	from mapFolding.oeis import getOEISids
	getOEISids()
