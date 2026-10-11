"""Inspect known OEIS values, map shapes, and triangle data.

(AI generated docstring)

You can use this module's functions to look up known folding totals, convert sequence indices
to map shapes, inspect triangle rows and diagonals, and print benchmark comparisons.

Contents
--------
Functions
	getTotalFoldsKnown
		Retrieve a known folding total for an exact map shape.
	getTriangleDiagonal
		Select a diagonal using the sequence's row layout.
	getTriangleRows
		Group known sequence values into numbered rows.
	makeLookupTotalFoldsKnown
		Build a map-shape lookup from registered map-folding sequences.
	makeMapShape
		Convert a supported sequence identifier and index to map dimensions.
	printEasyRunBenchmark
		Print computed and known values with elapsed time and optional ratio.
	printEasyRunHeader
		Print two labels with reproducible terminal colors.
	readBFileDiagonal
		Read a local b-file and select a diagonal.
	readBFileTriangle
		Read a local b-file and group its values into rows.

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

from functools import cache, partial
from humpy_cytoolz import keymap, merge
from hunterMakesPy import ansiColor, ansiColorReset, errorL33T
from itertools import count
from mapFolding.dataStructures import makeLookupDiagonal, makeLookupTriangle
from mapFolding.kitFilesystem import readText
from mapFolding.oeis import getMetadata, getValuesKnown
from mapFolding.oeis._beDRY import parseDiagonalBFile, parseTriangleBFile
from mapFolding.oeis._metadata import dictionaryOEIS
from mapFolding.oeis._theSSOT import oeisIDsMapFoldingImplemented
from more_itertools import loops
from typing import TYPE_CHECKING
import anyascii
import sys
import time

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding.oeis._dataBaskets import MetadataOEISid
	from mapFolding.theTypes import OEISid, 形Triangle
	from pathlib import Path
	from typing import Literal, LiteralString

@cache
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

	Caching
	-------
	`functools.cache` [2] retains results for previously requested shapes. Each uncached call
	rebuilds the complete lookup before selecting the total for `mapShape`.

	Examples
	--------
	`mapFolding.kitAST.RecipeJob.fromMapShape` [3] queries the known total before estimating a
	job's size.

		```python
		from mapFolding.oeis import getTotalFoldsKnown

		getTotalFoldsKnown(state.mapShape)
		```

	References
	----------
	[1] `makeLookupTotalFoldsKnown`

	[2] functools.cache - Python standard library.
		https://docs.python.org/3/library/functools.html#functools.cache
	[3] `mapFolding.kitAST.RecipeJob.fromMapShape`

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

def getTriangleRows(oeisID: OEISid) -> 形Triangle:
	"""Group known values for `oeisID` into numbered triangle rows.

	(AI generated docstring)

	You can use this function to access a supported sequence as rows for further analysis. This
	function obtains the sequence metadata through `mapFolding.oeis.getMetadata` [1] and returns
	a new dictionary mapping row numbers to lists of known values.

	Row Boundaries
	--------------
	`mapFolding.oeis._dataBaskets.MetadataOEISid` [2] supplies the starting row through `rowStart`,
	falling back to `offset`. When metadata supplies `rowLength`, the function evaluates `rowLength`
	for successive row numbers. Otherwise, requested row lengths begin at one and increase by one,
	regardless of the starting row number.

	`mapFolding.dataStructures.makeLookupTriangle` [3] groups `valuesKnown.values()` in dictionary
	insertion order. Sequence index keys do not determine row boundaries and are not sorted. A
	partially known final row is retained. Grouping stops at the first empty row, including an empty
	row caused by a requested length of zero. The function does not check whether the sequence
	represents a triangle.

	Parameters
	----------
	oeisID : OEISid
		Identifier of a sequence present in the package metadata. Lookup ignores letter case and
		leading or trailing whitespace.

	Returns
	-------
	triangleRows : 形Triangle
		Consecutive row numbers mapped to newly allocated lists of known sequence values.

	Unknown Sequences
	-----------------
	This function propagates `KeyError` from the metadata lookup if the normalized `oeisID` is
	absent from the package metadata.

	Examples
	--------
	`getTriangleDiagonal` [4] obtains rows with the following call before selecting a diagonal,
	using the caller's `oeisID`.

		```python
		from mapFolding.oeis import getTriangleRows

		getTriangleRows(oeisID)
		```

	References
	----------
	[1] `mapFolding.oeis.getMetadata`

	[2] `mapFolding.oeis._dataBaskets.MetadataOEISid`

	[3] `mapFolding.dataStructures.makeLookupTriangle`

	[4] `getTriangleDiagonal`

	"""
	metadata: MetadataOEISid = getMetadata(oeisID)
	rowStart: int = metadata.get('rowStart', metadata['offset'])
	rowLengths: Iterable[int] | None = None
	if 'rowLength' in metadata:
		rowLengths = map(metadata['rowLength'], count(rowStart))
	return makeLookupTriangle(metadata['valuesKnown'].values(), rowLengths, rowStart)

# TODO
def getTriangleDiagonal(oeisID: OEISid, 次diagonal: int, *, fromRight: bool = True) -> dict[int, int]:  # ruff: ignore[undocumented-public-function]
	metadata: MetadataOEISid = getMetadata(oeisID)
	return makeLookupDiagonal(getTriangleRows(oeisID), 次diagonal, fromRight=fromRight,
		rowLength=metadata.get('rowLength', lambda rowNumber: rowNumber - metadata.get('rowStart', metadata['offset']) + 1))

# TODO remove
def readBFileTriangle(pathFilename: Path, rowLengths: Iterable[int] | None = None, rowStart: int = 1) -> 形Triangle:  # ruff: ignore[undocumented-public-function]
	return parseTriangleBFile(readText(pathFilename), rowLengths, rowStart)

# TODO remove
def readBFileDiagonal(pathFilename: Path, 次diagonal: int, *, rowStart: int = 1,  # ruff: ignore[undocumented-public-function]
	rowLength: Callable[[int], int] | None = None, fromRight: bool = True) -> dict[int, int]:
	return parseDiagonalBFile(readText(pathFilename), 次diagonal, rowStart=rowStart, rowLength=rowLength, fromRight=fromRight)

def printEasyRunBenchmark(oeisID: str, n: int, computed: int, timeStart: float, *, ratio: bool = False) -> None:
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
	ratio : bool = False
		Print the known-to-computed ratio when `True` and `computed` is nonzero.

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

		printEasyRunBenchmark(oeisID, n, totalFolds, timeStart, ratio=False)
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
		integer = (known / computed).is_integer()
		sys.stdout.write(f"{(ansiColor.YellowOnRed, ansiColor.GreenOnBlack)[integer]}{known / computed}{ansiColorReset}\t")
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
