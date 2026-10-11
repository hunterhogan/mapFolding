from __future__ import annotations

from functools import partial
from humpy_cytoolz import compose, take
from hunterMakesPy import errorL33T
from itertools import count, filterfalse
from mapFolding.dataStructures import makeLookupDiagonal, makeLookupTriangle
from operator import add, methodcaller
from typing import cast as ILiterallyPromiseLiteralLiterallyMeansLiteral, LiteralString, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping
	from mapFolding.theTypes import OEISid

# TODO Remove all of this duplicate code, or more likely, dig into the commit history of the functions
# displaced by this crap, get the superior code, and replace this.
def formatBFile(sequence: Mapping[int, int]) -> str:
	return ''.join(map(lambda term: f"{term[0]} {term[1]}\n", sorted(sequence.items())))  # ruff: ignore[unnecessary-map]

def parse_bFile(oeisData: str) -> dict[int, int]:
	"""Parse OEIS b-file text into a dictionary of sequence indices and values.

	(AI generated docstring)

	You can use this function to convert downloaded or cached OEIS b-file text [1] into integer
	index-value pairs. This function returns the parsed dictionary, with a sentinel entry when
	`oeisData` is the empty string.

	Line Parsing
	------------
	For nonempty `oeisData`, the function strips whitespace around the complete input, splits the
	input into lines, and discards empty lines and lines starting with `'#'`. Individual lines are
	not stripped before comment detection. A comment with leading whitespace inside the input is
	therefore treated as data. Likewise, an interior line containing only whitespace is not an
	empty line and fails parsing.

	Each remaining line is split on whitespace [2]. Only the first two fields are converted to
	integers; additional fields are ignored. Repeated indices keep the last value at the first
	occurrence's dictionary position [3]. The function does not sort indices or check that indices
	are consecutive.

	Parameters
	----------
	oeisData : str
		Text containing integer index-value pairs, or the empty string when no text is available.

	Returns
	-------
	sequence : dict[int, int]
		Parsed integer pairs. The empty string returns `{-errorL33T: -errorL33T}`, using
		`hunterMakesPy.errorL33T` [4]. Nonempty text that leaves no data lines after filtering,
		including whitespace-only input or recognized comment-only input, returns an empty
		dictionary instead.

	Malformed Lines
	---------------
	This function propagates `ValueError` from dictionary construction if a retained line has
	fewer than two fields, or from integer conversion if either of the first two fields cannot
	be converted to an integer.

	Examples
	--------
	`parseTriangleBFile` [5] parses its incoming `contents` before grouping the sequence into rows.

		```python
		from mapFolding.oeis._beDRY import parse_bFile

		sequence: dict[int, int] = parse_bFile(contents)
		```

	References
	----------
	[1] B-files - OEIS Wiki.
		https://oeis.org/wiki/B-files
	[2] str.split - Python standard library.
		https://docs.python.org/3/library/stdtypes.html#str.split
	[3] Dictionary insertion order - Python standard library.
		https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
	[4] hunterMakesPy - Official repository.
		https://github.com/hunterhogan/hunterMakesPy
	[5] `parseTriangleBFile`

	"""
	n_aOFn: dict[int, int] = {-errorL33T: -errorL33T}
	if oeisData:
		n_aOFn = dict(map(compose(tuple[int, int], partial(map, int), partial(take, 2))
					, map(methodcaller('split'), filterfalse(methodcaller('startswith', '#'), filter(None, oeisData.strip().splitlines()))
		)))
	return n_aOFn

def parseTriangleBFile(contents: str, rowLengths: Iterable[int] | None = None, rowStart: int = 1) -> dict[int, list[int]]:
	"""Group OEIS b-file values into consecutively numbered rows.

	(AI generated docstring)

	You can use this function to recover rows from b-file text. This function parses `contents`
	with `parse_bFile` [1] and groups the parsed values with
	`mapFolding.dataStructures.makeLookupTriangle` [2].

	Parameters
	----------
	contents : str
		B-file text containing sequence indices and values.
	rowLengths : Iterable[int] | None = None
		Requested lengths for successive rows. `None` or an empty container selects lengths
		starting at one and increasing by one; an empty iterator produces no rows.
	rowStart : int = 1
		Number assigned to the first row. This number does not change default row lengths.

	Returns
	-------
	triangle : dict[int, list[int]]
		Nonempty rows, including a partially known final row. Values follow parsed dictionary
		insertion order; sequence index keys do not determine row boundaries.

	Row Consumption
	---------------
	Grouping stops at the first empty row or when requested row lengths are exhausted. Empty
	`contents` retains the sentinel value produced by `parse_bFile` in the first default row.

	References
	----------
	[1] `parse_bFile`

	[2] `mapFolding.dataStructures.makeLookupTriangle`

	"""
	sequence: dict[int, int] = parse_bFile(contents)
	return makeLookupTriangle(sequence.values(), rowLengths, rowStart)

def parseDiagonalBFile(contents: str, 次diagonal: int, *, rowStart: int = 1,
	rowLength: Callable[[int], int] | None = None, fromRight: bool = True) -> dict[int, int]:
	"""Select a diagonal from OEIS b-file values grouped into rows.

	(AI generated docstring)

	You can use this function to read one diagonal from b-file text. This function groups values
	with `parseTriangleBFile` [1], then selects available columns with
	`mapFolding.dataStructures.makeLookupDiagonal` [2].

	Parameters
	----------
	contents : str
		B-file text containing sequence indices and values in row order.
	次diagonal : int
		Distance from the selected row edge, normally starting at one. Zero is accepted.
	rowStart : int = 1
		Number assigned to the first row.
	rowLength : Callable[[int], int] | None = None
		Expected complete row length from a row number. `None` selects lengths starting at one
		and increasing by one. The callable controls grouping for either edge and supplies the
		right edge when `fromRight=True`.
	fromRight : bool = True
		Select from the right edge when `True`, or the left edge when `False`.

	Returns
	-------
	diagonal : dict[int, int]
		Available diagonal values keyed by ascending row number. Rows without the requested
		column are omitted; a partial row uses the expected length to locate its right edge.

	Invalid Requests
	----------------
	This function propagates `ValueError` if `次diagonal` is negative or retained b-file fields
	cannot be parsed as integer pairs.

	References
	----------
	[1] `parseTriangleBFile`

	[2] `mapFolding.dataStructures.makeLookupDiagonal`

	"""
	if rowLength is None:
		rowLength = partial(add, 1 - rowStart)
	return makeLookupDiagonal(parseTriangleBFile(contents, map(rowLength, count(rowStart)), rowStart),
		次diagonal, fromRight=fromRight, rowLength=rowLength)

def formatOEISid(oeisID: OEISid) -> OEISid:
	"""Uppercase an OEIS sequence identifier and strip surrounding whitespace.

	You can use this function to standardize an identifier before a sequence lookup. This function
	converts `oeisID` to text, uppercases the text, and removes leading and trailing whitespace.
	This function does not remove internal whitespace or validate the identifier's syntax.

	Parameters
	----------
	oeisID : OEISid
		The OEIS sequence identifier to standardize.

	Returns
	-------
	oeisIDstandardized : OEISid
		Uppercase text with no surrounding whitespace.

	"""
	return ILiterallyPromiseLiteralLiterallyMeansLiteral("LiteralString", str(oeisID).upper().strip())
