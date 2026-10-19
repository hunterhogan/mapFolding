from __future__ import annotations

from functools import partial
from humpy_cytoolz import compose, take
from hunterMakesPy import ansiColor, ansiColorReset, errorL33T
from itertools import filterfalse
from operator import methodcaller
from typing import cast as ILiterallyPromiseLiteralLiterallyMeansLiteral, LiteralString, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Mapping
	from mapFolding.theTypes import OEISid

def formatBFile(sequence: Mapping[int, int]) -> str:
	"""Format `sequence` as b-file records in ascending index order.

	(AI generated docstring)

	You can use this function to prepare integer sequence terms for OEIS b-file storage [1].
	Each record contains an index, a space, a value, and a newline. No header or comments are added.

	Parameters
	----------
	sequence : Mapping[int, int]
		Sequence indices mapped to integer values.

	Returns
	-------
	contents : str
		Sorted b-file records, or an empty string when no terms are supplied.

	References
	----------
	[1] OEIS b-files.
		https://oeis.org/wiki/B-files
	"""
	return ''.join(map(lambda term: f"{term[0]} {term[1]}\n", sorted(sequence.items())))  # ruff: ignore[unnecessary-map]

def formatBenchmarkRatio(known: int, computed: int, ratio: int) -> str:
	formatted: str = ''
	if ratio:
		formatted = '-'
		if computed:
			quotient: float = known / computed
			formatted = f"{(ansiColor.YellowOnRed, ansiColor.GreenOnBlack)[quotient.is_integer()]}{quotient:.{ratio}f}{ansiColorReset}"
	return formatted

def parse_bFile(oeisData: str) -> dict[int, int]:
	"""Parse OEIS b-file text into a dictionary of sequence indices and values.

	(AI generated docstring)

	You can use this function to convert downloaded or cached OEIS b-file text [1] into integer
	index-value pairs. This function returns the parsed dictionary, with a sentinel entry when
	`oeisData` is the empty string.

	Line Parsing
	------------
	Each line is stripped before empty lines and lines starting with `'#'` are removed. Indented
	comments and whitespace-only lines are therefore ignored.

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

	"""
	n_aOFn: dict[int, int] = {-errorL33T: -errorL33T}
	if oeisData:
		n_aOFn = dict(map(compose(tuple[int, int], partial(map, int), partial(take, 2))
					, map(methodcaller('split'), filterfalse(methodcaller('startswith', '#'), filter(None, map(str.strip, oeisData.splitlines())))
		)))
	return n_aOFn

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
