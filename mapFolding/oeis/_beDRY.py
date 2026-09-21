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
	# DOCUMENT
	n_aOFn: dict[int, int] = {-errorL33T: -errorL33T}
	if oeisData:
		n_aOFn = dict(map(compose(tuple[int, int], partial(map, int), partial(take, 2))
					, map(methodcaller('split'), filterfalse(methodcaller('startswith', '#'), filter(None, oeisData.strip().splitlines()))
		)))
	return n_aOFn

def parseTriangleBFile(contents: str, rowLengths: Iterable[int] | None = None, rowStart: int = 1) -> dict[int, list[int]]:
	sequence: dict[int, int] = parse_bFile(contents)
	return makeLookupTriangle(sequence.values(), rowLengths, rowStart)

def parseDiagonalBFile(contents: str, 次diagonal: int, *, rowStart: int = 1,
	rowLength: Callable[[int], int] | None = None, fromRight: bool = True) -> dict[int, int]:
	if rowLength is None:
		rowLength = partial(add, 1 - rowStart)
	return makeLookupDiagonal(parseTriangleBFile(contents, map(rowLength, count(rowStart)), rowStart),
		次diagonal, fromRight=fromRight, rowLength=rowLength)

def formatOEISid(oeisID: OEISid) -> OEISid:
	"""I use this to normalize OEIS sequence identifiers to a canonical form.

	This shared normalization function ensures consistent OEIS sequence ID formatting across all
	retrieval, lookup, and computation operations throughout the module. The function converts the
	identifier to uppercase and removes leading and trailing whitespace to ensure reliable dictionary
	lookups and cache key formation.

	Parameters
	----------
	oeisID : OEISid
		The OEIS sequence identifier to standardize.

	Returns
	-------
	oeisIDstandardized : OEISid
		Uppercase, alphanumeric OEIS ID with no whitespace.

	"""
	return ILiterallyPromiseLiteralLiterallyMeansLiteral("LiteralString", str(oeisID).upper().strip())
