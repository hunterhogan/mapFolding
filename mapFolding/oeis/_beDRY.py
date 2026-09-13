from __future__ import annotations

from functools import partial
from itertools import count
from mapFolding.beDRY import makeLookupDiagonal, makeLookupTriangle
from operator import add, itemgetter
from typing import cast as ILiterallyPromiseLiteralLiterallyMeansLiteral, LiteralString, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping
	from mapFolding.theTypes import OEISid

# TODO Remove all of this duplicate code.
def formatBFile(sequence: Mapping[int, int]) -> str:
	return ''.join(map(lambda term: f"{term[0]} {term[1]}\n", sorted(sequence.items())))

def parseBFile(contents: str) -> dict[int, int]:
	records: tuple[tuple[int, ...], ...] = tuple(map(lambda line: tuple(map(int, line.split())),
		filter(bool, map(lambda line: line.partition('#')[0].strip(), contents.splitlines()))))
	if not all(map(lambda record: len(record) == 2, records)):
		message: str = "I received a b-file record without exactly two integers: an index and a value."
		raise ValueError(message)
	sequence: dict[int, int] = dict(map(itemgetter(0, 1), records))
	if len(sequence) != len(records):
		message = "I received duplicate indices in the b-file."
		raise ValueError(message)
	return dict(sorted(sequence.items()))

def parseTriangleBFile(contents: str, rowLengths: Iterable[int] | None = None, rowStart: int = 1) -> dict[int, list[int]]:
	sequence: dict[int, int] = parseBFile(contents)
	if sequence and tuple(sequence) != tuple(range(min(sequence), max(sequence) + 1)):
		message: str = "I received gaps in the b-file indices, so I cannot locate the triangle rows."
		raise ValueError(message)
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
