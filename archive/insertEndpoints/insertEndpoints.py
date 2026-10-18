# ruff: file-ignore[p-print]
"""Construct overlap-free arch-diagram albums with Sade's 1949 insertion algorithm.

You can use this module to study Sade's constructive enumeration of overlap-free permutations [1].
Sade labels the endpoints from 1 through n, while `mapFolding` labels the same endpoints from 0 through n−1.
Every `Folding` in this module therefore begins with `endpointOrigin`, and each album preserves Sade's
right-to-left insertion order.

The construction relies on one invariant. A parent `Folding` has no arch crossings, so inserting
the newest `Leaf` can introduce a crossing only between the newest arch and an older arch on the
same side of the axis. The module imports `archesCross吗` from the shared crossing predicate [2].

Contents
--------
Functions
    countContributions
        Count the valid descendants contributed by one `Folding`.
    getAlbum
        Construct the album for one number of endpoints.
    makeAlbums
        Construct every album through one number of endpoints.
    makeDescendants
        Construct the valid descendants contributed by one `Folding`.
    deconstructAlbumToRow
        Count how many album members have each contribution total.

References
----------
[1] Albert Sadé (1949). Sur les Chevauchements des Permutations. Chez l’auteur, Marseille, France.
Available at https://oeis.org/A000108/a000108_17.pdf.

[2] `mapFolding._e.algorithms.iff.creaseViolation吗`
"""

from __future__ import annotations

from collections import Counter
from functools import partial
from itertools import chain
from pprint import pprint
from typing import TYPE_CHECKING

from hunterMakesPy import decreasing, inclusive
from mapFolding import endpointOrigin, positionOrigin
from mapFolding._e.algorithms.iff import creaseViolation吗 as archesCross吗
from mapFolding._e.algorithms.iff import foldingValid吗 as diagramValid吗

if TYPE_CHECKING:
	from collections.abc import Collection

	from mapFolding.theTypes import Folding, Leaf, Pile

def getAlbum(totalEndpoints: int) -> tuple[Folding, ...]:
	"""Construct the Sade album for `totalEndpoints`.

	(AI generated docstring)

	You can use this function when only the final generation of Sade's construction is
	relevant. The function still constructs each preceding album because every album derives
	from the preceding album.

	Parameters
	----------
	totalEndpoints : int
		The number of endpoints in every returned `Folding`. `totalEndpoints` must be at least 2.

	Returns
	-------
	diagramAlbum : tuple[Folding, ...]
		The overlap-free permutations in Sade's album order.
	"""
	return makeAlbums(totalEndpoints)[totalEndpoints]

def makeAlbums(totalEndpoints: int) -> dict[int, tuple[Folding, ...]]:
	"""Construct every Sade album through `totalEndpoints`.

	(AI generated docstring)

	You can use this function to inspect every generation of Sade's construction. Album 2
	contains only `(0, 1)`. Each later album contains all valid descendants of the preceding
	album, with both parent order and right-to-left gap order preserved.

	Parameters
	----------
	totalEndpoints : int
		The inclusive number of endpoints for the final album. `totalEndpoints` must be at least 2.

	Returns
	-------
	sadeAlbums : dict[int, tuple[Folding, ...]]
		Each number of endpoints mapped to the corresponding ordered album.

	Raises
	------
	ValueError
		If `totalEndpoints` is less than 2.

	References
	----------
	[1] Albert Sade (1949). Sur les Chevauchements des Permutations, sections 4–7.
		Marseille, France.
	"""
	if totalEndpoints < 2:
		message: str = f'I received `{totalEndpoints = }`, but a Sade album needs at least two endpoints.'
		raise ValueError(message)

	endpointsActive: int = 2
	sadeAlbums: dict[int, tuple[Folding, ...]] = {endpointsActive: ((endpointOrigin, 1),)}
	while endpointsActive < totalEndpoints:
		sadeAlbums[endpointsActive + 1] = tuple(
			chain.from_iterable(map(_makeDescendants, sadeAlbums[endpointsActive]))
		)
		endpointsActive += 1
	return sadeAlbums

def _makeDescendants(diagram: Folding) -> tuple[Folding, ...]:
	return tuple(filter(_diagramValid吗, map(partial(_insertEndpointLast, diagram), range(len(diagram), positionOrigin, decreasing))))

def _insertEndpointLast(diagram: Folding, position: Pile) -> Folding:
	return (*diagram[:position], len(diagram), *diagram[position:])

def _diagramValid吗(diagram: Folding) -> bool:
	return not _newArchCrosses吗(diagram)

def _newArchCrosses吗(diagram: Folding) -> bool:
	lookupPosition: dict[Leaf, Pile] = dict(zip(diagram, range(len(diagram)), strict=True))
	endpointArchLast: Leaf = len(diagram) - 2
	return any(map(partial(_archesCross吗, lookupPosition, endpointArchLast), range(endpointArchLast - 2, endpointOrigin - inclusive, 2 * decreasing)))

def _archesCross吗(lookupPosition: dict[Leaf, Pile], endpoint: Leaf, endpointComparand: Leaf) -> bool:
	archesSorted: list[tuple[Pile, Pile]] = sorted((
				(lookupPosition[endpoint], lookupPosition[endpoint + 1])
				, (lookupPosition[endpointComparand], lookupPosition[endpointComparand + 1])
			))
	return archesCross吗(archesSorted[0][0], archesSorted[1][0], archesSorted[0][1], archesSorted[1][1])

#================== Deconstruct Album to Row ======================================================

def deconstructAlbumToRow(album: Collection[Folding]) -> dict[int, int]:
	"""Count how many album members have each Sade contribution total.

	(AI generated docstring)

	You can use this function to compute one row of Sade's V(n, i) triangle. Each key is a
	contribution total i, and each value counts the album members that produce exactly i valid
	descendants.

	Parameters
	----------
	album : Collection[Folding]
		A nonempty collection of normalized, overlap-free `Folding` values with one common number of
		endpoints.

	Returns
	-------
	contributionRow : dict[int, int]
		Each contribution total mapped to its number of occurrences, ordered by contribution total.

	Raises
	------
	ValueError
		If `album` is empty or contains different diagram lengths.

	References
	----------
	[1] Albert Sade (1949). Sur les Chevauchements des Permutations, sections 3–7.
		Marseille, France.
	"""
	diagramAlbum: tuple[Folding, ...] = tuple(album)
	if not diagramAlbum:
		message: str = f'I received `{diagramAlbum = }`, but a Sade contribution row needs at least one `Folding`.'
		raise ValueError(message)

	endpointCounts: tuple[int, ...] = tuple(map(len, diagramAlbum))
	if len(frozenset(endpointCounts)) != 1:
		message = f'I received `{endpointCounts = }`, but every `Folding` in one Sade album must have the same length.'
		raise ValueError(message)

	return dict(sorted(Counter(map(countContributions, diagramAlbum)).items()))

def countContributions(diagram: Folding) -> int:
	"""Count the valid Sade descendants contributed by one `Folding`.

	(AI generated docstring)

	You can use this function to obtain Sade's contribution value i for one member of an album. The
	contribution value is the number of gaps that accept the newest `Leaf` without creating an arch
	crossing.

	Parameters
	----------
	diagram : Folding
		An overlap-free permutation of consecutive `Leaf` values that begins with `endpointOrigin`.

	Returns
	-------
	contributionsTotal : int
		The number of overlap-free descendants produced from `diagram`.
	"""
	return len(makeDescendants(diagram))

def makeDescendants(diagram: Folding) -> tuple[Folding, ...]:
	"""Construct the valid Sade descendants contributed by one `Folding`.

	(AI generated docstring)

	You can use this function to inspect one step of Sade's construction. The function inserts the
	newest `Leaf` into each gap after `endpointOrigin`, starting with the rightmost gap, and retains each
	child whose newest arch does not cross an older arch of matching parity [1].

	Parameters
	----------
	diagram : Folding
		An overlap-free permutation of consecutive `Leaf` values that begins with `endpointOrigin`.

	Returns
	-------
	diagramDescendants : tuple[Folding, ...]
		The overlap-free descendants in Sade's right-to-left insertion order.

	References
	----------
	[1] Albert Sade (1949). Sur les Chevauchements des Permutations, sections 4–7.
		Marseille, France.
	"""
	_validateDiagram(diagram)
	return _makeDescendants(diagram)

def _validateDiagram(diagram: Folding) -> None:
	if len(diagram) < 2:
		message: str = f'I received `{diagram = }`, but a Sade album `Folding` must contain at least two endpoints.'
		raise ValueError(message)

	if diagram[positionOrigin] != endpointOrigin:
		message = f'I received `{diagram = }`, but a Sade album `Folding` must begin with `{endpointOrigin = }`.'
		raise ValueError(message)

	if frozenset(diagram) != frozenset(range(len(diagram))):
		message = f'I received `{diagram = }`, but I expected each `Leaf` from 0 through {len(diagram) - 1}.'
		raise ValueError(message)

	if not diagramValid吗(diagram, (len(diagram),)):
		message = f'I received `{diagram = }`, but a Sade album `Folding` cannot contain an arch crossing.'
		raise ValueError(message)

if __name__ == '__main__':
	totalEndpoints: int = 6
	album: dict[int, tuple[Folding, ...]] = makeAlbums(totalEndpoints)
	pprint(album, width=120, compact=True)
	pprint(deconstructAlbumToRow(album[totalEndpoints]), width=120, compact=True)
