# ruff: file-ignore[p-print]
"""Construct overlap-free folding albums with Sade's 1949 insertion algorithm.

You can use this module to study Sade's constructive enumeration of overlap-free permutations [1].
Sade labels the leaves from 1 through n, while `mapFolding` labels the same leaves from 0 through n−1.
Every `Folding` in this module therefore begins with `leafOrigin`, and each album preserves Sade's
right-to-left insertion order.

The construction relies on one invariant. A parent `Folding` has no crease violations, so inserting
the newest `Leaf` can introduce a violation only between the newest crease and an older crease of
matching parity. The module delegates that comparison to `mapFolding._e.algorithms.iff.creaseViolation
吗` [2].

Contents
--------
Functions
    countFoldingContributions
        Count the valid descendants contributed by one `Folding`.
    getAlbum
        Construct the album for one number of leaves.
    makeAlbums
        Construct every album through one number of leaves.
    makeDescendants
        Construct the valid descendants contributed by one `Folding`.
    makeSadeContributionRow
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
from hunterMakesPy import decreasing, inclusive
from itertools import chain
from mapFolding._e import leafOrigin, pileOrigin
from mapFolding._e.algorithms.iff import creaseViolation吗, foldingValid吗
from pprint import pprint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Collection
	from mapFolding._e.theTypes import Folding, Leaf, Pile

def getAlbum(totalLeaves: int) -> tuple[Folding, ...]:
	"""Construct the Sade album for `totalLeaves`.

	(AI generated docstring)

	You can use this function when only the final generation of Sade's construction is
	relevant. The function still constructs each preceding album because every album derives
	from the preceding album.

	Parameters
	----------
	totalLeaves : int
		The number of leaves in every returned `Folding`. `totalLeaves` must be at least 2.

	Returns
	-------
	foldingAlbum : tuple[Folding, ...]
		The overlap-free permutations in Sade's album order.
	"""
	return makeAlbums(totalLeaves)[totalLeaves]

def makeAlbums(totalLeaves: int) -> dict[int, tuple[Folding, ...]]:
	"""Construct every Sade album through `totalLeaves`.

	(AI generated docstring)

	You can use this function to inspect every generation of Sade's construction. Album 2
	contains only `(0, 1)`. Each later album contains all valid descendants of the preceding
	album, with both parent order and right-to-left gap order preserved.

	Parameters
	----------
	totalLeaves : int
		The inclusive number of leaves for the final album. `totalLeaves` must be at least 2.

	Returns
	-------
	dictionarySadeAlbums : dict[int, tuple[Folding, ...]]
		Each number of leaves mapped to the corresponding ordered album.

	Raises
	------
	ValueError
		If `totalLeaves` is less than 2.

	References
	----------
	[1] Albert Sade (1949). Sur les Chevauchements des Permutations, sections 4–7.
		Marseille, France.
	"""
	if totalLeaves < 2:
		message: str = f'I received `{totalLeaves = }`, but a Sade album needs at least two leaves.'
		raise ValueError(message)

	totalLeavesActive: int = 2
	dictionarySadeAlbums: dict[int, tuple[Folding, ...]] = {totalLeavesActive: ((leafOrigin, 1),)}
	while totalLeavesActive < totalLeaves:
		dictionarySadeAlbums[totalLeavesActive + 1] = tuple(
			chain.from_iterable(map(_makeDescendants, dictionarySadeAlbums[totalLeavesActive]))
		)
		totalLeavesActive += 1
	return dictionarySadeAlbums

def _makeDescendants(folding: Folding) -> tuple[Folding, ...]:
	return tuple(filter(_foldingValid吗, map(partial(_insertLeafLastAtPile, folding), range(len(folding), pileOrigin, decreasing))))

def _insertLeafLastAtPile(folding: Folding, pile: Pile) -> Folding:
	return (*folding[:pile], len(folding), *folding[pile:])

def _foldingValid吗(folding: Folding) -> bool:
	return not _leafLastHasCreaseViolation吗(folding)

def _leafLastHasCreaseViolation吗(folding: Folding) -> bool:
	dictionaryLeafPile: dict[Leaf, Pile] = dict(zip(folding, range(len(folding)), strict=True))
	leafLastCreaseAnte: Leaf = len(folding) - 2
	return any(map(partial(_creaseViolation吗, dictionaryLeafPile, leafLastCreaseAnte), range(leafLastCreaseAnte - 2, leafOrigin - inclusive, 2 * decreasing)))

def _creaseViolation吗(dictionaryLeafPile: dict[Leaf, Pile], leaf: Leaf, leafComparand: Leaf) -> bool:
	creasesPileSorted: list[tuple[Pile, Pile]] = sorted((
				(dictionaryLeafPile[leaf], dictionaryLeafPile[leaf + 1])
				, (dictionaryLeafPile[leafComparand], dictionaryLeafPile[leafComparand + 1])
			))
	return creaseViolation吗(creasesPileSorted[0][0], creasesPileSorted[1][0], creasesPileSorted[0][1], creasesPileSorted[1][1])

#================== Deconstruct Album to Row ======================================================

def deconstructAlbumToRow(albumFolding: Collection[Folding]) -> dict[int, int]:
	"""Count how many album members have each Sade contribution total.

	(AI generated docstring)

	You can use this function to compute one row of Sade's V(n, i) triangle. Each key is a
	contribution total i, and each value counts the album members that produce exactly i valid
	descendants.

	Parameters
	----------
	albumFolding : Collection[Folding]
		A nonempty collection of normalized, overlap-free `Folding` values with one common number of
		leaves.

	Returns
	-------
	dictionarySadeContributionRow : dict[int, int]
		Each contribution total mapped to its number of occurrences, ordered by contribution total.

	Raises
	------
	ValueError
		If `albumFolding` is empty or contains different folding lengths.

	References
	----------
	[1] Albert Sade (1949). Sur les Chevauchements des Permutations, sections 3–7.
		Marseille, France.
	"""
	tupleFoldingAlbum: tuple[Folding, ...] = tuple(albumFolding)
	if not tupleFoldingAlbum:
		message: str = f'I received `{tupleFoldingAlbum = }`, but a Sade contribution row needs at least one `Folding`.'
		raise ValueError(message)

	tupleTotalLeavesByFolding: tuple[int, ...] = tuple(map(len, tupleFoldingAlbum))
	if len(frozenset(tupleTotalLeavesByFolding)) != 1:
		message = f'I received `{tupleTotalLeavesByFolding = }`, but every `Folding` in one Sade album must have the same length.'
		raise ValueError(message)

	return dict(sorted(Counter(map(countLeafContributions, tupleFoldingAlbum)).items()))

def countLeafContributions(folding: Folding) -> int:
	"""Count the valid Sade descendants contributed by one `Folding`.

	(AI generated docstring)

	You can use this function to obtain Sade's contribution value i for one member of an album. The
	contribution value is the number of gaps that accept the newest `Leaf` without creating a crease
	violation.

	Parameters
	----------
	folding : Folding
		An overlap-free permutation of consecutive `Leaf` values that begins with `leafOrigin`.

	Returns
	-------
	contributionsTotal : int
		The number of overlap-free descendants produced from `folding`.
	"""
	return len(makeDescendants(folding))

def makeDescendants(folding: Folding) -> tuple[Folding, ...]:
	"""Construct the valid Sade descendants contributed by one `Folding`.

	(AI generated docstring)

	You can use this function to inspect one step of Sade's construction. The function inserts the
	newest `Leaf` into each gap after `leafOrigin`, starting with the rightmost gap, and retains each
	child whose newest crease does not cross an older crease of matching parity [1].

	Parameters
	----------
	folding : Folding
		An overlap-free permutation of consecutive `Leaf` values that begins with `leafOrigin`.

	Returns
	-------
	foldingDescendants : tuple[Folding, ...]
		The overlap-free descendants in Sade's right-to-left insertion order.

	References
	----------
	[1] Albert Sade (1949). Sur les Chevauchements des Permutations, sections 4–7.
		Marseille, France.
	"""
	_validateFolding(folding)
	return _makeDescendants(folding)

def _validateFolding(folding: Folding) -> None:
	if len(folding) < 2:
		message: str = f'I received `{folding = }`, but a Sade album `Folding` must contain at least two leaves.'
		raise ValueError(message)

	if folding[pileOrigin] != leafOrigin:
		message = f'I received `{folding = }`, but a Sade album `Folding` must begin with `{leafOrigin = }`.'
		raise ValueError(message)

	if frozenset(folding) != frozenset(range(len(folding))):
		message = f'I received `{folding = }`, but I expected each `Leaf` from 0 through {len(folding) - 1}.'
		raise ValueError(message)

	if not foldingValid吗(folding, (len(folding),)):
		message = f'I received `{folding = }`, but a Sade album `Folding` cannot contain a crease violation.'
		raise ValueError(message)

if __name__ == '__main__':
	totalLeaves: int = 6
	albumFolding: dict[int, tuple[Folding, ...]] = makeAlbums(totalLeaves)
	pprint(albumFolding, width=120, compact=True)
	pprint(deconstructAlbumToRow(albumFolding[totalLeaves]), width=120, compact=True)
