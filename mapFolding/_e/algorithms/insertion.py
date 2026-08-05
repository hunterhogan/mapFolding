# ruff: file-ignore[commented-out-code, p-print, print]
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""

from __future__ import annotations

from functools import partial
from hunterMakesPy import decreasing, inclusive, zeroIndexed
from itertools import chain
from mapFolding._e import leafOrigin, pileOrigin
from mapFolding._e.algorithms.iff import creaseViolation吗
from mapFolding.beDRY import getLeavesTotal
from mapFolding.kitFilesystem import makePathFilenameFolds
from mapFolding.oeis import getMapShape, getValuesKnown
from mapFolding.theSSOT import settingsPackage
from pprint import pprint
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Collection, Iterable
	from mapFolding._e.theTypes import Folding, Leaf, Pile, PinnedLeaves
	from pathlib import Path

# DEVELOPMENT Major overhaul
# ⌧ `Folding` -> `PinnedLeaves`. This requires overwriting keys or reading/changing a ton of keys.
# ⌧ creaseAnte -> creasePost. ante makes it easier to count backwards from the last leaf.
# `leavesTotal` -> a function for incrementing the map size.
# 	E.g., getMapNext: Callable[[int], tuple[int, ...]] = partial(getMapShape, 'A001417')
# Don't create all `Folding`: create 1/2, Theorem 2, or 1/d!, Theorem 4. See mapFolding._e.algorithms.elimination theorem2b and theorem4.
# Check each crease as it is added. A violation will invalidate an entire branch. As opposed to
#   building a complete set of new `Folding` before checking any creases. At the moment, I think
#   no algorithm is _this_ efficient. "Lazy" construction allows many steps of each algorithm to be
#   efficient, but I never tried to achieve this level of efficiency.
# Save and load albums. csv of Leaf in Pile order, with header row. See, e.g., mapFolding/_e/_development/dataRaw/p2d4.csv
# 	pathAlbum = settingsPackage.pathPackage / "mapFolding/_e/_development" / "albums"
# Concurrency.

def getAlbum(leavesTotal: int) -> tuple[Folding, ...]:
	"""Get the Sade album for `leavesTotal`."""
	return makeAlbums(leavesTotal)[leavesTotal]

# def makeAlbums(getMapNext: Callable[[int], tuple[int, ...]], n: int, nFinal: int, pathAlbum: Path):
def makeAlbums(leavesTotal: int) -> dict[int, tuple[Folding, ...]]:
	"""Construct every Sade album through `leavesTotal`.

	(AI generated docstring)

	You can use this function to inspect every generation of Sade's construction. Album 2
	contains only `(0, 1)`. Each later album contains all valid descendants of the preceding
	album, with both parent order and right-to-left gap order preserved.

	Parameters
	----------
	leavesTotal : int
		The inclusive number of leaves for the final album. `leavesTotal` must be at least 2.

	Returns
	-------
	dictionarySadeAlbums : dict[int, tuple[Folding, ...]]
		Each number of leaves mapped to the corresponding ordered album.

	References
	----------
	[1] Albert Sade (1949). Sur les Chevauchements des Permutations, sections 4–7.
		Marseille, France.
	"""
	# mapShape: tuple[int, ...] = getMapNext(n)
	# pathFilenameAlbum: Path = makePathFilenameFoldsTotal(mapShape, pathAlbum, suffix='.album')

	# leavesTotal: int = getLeavesTotal(mapShape)

	leavesTotalActive: int = 2
	# DEVELOPMENT Just have two albums in memory: the known album, and the descendant album.
	dictionarySadeAlbums: dict[int, tuple[Folding, ...]] = {leavesTotalActive: ((leafOrigin, 1),)}
	while leavesTotalActive < leavesTotal:
		dictionarySadeAlbums[leavesTotalActive + 1] = tuple(
			chain.from_iterable(map(_makeDescendants, dictionarySadeAlbums[leavesTotalActive]))
		)
		# DEVELOPMENT This line is effectively `getMapShape('A000136', leavesTotalActive + 1)` for the new album.
		leavesTotalActive += 1
	return dictionarySadeAlbums

def _makeDescendants(folding: Folding) -> Iterable[Folding]:
	# DEVELOPMENT With a 1Xn map, an increase in map size only adds one leaf. There are only n-ways to insert 1 leaf.
	# With a 2Xn map, there are 2 new leaves, so there are <=n*(n-1) ways to insert 2 leaves.
	return filter(_foldingValid吗, map(partial(_insertLeafAtPile, folding, len(folding)), range(len(folding), pileOrigin, decreasing)))

def _insertLeafAtPile(folding: Folding, leaf: Leaf, pile: Pile) -> Folding:
	return (*folding[:pile], leaf, *folding[pile:])

def _foldingValid吗(folding: Folding) -> bool:
	# DEVELOPMENT You don't need to check every pair of creases because the existing folding is valid.
	# You just need to check new creases against same-parity-in-dimension creases.
	dictionaryLeafPile: dict[Leaf, Pile] = dict(zip(folding, range(len(folding)), strict=True))
	leafLastCreaseAnte: Leaf = len(folding) - 2
	pileCreasePile: tuple[Pile, Pile] = (dictionaryLeafPile[leafLastCreaseAnte], dictionaryLeafPile[leafLastCreaseAnte + 1])
	# DEVELOPMENT The step size of two is essentially hardcoding the parity of `leafComparand` to match the parity of `leafLastCreaseAnte`.
	return not any(map(partial(_creaseViolation吗, dictionaryLeafPile, pileCreasePile), range(leafLastCreaseAnte - 2, leafOrigin - inclusive, 2 * decreasing)))

def _creaseViolation吗(dictionaryLeafPile: dict[Leaf, Pile], pileCreasePile: tuple[Pile, Pile], leafComparand: Leaf) -> bool:
	# DEVELOPMENT With the proper sorting, you only need to check a subset of the new pairs of creases.
	# Don't sort here: sort once when the dictionary is created--maybe. It depends on how you generate the creases to be checked.
	creasesPileSorted: list[tuple[Pile, Pile]] = sorted((pileCreasePile
				, (dictionaryLeafPile[leafComparand], dictionaryLeafPile[leafComparand + 1])
			))
	return creaseViolation吗(creasesPileSorted[0][0], creasesPileSorted[1][0], creasesPileSorted[0][1], creasesPileSorted[1][1])

if __name__ == '__main__':
	leavesTotal: int = 14
	start: float = perf_counter()
	aa = makeAlbums(leavesTotal)
	print(f"{perf_counter() - start:.2f}")
	vv = getValuesKnown('A000682')
	pprint(aa[2], width=160, compact=True)
	print([len(aa[n]) == vv[n] for n in range(2, leavesTotal + inclusive)])
