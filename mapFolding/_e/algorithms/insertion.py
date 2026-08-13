# DEVELOPMENT module.
# pyright: reportUnusedVariable=false, reportUnusedImport=false
# ruff: file-ignore[print]
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""
from __future__ import annotations

from functools import partial
from hunterMakesPy import decreasing, inclusive, zeroIndexed
from itertools import chain, filterfalse
from mapFolding._e import leafOrigin, pileOrigin
from mapFolding._e.algorithms.iff import creaseViolation吗
from mapFolding.beDRY import defineProcessorLimit
from mapFolding.kitFilesystem import makePathFilenameFolds, streamAlbum, writeAlbum
from mapFolding.oeis import getValuesKnown
from mapFolding.theSSOT import settingsPackage
from multiprocessing.pool import Pool
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable
	from hunterMakesPy.theTypes import Limitation
	from mapFolding._e.theTypes import Folding, Leaf, Pile
	from pathlib import Path

pathAlbum: Path = settingsPackage.pathPackage / '_e' / '_development' / 'albums'

def makeAlbums1xn(n: int, nFinal: int, workersMaximum: int) -> Path:
	"""Construct every album through `nFinal`."""
	pathFilename: Path = makePathFilenameFolds((1, n), pathAlbum, suffix='.album')
	album: Iterable[Folding] = streamAlbum(pathFilename)

	processManager: Pool = Pool(workersMaximum)
	while n < nFinal:
		n += 1
		pathFilename = makePathFilenameFolds((1, n), pathFilename.parent, suffix='.album')

		if pathFilename.exists():
			album = streamAlbum(pathFilename)
		else:
			album = tuple(chain.from_iterable(processManager.imap_unordered(_makeDescendants, album, chunksize=2**10)))
			writeAlbum(album, pathFilename)

	return pathFilename

def _makeDescendants(folding: Folding) -> tuple[Folding, ...]:
	inserting: Iterable[Folding] = map(partial(_insertLeafAtPile, folding, len(folding)), range(len(folding), pileOrigin, decreasing))
	return tuple(filter(_foldingValid吗, inserting))

def _insertLeafAtPile(folding: Folding, leaf: Leaf, pile: Pile) -> Folding:
	return (*folding[:pile], leaf, *folding[pile:])

def _foldingValid吗(folding: Folding) -> bool:
	def tt(leafComparandCrease: Leaf) -> tuple[Pile, Pile]:
		return (lookupLeafPile[leafComparandCrease], lookupLeafPile[leafComparandCrease + 1])
	lookupLeafPile: dict[Leaf, Pile] = dict(zip(folding, range(len(folding)), strict=True))
	leafCrease: Leaf = len(folding) - zeroIndexed - 1
	pileCreasePile: tuple[Pile, Pile] = (lookupLeafPile[leafCrease], lookupLeafPile[leafCrease + 1])
	qq = partial(_creaseViolation吗, pileCreasePile)
	ww = map(tt, range(leafCrease - 2, leafOrigin - inclusive, 2 * decreasing))
	return not any(map(qq, ww))

def _creaseViolation吗(pileCreasePile: tuple[Pile, Pile], pileComparandCreasePileComparand: tuple[Pile, Pile]) -> bool:
	creasesPileSorted: list[tuple[Pile, Pile]] = sorted((pileCreasePile, pileComparandCreasePileComparand))
	return creaseViolation吗(creasesPileSorted[0][0], creasesPileSorted[1][0], creasesPileSorted[0][1], creasesPileSorted[1][1])

def doTheNeedful(n: int, nFinal: int, CPUlimit: Limitation = None) -> Path:
	def preemptiveTheorem2(folding: Folding) -> bool:
		# DEVELOPMENT Because all albums descend from this album, enforcing k before r now is
		# preserved, and all albums will be half the size.
		leaf_k: Leaf = 1
		leaf_r: Leaf = 2
		return folding.index(leaf_k) < folding.index(leaf_r)

	if (n < 2) or (nFinal <= n):
		raise ValueError

	album: Iterable[Folding] = ((leafOrigin, 1),)
	pathFilename: Path = makePathFilenameFolds((1, 2), pathAlbum, suffix='.album')

	if pathFilename.exists():
		album = streamAlbum(pathFilename)
	else:
		writeAlbum(album, pathFilename)

	pathFilename = makePathFilenameFolds((1, 3), pathFilename.parent, suffix='.album')

	if pathFilename.exists():
		pass
	else:
		album = tuple(filter(preemptiveTheorem2, chain.from_iterable(map(_makeDescendants, album))))
		writeAlbum(album, pathFilename)

	pathFilename = makePathFilenameFolds((1, n), pathFilename.parent, suffix='.album')

	if pathFilename.exists():
		pass
	else:
		# start lower
		raise ValueError

	workersMaximum: int = defineProcessorLimit(CPUlimit)
	return makeAlbums1xn(n, nFinal, workersMaximum)

if __name__ == '__main__':
	nFinal: int = 17
	start: float = perf_counter()
	aa = doTheNeedful(2, nFinal, -2)
	print(f"{perf_counter() - start:.2f}")
	cc = len(aa.read_text(encoding="utf-8").splitlines()) * 2
	vv = getValuesKnown('A000682')
	print(cc == vv[nFinal], cc, vv[nFinal])

# DEVELOPMENT Changes:
# TODO to make the files smaller, use a truncated notation. The graph notation I created is very
# compact: one delimiter and one `Leaf` represents one `Folding`, if the `Folding` are sorted. A
# `Folding` is a permutation, and there is a special notation for permutations, but it's opaque to
# me. However, there are packages that implement it, so that would likely be more robust, even if
# it is not more compact.

# Check each crease as it is added: a violation will invalidate an entire branch. To do this, I would
# need a new way of iterating(insert, check).

# Don't check every pair of creases because the existing folding is valid. Check the new crease
# against same parity creases.

# `streamAlbum` only read when requested: good for iterating past existing files.

# No: `Folding` -> `PinnedLeaves`. This requires overwriting keys or reading/changing a ton of keys.
# No: creaseAnte -> creasePost. ante makes it easier to count backwards from the last leaf.
