# DEVELOPMENT module.
# pyright: reportUnusedVariable=false, reportUnusedImport=false
# ruff: file-ignore[print]
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""
from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from hunterMakesPy import decreasing
from itertools import chain
from mapFolding import leafOrigin, pileOrigin
from mapFolding._e import getMapShapeProducts
from mapFolding._e.algorithms.iff import creaseViolation吗, getCreasePost, oddLeaf吗
from mapFolding.beDRY import defineProcessorLimit, getTotalLeaves, validateMapShape
from mapFolding.kitFilesystem import makePathFilenameFolds, writeAlbum
from mapFolding.oeis import getValuesKnown
from mapFolding.theSSOT import settingsPackage
from multiprocessing.pool import Pool
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import Folding, Leaf, Pile
	from pathlib import Path

pathAlbum: Path = settingsPackage.pathPackage / '_e' / 'research' / 'albums'

def _makeDescendants(folding: Folding, mapShape: tuple[int, ...]) -> tuple[Folding, ...]:
	inserting: Iterable[Folding] = map(partial(_insertLeafAtPile, folding, len(folding)), range(len(folding), pileOrigin, decreasing))
	return tuple(filter(partial(_foldingValid吗, mapShape=mapShape), inserting))

def _insertLeafAtPile(folding: Folding, leaf: Leaf, pile: Pile) -> Folding:
	return (*folding[:pile], leaf, *folding[pile:])

def _foldingValid吗(folding: Folding, mapShape: tuple[int, ...]) -> bool:
	lookupLeafPile: dict[Leaf, Pile] = dict(zip(folding, range(len(folding)), strict=True))
	leafInserted: Leaf = len(folding) - 1
	mapShapeProducts: tuple[int, ...] = getMapShapeProducts(mapShape)

	def dimensionValid吗(dimension: int) -> bool:
		leafCrease: Leaf = leafInserted - mapShapeProducts[dimension]
		if leafCrease < leafOrigin or getCreasePost(mapShape, leafCrease, dimension) != leafInserted:
			return True

		pileCreasePile: tuple[Pile, Pile] = (lookupLeafPile[leafCrease], lookupLeafPile[leafInserted])
		parityCrease: int = oddLeaf吗(mapShape, leafCrease, dimension)

		def comparandViolates吗(leafComparand: Leaf) -> bool:
			leafComparandCrease: Leaf | None = getCreasePost(mapShape, leafComparand, dimension)
			if leafComparandCrease is None or leafComparandCrease >= leafInserted:
				return False
			if oddLeaf吗(mapShape, leafComparand, dimension) != parityCrease:
				return False
			pileComparandCreasePile: tuple[Pile, Pile] = (lookupLeafPile[leafComparand], lookupLeafPile[leafComparandCrease])
			return _creaseViolation吗(pileCreasePile, pileComparandCreasePile)

		return not any(map(comparandViolates吗, range(leafInserted)))

	return all(map(dimensionValid吗, range(len(mapShape))))

def _creaseViolation吗(pileCreasePile: tuple[Pile, Pile], pileComparandCreasePileComparand: tuple[Pile, Pile]) -> bool:
	creasesPileSorted: list[tuple[Pile, Pile]] = sorted((pileCreasePile, pileComparandCreasePileComparand))
	return creaseViolation吗(creasesPileSorted[0][0], creasesPileSorted[1][0], creasesPileSorted[0][1], creasesPileSorted[1][1])

def doTheNeedful(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit = None) -> Path:
	mapShape = validateMapShape(mapShape)
	totalLeaves: int = getTotalLeaves(mapShape)
	if totalLeaves == 0:
		message: str = f'`mapShape` must have at least one leaf: {mapShape!r}.'
		raise ValueError(message)

	leavesInserted: int = 1
	foldingsAtDepth: Iterable[Folding] = ((leafOrigin,),)
	pathFilenameAlbum: Path = makePathFilenameFolds(mapShape, pathAlbum, suffix='.album')
	if pathFilenameAlbum.exists():
		message = f'Album already exists: {pathFilenameAlbum}.'
		raise FileExistsError(message)
	workersMaximum: int = defineProcessorLimit(CPUlimit)

	with ExitStack() as resourceManager:
		processManager: Pool | None = None
		if workersMaximum > 1:
			processManager = resourceManager.enter_context(Pool(workersMaximum))
		while leavesInserted < totalLeaves:
			leavesInserted += 1
			if processManager is None:
				foldingsAtDepth = chain.from_iterable(map(partial(_makeDescendants, mapShape=mapShape), foldingsAtDepth))
			else:
				foldingsAtDepth = chain.from_iterable(processManager.imap_unordered(partial(_makeDescendants, mapShape=mapShape), foldingsAtDepth, chunksize=2**10))
			if leavesInserted == 3:
				foldingsAtDepth = filter(lambda folding: folding.index(1) < folding.index(2), foldingsAtDepth)
		writeAlbum(foldingsAtDepth, pathFilenameAlbum)

	return pathFilenameAlbum

if __name__ == '__main__':
	mapShape: tuple[int, ...] = (2, 3)
	start: float = perf_counter()
	pathFilenameAlbum: Path = doTheNeedful(mapShape, -2)
	print(f"{perf_counter() - start:.2f}")
	countTotal: int = len(pathFilenameAlbum.read_text(encoding='utf-8').splitlines()) * 2 * getTotalLeaves(mapShape)
	valuesKnown: dict[int, int] = getValuesKnown('A001415')
	print(countTotal == valuesKnown[3], countTotal, valuesKnown[3])

# DEVELOPMENT Changes:
# TODO to make the files smaller, use a truncated notation. The graph notation I created is very
# compact: one delimiter and one `Leaf` represents one `Folding`, if the `Folding` are sorted. A
# `Folding` is a permutation, and there is a special notation for permutations, but it's opaque to
# me. However, there are packages that implement it, so that would likely be more robust, even if
# it is not more compact.

# On hold:
# Check each crease as it is added: a violation will invalidate an entire branch. To do this, I would
# need a new way of iterating(insert, check).

# Completed:
# Don't check every pair of creases because the existing folding is valid. Check the new crease
# against same parity creases.

# `streamAlbum` only read when requested: good for iterating past existing files.

# No: `Folding` -> `PinnedLeaves`. This requires overwriting keys or reading/changing a ton of keys.
# No: creaseAnte -> creasePost. ante makes it easier to count backwards from the last leaf.
