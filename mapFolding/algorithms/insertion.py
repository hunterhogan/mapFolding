"""Generate and save arch-diagram representatives by inserting leaves.

(AI generated docstring)

You can build leaf orders incrementally and save surviving diagrams as an album.
You check the arches of each newly inserted leaf in every dimension of the supplied map shape.

Contents
--------
Functions
	doTheNeedful
		Generate arch-diagram representatives and write an album.
"""

# DEVELOPMENT module.
# pyright: reportUnusedVariable=false, reportUnusedImport=false
from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from hunterMakesPy import decreasing
from itertools import chain
from mapFolding import leafOrigin, pileOrigin
from mapFolding.algorithms.iff import creaseViolation吗 as archesCross吗, getCreasePost as getArchPost, oddLeaf吗
from mapFolding.beDRY import defineProcessorLimit, getMapShapeProducts, getTotalLeaves, validateMapShape
from mapFolding.kitFilesystem import makePathFilenameFolds, writeAlbum
from mapFolding.theSSOT import settingsPackage
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import Folding, Leaf, Pile
	from pathlib import Path

def _makeDescendants(diagram: Folding, mapShape: tuple[int, ...]) -> tuple[Folding, ...]:
	inserting: Iterable[Folding] = map(partial(_insertLeaf, diagram, len(diagram)), range(len(diagram), pileOrigin, decreasing))
	return tuple(filter(partial(_archDiagramValid吗, mapShape=mapShape), inserting))

def _insertLeaf(diagram: Folding, leaf: Leaf, pile: Pile) -> Folding:
	return (*diagram[:pile], leaf, *diagram[pile:])

def _archDiagramValid吗(diagram: Folding, mapShape: tuple[int, ...]) -> bool:
	lookupPile: dict[Leaf, Pile] = dict(zip(diagram, range(len(diagram)), strict=True))
	leafInserted: Leaf = len(diagram) - 1
	mapShapeProducts: tuple[int, ...] = getMapShapeProducts(mapShape)

	def dimensionValid吗(dimension: int) -> bool:
		leafAnte: Leaf = leafInserted - mapShapeProducts[dimension]
		if leafAnte < leafOrigin or getArchPost(mapShape, leafAnte, dimension) != leafInserted:
			return True

		pilesArch: tuple[Pile, Pile] = (lookupPile[leafAnte], lookupPile[leafInserted])
		archParity: int = oddLeaf吗(mapShape, leafAnte, dimension)

		def comparandViolates吗(leafComparand: Leaf) -> bool:
			leafPost: Leaf | None = getArchPost(mapShape, leafComparand, dimension)
			if leafPost is None or leafPost >= leafInserted:
				return False
			if oddLeaf吗(mapShape, leafComparand, dimension) != archParity:
				return False
			pilesComparand: tuple[Pile, Pile] = (lookupPile[leafComparand], lookupPile[leafPost])
			return _archesCross吗(pilesArch, pilesComparand)

		return not any(map(comparandViolates吗, range(leafInserted)))

	return all(map(dimensionValid吗, range(len(mapShape))))

def _archesCross吗(pilesArch: tuple[Pile, Pile], pilesComparand: tuple[Pile, Pile]) -> bool:
	archesSorted: list[tuple[Pile, Pile]] = sorted((pilesArch, pilesComparand))
	return archesCross吗(archesSorted[0][0], archesSorted[1][0], archesSorted[0][1], archesSorted[1][1])

def doTheNeedful(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit = None) -> Path:
	"""Generate arch-diagram representatives and write an album.

	(AI generated docstring)

	You can insert leaves in label order, retain diagrams whose new arches do not cross existing
	arches on the same side of the axis, and save them with `mapFolding.kitFilesystem.writeAlbum` [1]. You use the
	shared `settingsPackage.pathAlbum` [2] as the output directory.

	Representative selection
	------------------------
	You keep the origin leaf at pile zero. At insertion depth three, you retain diagrams where leaf
	one precedes leaf two. You receive an album containing representatives subject to those restrictions,
	without a returned count or recorded symmetry multipliers.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Nonempty tuple of nonnegative dimension lengths defining leaf connections, with at least one leaf.
	CPUlimit : ConcurrencyLimit = None
		Processor limit interpreted by `mapFolding.beDRY.defineProcessorLimit` [3]. `None`, `False`, and `0`
		use available processors; `True` selects one. Positive values limit usage and negative values reserve
		processors.

	Returns
	-------
	pathFilenameAlbum : Path
		Path of the album written for `mapShape`.

	Raises
	------
	ValueError
		If shape validation fails or the diagram contains zero leaves.
	FileExistsError
		If the destination album already exists.

	Examples
	--------
	The `easyRun.insertion` module generates albums for the connections of a rectangular map.

		```python
		mapShape = (2, 3)
		pathFilenameAlbum = doTheNeedful(mapShape, -2)
		```

	References
	----------
	[1] `mapFolding.kitFilesystem.writeAlbum`

	[2] `mapFolding.theSSOT.settingsPackage.pathAlbum`

	[3] `mapFolding.beDRY.defineProcessorLimit`
	"""
	mapShape = validateMapShape(mapShape)
	totalLeaves: int = getTotalLeaves(mapShape)
	if totalLeaves == 0:
		message: str = f'`mapShape` must have at least one leaf: {mapShape!r}.'
		raise ValueError(message)

	leavesInserted: int = 1
	diagramsAtDepth: Iterable[Folding] = ((leafOrigin,),)
	pathFilenameAlbum: Path = makePathFilenameFolds(mapShape, settingsPackage.pathAlbum, suffix='.album')
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
				diagramsAtDepth = chain.from_iterable(map(partial(_makeDescendants, mapShape=mapShape), diagramsAtDepth))
			else:
				diagramsAtDepth = chain.from_iterable(processManager.imap_unordered(partial(_makeDescendants, mapShape=mapShape), diagramsAtDepth, chunksize=2**10))
			if leavesInserted == 3:
				diagramsAtDepth = filter(lambda diagram: diagram.index(1) < diagram.index(2), diagramsAtDepth)
		writeAlbum(diagramsAtDepth, pathFilenameAlbum)

	return pathFilenameAlbum

# DEVELOPMENT Changes:
# TODO to make the files smaller, use a truncated notation. The graph notation I created is very
# compact: one delimiter and one `Leaf` represents one `Folding`, if the diagrams are sorted. An
# `Folding` is a permutation, and there is a special notation for permutations, but it's opaque to
# me. However, there are packages that implement it, so that would likely be more robust, even if
# it is not more compact.

# On hold:
# Check each arch as it is added: a crossing will invalidate an entire branch. To do this, I would
# need a new way of iterating(insert, check).

# Completed:
# Don't check every pair of arches because the existing diagram is valid. Check the new arch
# against same parity arches.

# `streamAlbum` only read when requested: good for iterating past existing files.

# No: `Folding` -> leaf piles. This requires overwriting keys or reading/changing a ton of keys.
# No: leafAnte -> leafPost. ante makes it easier to count backwards from the last leaf.
