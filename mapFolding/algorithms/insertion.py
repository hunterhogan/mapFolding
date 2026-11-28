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
	"""I use this operation to retain valid diagrams after inserting the next leaf.

	(AI generated docstring)

	Parameters
	----------
	diagram : Folding
		Valid partial leaf order with the origin leaf fixed at pile zero.
	mapShape : tuple[int, ...]
		Dimension lengths defining the leaf connections.

	Returns
	-------
	descendants : tuple[Folding, ...]
		Valid insertions of leaf `len(diagram)` at piles from `len(diagram)` down to one.
	"""
	inserting: Iterable[Folding] = map(partial(_insertLeaf, diagram, len(diagram)), range(len(diagram), pileOrigin, decreasing))
	return tuple(filter(partial(_archDiagramValid吗, mapShape=mapShape), inserting))

def _insertLeaf(diagram: Folding, leaf: Leaf, pile: Pile) -> Folding:
	"""I use this operation to insert a leaf at a pile in a new diagram.

	(AI generated docstring)

	Parameters
	----------
	diagram : Folding
		Leaf order before insertion.
	leaf : Leaf
		Label of the leaf to insert.
	pile : Pile
		Insertion position, interpreted as a Python slice boundary.

	Returns
	-------
	diagramInserted : Folding
		New leaf order with `leaf` between the two slices of `diagram`.
	"""
	return (*diagram[:pile], leaf, *diagram[pile:])

def _archDiagramValid吗(diagram: Folding, mapShape: tuple[int, ...]) -> bool:
	"""I use this test to reject crossings introduced by the last inserted leaf.

	(AI generated docstring)

	Parameters
	----------
	diagram : Folding
		Partial leaf order containing consecutive labels from zero through `len(diagram) - 1`.
		The diagram before insertion must already be valid.
	mapShape : tuple[int, ...]
		Dimension lengths defining the leaf connections and arch parity.

	Returns
	-------
	isValid : bool
		`True` when every new arch avoids existing arches of the same parity in its dimension.
	"""
	lookupPile: dict[Leaf, Pile] = dict(zip(diagram, range(len(diagram)), strict=True))
	leafInserted: Leaf = len(diagram) - 1
	mapShapeProducts: tuple[int, ...] = getMapShapeProducts(mapShape)

	def dimensionValid吗(dimension: int) -> bool:
		"""Check the newly inserted leaf's arch in one dimension for crossings.

		(AI generated docstring)

		Parameters
		----------
		dimension : int
			Zero-based dimension in the enclosing map shape.

		Returns
		-------
		isValid : bool
			`True` when no arch ends at the inserted leaf in this dimension, or its arch
			crosses no existing arch of the same parity.
		"""
		leafAnte: Leaf = leafInserted - mapShapeProducts[dimension]
		if leafAnte < leafOrigin or getArchPost(mapShape, leafAnte, dimension) != leafInserted:
			return True

		pilesArch: tuple[Pile, Pile] = (lookupPile[leafAnte], lookupPile[leafInserted])
		archParity: int = oddLeaf吗(mapShape, leafAnte, dimension)

		def comparandViolates吗(leafComparand: Leaf) -> bool:
			"""Check whether an existing arch crosses the newly inserted leaf's arch.

			(AI generated docstring)

			Parameters
			----------
			leafComparand : Leaf
				Initial leaf of a possible existing arch in the enclosing dimension.

			Returns
			-------
			isViolation : bool
				`True` when the arch is complete, has the new arch's parity, and crosses it.
			"""
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
	"""I use this test to detect alternating endpoints of two arches.

	(AI generated docstring)

	Parameters
	----------
	pilesArch : tuple[Pile, Pile]
		Pile positions of the first arch's endpoints, in leaf-label order.
	pilesComparand : tuple[Pile, Pile]
		Pile positions of the second arch's endpoints, in leaf-label order.

	Returns
	-------
	isCrossing : bool
		`True` when the arches have interleaving endpoints and therefore cross on the same side.
	"""
	archesSorted: list[tuple[Pile, Pile]] = sorted((pilesArch, pilesComparand))
	return archesCross吗(archesSorted[0][0], archesSorted[1][0], archesSorted[0][1], archesSorted[1][1])

def doTheNeedful(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit = None) -> Path:
	"""Generate arch-diagram representatives and write an album.

	(AI generated docstring)

	You can insert leaves in label order, retain diagrams whose new arches do not cross existing
	arches on the same side of the axis, and save them with `mapFolding.kitFilesystem.writeAlbum`. You
	use the shared `settingsPackage.pathAlbum` as the output directory.

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
		Processor limit interpreted by `mapFolding.beDRY.defineProcessorLimit`. `None`, `False`, and `0`
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

	"""
	mapShape = validateMapShape(mapShape)
	totalLeaves: int = getTotalLeaves(mapShape)
	if totalLeaves == 0:
		message: str = f'I received `{mapShape = }` with `{totalLeaves = }`, but I need at least one leaf.'
		raise ValueError(message)

	leavesInserted: int = 1
	diagramsAtDepth: Iterable[Folding] = ((leafOrigin,),)
	pathFilenameAlbum: Path = makePathFilenameFolds(mapShape, settingsPackage.pathAlbum, suffix='.album')
	if pathFilenameAlbum.exists():
		message: str = f'I found that `{pathFilenameAlbum = }` already exists.'
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
# compact: one delimiter and one `Leaf` represents one `Folding`, if the diagrams are sorted. A
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
