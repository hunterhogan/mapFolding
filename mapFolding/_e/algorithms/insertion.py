"""Generate and save arch-diagram representatives by inserting endpoints.

You can build endpoint orders incrementally and save surviving diagrams as an album.
You check the arches of each newly inserted endpoint in every dimension of the supplied map shape.

Contents
--------
Functions
	doTheNeedful
		Generate arch-diagram representatives and write an album.
Output paths
	pathAlbum
		Locate the default directory for generated arch-diagram albums.
"""

# DEVELOPMENT module.
# pyright: reportUnusedVariable=false, reportUnusedImport=false
# ruff: file-ignore[print]
from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from hunterMakesPy import decreasing
from itertools import chain
from mapFolding import endpointOrigin, positionOrigin
from mapFolding._e import getMapShapeProducts
from mapFolding._e.algorithms.iff import creaseViolation吗 as archesCross吗, getCreasePost as getArchPost, oddLeaf吗 as endpointOdd吗
from mapFolding.beDRY import defineProcessorLimit, getTotalLeaves as getTotalEndpoints, validateMapShape
from mapFolding.kitFilesystem import makePathFilenameFolds, writeAlbum
from mapFolding.oeis import getValuesKnown
from mapFolding.theSSOT import settingsPackage
from multiprocessing.pool import Pool
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import ArchDiagram, Endpoint, EndpointPosition
	from pathlib import Path

pathAlbum: Path = settingsPackage.pathPackage / '_e' / 'research' / 'albums'
"""Locate the default directory for generated arch-diagram albums.

(AI generated docstring)

You can locate the research album directory within the package. Use the insertion entry point to
derive album filenames from the map shape."""

def _makeDescendants(diagram: ArchDiagram, mapShape: tuple[int, ...]) -> tuple[ArchDiagram, ...]:
	inserting: Iterable[ArchDiagram] = map(partial(_insertEndpointAt, diagram, len(diagram)), range(len(diagram), positionOrigin, decreasing))
	return tuple(filter(partial(_archDiagramValid吗, mapShape=mapShape), inserting))

def _insertEndpointAt(diagram: ArchDiagram, endpoint: Endpoint, position: EndpointPosition) -> ArchDiagram:
	return (*diagram[:position], endpoint, *diagram[position:])

def _archDiagramValid吗(diagram: ArchDiagram, mapShape: tuple[int, ...]) -> bool:
	lookupPosition: dict[Endpoint, EndpointPosition] = dict(zip(diagram, range(len(diagram)), strict=True))
	endpointInserted: Endpoint = len(diagram) - 1
	mapShapeProducts: tuple[int, ...] = getMapShapeProducts(mapShape)

	def dimensionValid吗(dimension: int) -> bool:
		endpointAnte: Endpoint = endpointInserted - mapShapeProducts[dimension]
		if endpointAnte < endpointOrigin or getArchPost(mapShape, endpointAnte, dimension) != endpointInserted:
			return True

		positionsArch: tuple[EndpointPosition, EndpointPosition] = (lookupPosition[endpointAnte], lookupPosition[endpointInserted])
		archParity: int = endpointOdd吗(mapShape, endpointAnte, dimension)

		def comparandViolates吗(endpointComparand: Endpoint) -> bool:
			endpointPost: Endpoint | None = getArchPost(mapShape, endpointComparand, dimension)
			if endpointPost is None or endpointPost >= endpointInserted:
				return False
			if endpointOdd吗(mapShape, endpointComparand, dimension) != archParity:
				return False
			positionsComparand: tuple[EndpointPosition, EndpointPosition] = (lookupPosition[endpointComparand], lookupPosition[endpointPost])
			return _archesCross吗(positionsArch, positionsComparand)

		return not any(map(comparandViolates吗, range(endpointInserted)))

	return all(map(dimensionValid吗, range(len(mapShape))))

def _archesCross吗(positionsArch: tuple[EndpointPosition, EndpointPosition], positionsComparand: tuple[EndpointPosition, EndpointPosition]) -> bool:
	archesSorted: list[tuple[EndpointPosition, EndpointPosition]] = sorted((positionsArch, positionsComparand))
	return archesCross吗(archesSorted[0][0], archesSorted[1][0], archesSorted[0][1], archesSorted[1][1])

def doTheNeedful(mapShape: tuple[int, ...], CPUlimit: ConcurrencyLimit = None) -> Path:
	"""Generate arch-diagram representatives and write an album.

	(AI generated docstring)

	You can insert endpoints in label order, retain diagrams whose new arches do not cross existing
	arches on the same side of the axis, and save them with `mapFolding.kitFilesystem.writeAlbum` [1]. You use the
	module's `pathAlbum` [2] as the output directory.

	Representative selection
	------------------------
	You keep the origin endpoint at position zero. At insertion depth three, you retain diagrams where endpoint
	one precedes endpoint two. You receive an album containing representatives subject to those restrictions,
	without a returned count or recorded symmetry multipliers.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Nonempty tuple of nonnegative dimension lengths defining endpoint connections, with at least one endpoint.
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
		If shape validation fails or the diagram contains zero endpoints.
	FileExistsError
		If the destination album already exists.

	Examples
	--------
	The module's execution block generates an album for the connections of a rectangular map.

		```python
		mapShape = (2, 3)
		pathFilenameAlbum = doTheNeedful(mapShape, -2)
		```

	References
	----------
	[1] `mapFolding.kitFilesystem.writeAlbum`

	[2] `pathAlbum`

	[3] `mapFolding.beDRY.defineProcessorLimit`
	"""
	mapShape = validateMapShape(mapShape)
	totalEndpoints: int = getTotalEndpoints(mapShape)
	if totalEndpoints == 0:
		message: str = f'`mapShape` must have at least one endpoint: {mapShape!r}.'
		raise ValueError(message)

	endpointsInserted: int = 1
	diagramsAtDepth: Iterable[ArchDiagram] = ((endpointOrigin,),)
	pathFilenameAlbum: Path = makePathFilenameFolds(mapShape, pathAlbum, suffix='.album')
	if pathFilenameAlbum.exists():
		message = f'Album already exists: {pathFilenameAlbum}.'
		raise FileExistsError(message)
	workersMaximum: int = defineProcessorLimit(CPUlimit)

	with ExitStack() as resourceManager:
		processManager: Pool | None = None
		if workersMaximum > 1:
			processManager = resourceManager.enter_context(Pool(workersMaximum))
		while endpointsInserted < totalEndpoints:
			endpointsInserted += 1
			if processManager is None:
				diagramsAtDepth = chain.from_iterable(map(partial(_makeDescendants, mapShape=mapShape), diagramsAtDepth))
			else:
				diagramsAtDepth = chain.from_iterable(processManager.imap_unordered(partial(_makeDescendants, mapShape=mapShape), diagramsAtDepth, chunksize=2**10))
			if endpointsInserted == 3:
				diagramsAtDepth = filter(lambda diagram: diagram.index(1) < diagram.index(2), diagramsAtDepth)
		writeAlbum(diagramsAtDepth, pathFilenameAlbum)

	return pathFilenameAlbum

if __name__ == '__main__':
	mapShape: tuple[int, ...] = (2, 3)
	start: float = perf_counter()
	pathFilenameAlbum: Path = doTheNeedful(mapShape, -2)
	print(f"{perf_counter() - start:.2f}")
	countTotal: int = len(pathFilenameAlbum.read_text(encoding='utf-8').splitlines()) * 2 * getTotalEndpoints(mapShape)
	valuesKnown: dict[int, int] = getValuesKnown('A001415')
	print(countTotal == valuesKnown[3], countTotal, valuesKnown[3])

# DEVELOPMENT Changes:
# TODO to make the files smaller, use a truncated notation. The graph notation I created is very
# compact: one delimiter and one `Endpoint` represents one `ArchDiagram`, if the diagrams are sorted. An
# `ArchDiagram` is a permutation, and there is a special notation for permutations, but it's opaque to
# me. However, there are packages that implement it, so that would likely be more robust, even if
# it is not more compact.

# On hold:
# Check each arch as it is added: a crossing will invalidate an entire branch. To do this, I would
# need a new way of iterating(insert, check).

# Completed:
# Don't check every pair of arches because the existing diagram is valid. Check the new arch
# against same parity arches.

# `streamAlbum` only read when requested: good for iterating past existing files.

# No: `ArchDiagram` -> endpoint positions. This requires overwriting keys or reading/changing a ton of keys.
# No: endpointAnte -> endpointPost. ante makes it easier to count backwards from the last endpoint.
