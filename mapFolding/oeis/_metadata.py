from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import dissoc, get_in, merge
from hunterMakesPy import errorL33T
from itertools import count, takewhile
from mapFolding.dataStructures import flattenTriangle, makeLookupDiagonal, makeTriangle
from mapFolding.kitFilesystem import getCacheOrURL, readText, writeText
from mapFolding.oeis._beDRY import format_bFile, formatOEISid, parse_bFile
from mapFolding.oeis._dataBaskets import MetadataOEISid, MetadataOEISidTriangle
from mapFolding.oeis._theSSOT import cacheDays, oeisIDsImplemented, pathCache
from oeis_tools import oeis_bfile, oeis_url
from operator import add
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
	from collections.abc import Mapping
	from mapFolding.theTypes import OEISid, 形Triangle
	from pathlib import Path

#================== Sausage =======================================================================

@cache
def getMetadata(oeisID: OEISid) -> MetadataOEISid:
	"""Retrieve metadata for a specified OEIS sequence.

	(AI generated docstring)

	You can use this function to inspect a registered sequence's description, starting index, and
	known values. This function normalizes `oeisID` with `mapFolding.oeis._beDRY.formatOEISid`
	and returns the corresponding entry from `dictionaryOEIS`.

	Parameters
	----------
	oeisID : OEISid
		Registered sequence identifier. Lookup ignores letter case and surrounding whitespace.

	Returns
	-------
	metadata : MetadataOEISid
		Shared metadata dictionary described by `mapFolding.oeis._dataBaskets.MetadataOEISid`.
		The result is not copied.

	Unknown Sequences
	-----------------
	This function propagates `KeyError` when the normalized `oeisID` is absent from
	`dictionaryOEIS`.

	Caching
	-------
	Metadata is loaded during module import. `functools.cache` retains lookup results for each
	argument; this function does not refresh cached files or request new sequence data.
	"""
	return dictionaryOEIS[formatOEISid(oeisID)]

@cache
def getValuesKnown(oeisID: OEISid) -> dict[int, int]:
	"""Retrieve known sequence values for a specified OEIS sequence.

	(AI generated docstring)

	You can use this function to look up the values loaded for `oeisID` during module import. This
	function normalizes `oeisID` with `mapFolding.oeis._beDRY.formatOEISid` and accesses
	`dictionaryOEIS` without fetching new data.

	Parameters
	----------
	oeisID : OEISid
		Sequence identifier. Lookup ignores letter case and surrounding whitespace.

	Returns
	-------
	valuesKnown : dict[int, int]
		Shared index-to-value dictionary, or `{-errorL33T: -errorL33T}` when the metadata entry or
		its `valuesKnown` field is absent. The sentinel uses `hunterMakesPy.errorL33T`. A stored
		empty dictionary remains empty.

	See Also
	--------
	`getMetadata`
		Retrieve the description, offset, and row settings as well as known values.

	Caching
	-------
	`functools.cache` retains lookup results for each argument. Returned dictionaries are not
	copied; mutations remain visible to later lookups using the same stored result.
	"""
	return get_in((str(formatOEISid(oeisID)),), dictionaryOEIS, dict[str, dict[int, int]]()).get(
		'valuesKnown', dict[int, int]([(-errorL33T, -errorL33T)]))

@cache
def getMetadataTriangle(oeisID: OEISid) -> MetadataOEISidTriangle:
	"""Retrieve metadata, including the row layout, for the triangle `oeisID`.

	(AI generated docstring)

	The result is the same shared dictionary that `getMetadata` returns for `oeisID`.

	Parameters
	----------
	oeisID : OEISid
		Identifier of a triangle in the package metadata. Lookup ignores letter case and surrounding
		whitespace.

	Returns
	-------
	metadata : MetadataOEISidTriangle
		Shared metadata dictionary described by `mapFolding.oeis._dataBaskets.MetadataOEISidTriangle`.
	"""
	return dictionaryTriangle[formatOEISid(oeisID)]

def getTriangle(oeisID: OEISid) -> dict[int, list[int]]:
	"""Group the known values of the triangle `oeisID` into rows.

	(AI generated docstring)

	The metadata of `oeisID` supplies the row layout: `rowStart` is the first row, `offset` is the
	index of its first cell, and `rowLength` is the length of each complete row.
	`mapFolding.dataStructures.makeTriangle` groups the known values with consecutive indices
	starting at `offset`. Grouping ends at the first missing index, so the final row can be partial.

	Parameters
	----------
	oeisID : OEISid
		Identifier of a triangle in the package metadata. Lookup ignores letter case and
		leading or trailing whitespace.

	Returns
	-------
	triangle : dict[int, list[int]]
		Consecutive row numbers mapped to newly allocated lists of known values.

	Examples
	--------
	`mapFolding.oeis.A400429oeisByFormula.checkFormulas` obtains the stored semi-meander rows.

		```python
		from mapFolding.oeis import getTriangle

		triangle = getTriangle('A400429')
		```
	"""
	metadata: MetadataOEISidTriangle = getMetadataTriangle(oeisID)
	return makeTriangle(map(metadata['valuesKnown'].__getitem__, takewhile(metadata['valuesKnown'].__contains__, count(metadata['offset'])))
		, metadata['rowLength'], metadata['rowStart'])

def getTriangleDiagonal(oeisID: OEISid, 次diagonal: int) -> dict[int, int]:
	"""Select a diagonal of the triangle `oeisID`.

	(AI generated docstring)

	Diagonal `次diagonal` contains the cell at distance `次diagonal` from the end of each complete row,
	so diagonal 1 is the last cell of each row. `mapFolding.dataStructures.makeLookupDiagonal` selects
	the cells of `getTriangle` using the metadata `rowLength`, so a partial final row contributes a
	cell only when the cell is known.

	Parameters
	----------
	oeisID : OEISid
		Identifier of a triangle in the package metadata, ignoring case and surrounding whitespace.
	次diagonal : int
		Distance from the end of each complete row, starting at one.

	Returns
	-------
	diagonal : dict[int, int]
		Known cells of the diagonal keyed by row number.
	"""
	return makeLookupDiagonal(getTriangle(oeisID), 次diagonal, getMetadataTriangle(oeisID)['rowLength'])

def updateValuesKnown(oeisID: OEISid, sequence: Mapping[int, int]) -> Path:
	"""Add missing terms to the local b-file for `oeisID` and update shared known values.

	(AI generated docstring)

	You can use this function to retain computed terms between research runs. This function merges
	`sequence` with the known values from `getMetadata` and the existing local b-file,
	then returns the local path. This function does not submit data to OEIS.

	Parameters
	----------
	oeisID : OEISid
		Registered sequence identifier, ignoring case and surrounding whitespace.
	sequence : Mapping[int, int]
		Computed values keyed by flat sequence index. Existing file and metadata values take
		precedence over supplied values.

	Returns
	-------
	pathFilename : Path
		B-file path under `pathCache`. The path may remain absent when there are no terms to add.

	Merge and Persistence
	---------------------
	Existing file values take precedence over metadata values. Metadata values take precedence
	over `sequence`. The negative `hunterMakesPy.errorL33T` sentinel is excluded from additions.
	When missing indices are added, comment lines are retained before all terms and
	`mapFolding.oeis.formatBFile` writes the terms in ascending index order. No rewrite occurs
	when there are no missing indices.

	The shared `valuesKnown` dictionary is cleared and updated in place from the merged local
	values, in index order. Existing references returned by `getValuesKnown` observe this
	update. `valueUnknown` becomes one greater than the largest stored index, or `offset` when
	there are no terms. Gaps below that index remain possible.

	Examples
	--------
	The cell writer in `easyRun/triangleSemiMake.py` saves each computed triangle term.

		```python
		updateValuesKnown('A400429', {nT: aOFn下k})
		```
	"""
	oeisID = formatOEISid(oeisID)
	metadata: MetadataOEISid = getMetadata(oeisID)
	pathFilename: Path = pathCache / oeis_bfile(oeisID)
	contents: str = readText(pathFilename, default='')
	known: dict[int, int] = {}
	if contents:
		known = parse_bFile(contents)
	sequence = dissoc(merge(sequence, metadata['valuesKnown']), -errorL33T, *known)
	if sequence:
		known.update(sequence)
		contents = ''.join(map('{}\n'.format, filter(lambda record: record.lstrip().startswith('#'), contents.splitlines())))
		writeText(contents + format_bFile(known), pathFilename)
	metadata['valuesKnown'].clear()
	metadata['valuesKnown'].update(sorted(known.items()))
	metadata['valueUnknown'] = max(known, default=metadata['offset'] - 1) + 1
	return pathFilename

def updateTriangle(oeisID: OEISid, triangle: 形Triangle) -> Path:
	"""Add the cells of `triangle` to the local b-file for the triangle `oeisID`.

	(AI generated docstring)

	You can use this function to save row computations without converting cells to flat indices
	yourself. `mapFolding.dataStructures.flattenTriangle` keys each cell by its flat index using the
	metadata row layout, then `updateValuesKnown` persists the cells and updates the shared values.

	Parameters
	----------
	oeisID : OEISid
		Identifier of a triangle in the package metadata, ignoring case and surrounding whitespace.
	triangle : 形Triangle
		Row numbers mapped to the cells of each row, starting with the first cell. A row can be a
		prefix of the complete row.

	Returns
	-------
	pathFilename : Path
		Local b-file path. Existing file and metadata values take precedence over supplied cells.
	"""
	metadata: MetadataOEISidTriangle = getMetadataTriangle(oeisID)
	return updateValuesKnown(oeisID, flattenTriangle(triangle, metadata['rowLength'], metadata['rowStart'], metadata['offset']))

#================== Meat grinders =================================================================

def _getMetadata_bFile(oeisID: OEISid) -> dict[int, int]:
	"""I use this to load b-file values while constructing the metadata registry.

	(AI generated docstring)

	This function obtains OEIS b-file text through `mapFolding.kitFilesystem.getCacheOrURL`
	and parses the text with `mapFolding.oeis._beDRY.parse_bFile`.

	Parameters
	----------
	oeisID : OEISid
		Sequence identifier supplied by the metadata construction routine.

	Returns
	-------
	valuesKnown : dict[int, int]
		Parsed sequence values. Empty text produces the parser's sentinel dictionary.

	Warns
	-----
	UserWarning
		If retrieval returns empty text.
	"""
	filename: str = oeis_bfile(oeisID)
	pathFilenameCache: Path = pathCache / filename
	url: str = oeis_url(oeisID, fmt='bfile')

	oeisData: str = getCacheOrURL(pathFilenameCache, cacheDays, url)

	if not oeisData:
		message: str = f"Failed to retrieve OEIS sequence information for {oeisID = }."
		warnings.warn(message, stacklevel=0)

	return parse_bFile(oeisData)

def _getMetadataAFile(oeisID: OEISid) -> tuple[str, int]:
	"""I use this to load descriptions and starting indices for the metadata registry.

	(AI generated docstring)

	This function normalizes `oeisID` with `mapFolding.oeis._beDRY.formatOEISid` and retrieves
	the sequence's text record through `mapFolding.kitFilesystem.getCacheOrURL`. The result
	contains the sequence description and starting index.

	Parameters
	----------
	oeisID : OEISid
		The identifier of the OEIS sequence to retrieve.

	Returns
	-------
	description : str
		Sequence description, or a diagnostic message when no description is found.
	offset : int
		Starting sequence index, or `hunterMakesPy.errorL33T` when no offset is found.

	Warns
	-----
	UserWarning
		If retrieval returns empty text, a description is missing, or an offset is missing.

	Text Fields
	-----------
	The parser reads `%N` descriptions and the first comma-separated `%O` offset from matching
	sequence records in the OEIS internal format [1]. Later matching fields replace earlier values.
	Invalid offset text propagates `ValueError` from integer conversion.

	References
	----------
	[1] Internal Format Used in the OEIS.
		https://oeis.org/eishelp1.html
	"""
	oeisID = formatOEISid(oeisID)
	pathFilenameCache: Path = pathCache / f"{oeisID}.txt"
	url: str = oeis_url(oeisID, fmt='text')

	oeisData: str = getCacheOrURL(pathFilenameCache, cacheDays, url)

	if not oeisData:
		message: str = f"Failed to retrieve OEIS sequence information for {oeisID = }."
		warnings.warn(message, stacklevel=0)

	description: str = ''
	offset: int | None = None
	if oeisData:
		for lineOEIS in oeisData.splitlines():
			lineOEIS = lineOEIS.strip().split()
			if 3 <= len(lineOEIS):
				fieldCode, sequenceID, *fieldData = lineOEIS
				if sequenceID == oeisID:
					if fieldCode == '%N':
						description = ' '.join(fieldData)
					elif fieldCode == '%O':
						offset = int(fieldData[0].split(',')[0])
	if not description:
		message: str = f"I could not find a description for `{oeisID = }`."
		warnings.warn(message, stacklevel=2)
		description = message
	if offset is None:
		message: str = f"I could not find an offset for `{oeisID = }`."
		warnings.warn(message, stacklevel=2)
		offset = errorL33T
	return description, offset

def _getMetadataOEISid(oeisID: OEISid) -> MetadataOEISid:
	valuesKnown: dict[int, int] = _getMetadata_bFile(oeisID)
	description, offset = _getMetadataAFile(oeisID)
	return MetadataOEISid(description=description, offset=offset, valuesKnown=valuesKnown, valueUnknown=max(valuesKnown) + 1)

dictionaryOEIS: dict[OEISid, MetadataOEISid] = dict(zip(oeisIDsImplemented, map(_getMetadataOEISid, oeisIDsImplemented), strict=True))
"""Metadata for every implemented OEIS sequence."""
dictionaryTriangle: dict[OEISid, MetadataOEISidTriangle] = {
	'A001263': MetadataOEISidTriangle(**dictionaryOEIS['A001263'], rowLength=int, rowStart=1, columnStart=1),
	'A008828': MetadataOEISidTriangle(**dictionaryOEIS['A008828'], rowLength=int, rowStart=1, columnStart=1),
	'A053121': MetadataOEISidTriangle(**dictionaryOEIS['A053121'], rowLength=partial(add, 1), rowStart=0, columnStart=0),
	'A060972': MetadataOEISidTriangle(**dictionaryOEIS['A060972'], rowLength=int, rowStart=1, columnStart=1),
	'A259689': MetadataOEISidTriangle(**dictionaryOEIS['A259689'], rowLength=lambda nRow: nRow // 2, rowStart=2, columnStart=2),
	'A259698': MetadataOEISidTriangle(**dictionaryOEIS['A259698'], rowLength=partial(add, -1), rowStart=2, columnStart=1),
	'A259701': MetadataOEISidTriangle(**dictionaryOEIS['A259701'], rowLength=partial(add, -1), rowStart=2, columnStart=1),
	'A259703': MetadataOEISidTriangle(**dictionaryOEIS['A259703'], rowLength=partial(add, -1), rowStart=2, columnStart=1),
	'A287548': MetadataOEISidTriangle(**dictionaryOEIS['A287548'], rowLength=int, rowStart=1, columnStart=1),
	'A331499': MetadataOEISidTriangle(**dictionaryOEIS['A331499'], rowLength=int, rowStart=1, columnStart=1),
	'A339179': MetadataOEISidTriangle(**dictionaryOEIS['A339179'], rowLength=lambda nRow: nRow // 2, rowStart=2, columnStart=2),
	'A380369': MetadataOEISidTriangle(**dictionaryOEIS['A380369'], rowLength=partial(add, 1), rowStart=0, columnStart=0),
	'A400429': MetadataOEISidTriangle(**dictionaryOEIS['A400429'], rowLength=lambda nRow: nRow // 2, rowStart=2, columnStart=1),
}
"""Metadata, including the row layout, for every implemented triangle."""
dictionaryOEIS.update(dictionaryTriangle)
