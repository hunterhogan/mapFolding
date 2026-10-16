from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import dissoc, get_in, merge
from hunterMakesPy import errorL33T
from itertools import count, takewhile
from mapFolding.kitFilesystem import getCacheOrURL, readText, writeText
from mapFolding.oeis._beDRY import formatBFile, formatOEISid, parse_bFile
from mapFolding.oeis._dataBaskets import MetadataOEISid
from mapFolding.oeis._theSSOT import cacheDays, oeisIDsImplemented, pathCache
from oeis_tools import oeis_bfile, oeis_url  # pyright: ignore[reportMissingTypeStubs] https://github.com/oeistools/oeis-tools/pull/12
from operator import add
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping, Sequence
	from mapFolding.theTypes import OEISid, 形Triangle, 形TriangleSparse
	from pathlib import Path

#================== Sausage =======================================================================

@cache
def getMetadata(oeisID: OEISid) -> MetadataOEISid:
	"""Retrieve metadata for a specified OEIS sequence.

	(AI generated docstring)

	You can use this function to inspect a registered sequence's description, starting index, and
	known values. This function normalizes `oeisID` with `mapFolding.oeis._beDRY.formatOEISid` [1]
	and returns the corresponding entry from `dictionaryOEIS` [2].

	Parameters
	----------
	oeisID : OEISid
		Registered sequence identifier. Lookup ignores letter case and surrounding whitespace.

	Returns
	-------
	metadata : MetadataOEISid
		Shared metadata dictionary described by `mapFolding.oeis._dataBaskets.MetadataOEISid` [3].
		The result is not copied.

	Unknown Sequences
	-----------------
	This function propagates `KeyError` when the normalized `oeisID` is absent from
	`dictionaryOEIS`.

	Caching
	-------
	Metadata is loaded during module import. `functools.cache` [4] retains lookup results for each
	argument; this function does not refresh cached files or request new sequence data.

	References
	----------
	[1] `mapFolding.oeis._beDRY.formatOEISid`

	[2] `dictionaryOEIS`

	[3] `mapFolding.oeis._dataBaskets.MetadataOEISid`

	[4] functools.cache - Python standard library.
		https://docs.python.org/3/library/functools.html#functools.cache
	"""
	return dictionaryOEIS[formatOEISid(oeisID)]

@cache
def getValuesKnown(oeisID: OEISid) -> dict[int, int]:
	"""Retrieve known sequence values for a specified OEIS sequence.

	(AI generated docstring)

	You can use this function to look up the values loaded for `oeisID` during module import. This
	function normalizes `oeisID` with `mapFolding.oeis._beDRY.formatOEISid` [1] and accesses
	`dictionaryOEIS` [2] without fetching new data.

	Parameters
	----------
	oeisID : OEISid
		Sequence identifier. Lookup ignores letter case and surrounding whitespace.

	Returns
	-------
	valuesKnown : dict[int, int]
		Shared index-to-value dictionary, or `{-errorL33T: -errorL33T}` when the metadata entry or
		its `valuesKnown` field is absent. The sentinel uses `hunterMakesPy.errorL33T` [3]. A stored
		empty dictionary remains empty.

	See Also
	--------
	`getMetadata`
		Retrieve the description, offset, and row settings as well as known values.

	Caching
	-------
	`functools.cache` [4] retains lookup results for each argument. Returned dictionaries are not
	copied; mutations remain visible to later lookups using the same stored result.

	References
	----------
	[1] `mapFolding.oeis._beDRY.formatOEISid`

	[2] `dictionaryOEIS`

	[3] hunterMakesPy - Official repository.
		https://github.com/hunterhogan/hunterMakesPy
	[4] functools.cache - Python standard library.
		https://docs.python.org/3/library/functools.html#functools.cache
	"""
	return get_in((str(formatOEISid(oeisID)),), dictionaryOEIS, dict[str, dict[int, int]]()).get(
		'valuesKnown', dict[int, int]([(-errorL33T, -errorL33T)]))

def getTriangleRows(oeisID: OEISid) -> 形Triangle:
	"""Group known values for `oeisID` into numbered triangle rows.

	(AI generated docstring)

	You can use this function to access a supported sequence as rows for further analysis. This
	function obtains the sequence metadata through `mapFolding.oeis.getMetadata` [1] and returns
	a new dictionary mapping row numbers to lists of known values.

	Row Boundaries
	--------------
	`mapFolding.oeis._dataBaskets.MetadataOEISid` [2] supplies the starting row through `rowStart`,
	falling back to `offset`. When metadata supplies `rowLength`, the function evaluates `rowLength`
	for successive row numbers. Otherwise, requested row lengths begin at one and increase by one,
	regardless of the starting row number.

	`mapFolding.dataStructures.makeTriangle` [3] groups consecutive known indices starting at
	`offset`. Grouping ends at the first missing sequence index. A
	partially known final row is retained. Grouping stops at the first empty row, including an empty
	row caused by a requested length of zero. The function does not check whether the sequence
	represents a triangle.

	Parameters
	----------
	oeisID : OEISid
		Identifier of a sequence present in the package metadata. Lookup ignores letter case and
		leading or trailing whitespace.

	Returns
	-------
	triangleRows : 形Triangle
		Consecutive row numbers mapped to newly allocated lists of known sequence values.

	Unknown Sequences
	-----------------
	This function propagates `KeyError` from the metadata lookup if the normalized `oeisID` is
	absent from the package metadata.

	Examples
	--------
	`getTriangleDiagonal` [4] obtains rows with the following call before selecting a diagonal,
	using the caller's `oeisID`.

		```python
		from mapFolding.oeis import getTriangleRows

		getTriangleRows(oeisID)
		```

	References
	----------
	[1] `mapFolding.oeis.getMetadata`

	[2] `mapFolding.oeis._dataBaskets.MetadataOEISid`

	[3] `mapFolding.dataStructures.makeTriangle`

	[4] `getTriangleDiagonal`

	"""
	from mapFolding.dataStructures import getLookupDiagonal, lookupToTriangle, makeGroupKeys, makeTriangle, triangleToLookup
	metadata: MetadataOEISid = getMetadata(oeisID)
	rowStart: int = metadata.get('rowStart', metadata['offset'])
	rowLengths: Iterable[int] | None = None
	if 'rowLength' in metadata:
		rowLengths = map(metadata['rowLength'], count(rowStart))
	return makeTriangle(map(metadata['valuesKnown'].__getitem__,
		takewhile(metadata['valuesKnown'].__contains__, count(metadata['offset']))), rowLengths, rowStart)

def getTriangleTerms(oeisID: OEISid) -> 形TriangleSparse:
	from mapFolding.dataStructures import getLookupDiagonal, lookupToTriangle, makeGroupKeys, makeTriangle, triangleToLookup
	metadata: MetadataOEISid = getMetadata(oeisID)
	rowStart: int = metadata.get('rowStart', metadata['offset'])
	rowLengths: Iterable[int] | None = None
	if 'rowLength' in metadata:
		rowLengths = map(metadata['rowLength'], count(rowStart))
	return lookupToTriangle(metadata['valuesKnown'], rowLengths, rowStart, metadata['offset'])

def getTriangleDiagonal(oeisID: OEISid, 次diagonal: int, *, fromRight: bool = True) -> dict[int, int]:
	from mapFolding.dataStructures import getLookupDiagonal, lookupToTriangle, makeGroupKeys, makeTriangle, triangleToLookup
	metadata: MetadataOEISid = getMetadata(oeisID)
	return getLookupDiagonal(getTriangleTerms(oeisID), 次diagonal, fromRight=fromRight,
		getGroupLength=metadata.get('rowLength', partial(add, 1 - metadata.get('rowStart', metadata['offset']))))

def getTriangleIndices(oeisID: OEISid, n: int) -> range:
	from mapFolding.dataStructures import getLookupDiagonal, lookupToTriangle, makeGroupKeys, makeTriangle, triangleToLookup
	metadata: MetadataOEISid = getMetadata(oeisID)
	return makeGroupKeys(n, keyGroupFirst=metadata.get('rowStart', metadata['offset']), keyFlatFirst=metadata['offset'],
		getGroupLength=metadata.get('rowLength'))

def updateBFile(oeisID: OEISid, sequence: Mapping[int, int]) -> Path:
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
		writeText(contents + formatBFile(known), pathFilename)
	metadata['valuesKnown'].clear()
	metadata['valuesKnown'].update(sorted(known.items()))
	metadata['valueUnknown'] = max(known, default=metadata['offset'] - 1) + 1
	return pathFilename

def updateTriangle(oeisID: OEISid, triangle: Mapping[int, Sequence[int] | Mapping[int, int]]) -> Path:
	from mapFolding.dataStructures import getLookupDiagonal, lookupToTriangle, makeGroupKeys, makeTriangle, triangleToLookup
	metadata: MetadataOEISid = getMetadata(oeisID)
	return updateBFile(oeisID, triangleToLookup(triangle, keyGroupFirst=metadata.get('rowStart', metadata['offset']),
		keyFlatFirst=metadata['offset'], getGroupLength=metadata.get('rowLength')))


#================== Meat grinders =================================================================

def _getMetadata_bFile(oeisID: OEISid) -> dict[int, int]:
	"""I use this to load b-file values while constructing the metadata registry.

	(AI generated docstring)

	This function obtains OEIS b-file text [1] through `mapFolding.kitFilesystem.getCacheOrURL` [2]
	and parses the text with `mapFolding.oeis._beDRY.parse_bFile` [3].

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

	References
	----------
	[1] B-files - OEIS Wiki.
		https://oeis.org/wiki/B-files
	[2] `mapFolding.kitFilesystem.getCacheOrURL`

	[3] `mapFolding.oeis._beDRY.parse_bFile`

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

	This function normalizes `oeisID` with `mapFolding.oeis._beDRY.formatOEISid` [1] and retrieves
	the sequence's text record through `mapFolding.kitFilesystem.getCacheOrURL` [2]. The result
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
		Starting sequence index, or `hunterMakesPy.errorL33T` [3] when no offset is found.

	Warns
	-----
	UserWarning
		If retrieval returns empty text, a description is missing, or an offset is missing.

	Text Fields
	-----------
	The parser reads `%N` descriptions and the first comma-separated `%O` offset from matching
	sequence records in the OEIS internal format [4]. Later matching fields replace earlier values.
	Invalid offset text propagates `ValueError` from integer conversion.

	References
	----------
	[1] `mapFolding.oeis._beDRY.formatOEISid`

	[2] `mapFolding.kitFilesystem.getCacheOrURL`

	[3] hunterMakesPy - Official repository.
		https://github.com/hunterhogan/hunterMakesPy
	[4] Internal Format Used in the OEIS.
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

def _makeDictionaryOEIS() -> dict[str, MetadataOEISid]:
	"""I use this to assemble the metadata registry during module import.

	This function loads descriptions, offsets, and b-file values for each identifier in
	`mapFolding.oeis._theSSOT.oeisIDsImplemented` [1]. The result also supplies row settings for
	`A259689` and `A400429`, with rows starting at two and complete lengths equal to half the row
	number rounded down.

	Returns
	-------
	dictionary : dict[str, MetadataOEISid]
		New metadata registry. Each `valueUnknown` is one greater than the largest key in
		`valuesKnown`, without checking for gaps or excluding fallback sentinels.

	Empty Values
	------------
	An empty `valuesKnown` dictionary propagates `ValueError` from `max` and prevents completion
	of the registry.

	References
	----------
	[1] `mapFolding.oeis._theSSOT.oeisIDsImplemented`

	"""
	dictionary: dict[str, MetadataOEISid] = {}
	for oeisID in oeisIDsImplemented:
		valuesKnown: dict[int, int] = _getMetadata_bFile(oeisID)
		description, offset = _getMetadataAFile(oeisID)
		dictionary[oeisID] = MetadataOEISid(description=description, offset=offset, valuesKnown=valuesKnown, valueUnknown=max(valuesKnown) + 1)

	dictionary['A259689'].update(rowLength=lambda nRow: nRow // 2, rowStart=2)
	dictionary['A400429'].update(rowLength=lambda nRow: nRow // 2, rowStart=2)

	return dictionary

dictionaryOEIS: dict[str, MetadataOEISid] = _makeDictionaryOEIS()
"""Access the mutable metadata registry loaded during module import."""
