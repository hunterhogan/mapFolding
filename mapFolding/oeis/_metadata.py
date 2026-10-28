from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import dissoc, get_in, merge
from hunterMakesPy import errorL33T
from itertools import count, takewhile
from mapFolding.dataStructures import getLookupDiagonal, lookupToTriangle, makeGroupKeys, makeTriangle, triangleToLookup
from mapFolding.kitFilesystem import getCacheOrURL, readText, writeText
from mapFolding.oeis._beDRY import formatBFile, formatOEISid, parse_bFile
from mapFolding.oeis._dataBaskets import MetadataOEISid
from mapFolding.oeis._theSSOT import cacheDays, oeisIDsImplemented, pathCache
from oeis_tools import oeis_bfile, oeis_url
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

def getTriangleRows(oeisID: OEISid) -> 形Triangle:
	"""Group known values for `oeisID` into numbered triangle rows.

	(AI generated docstring)

	You can use this function to access a supported sequence as rows for further analysis. This
	function obtains the sequence metadata through `mapFolding.oeis.getMetadata` and returns
	a new dictionary mapping row numbers to lists of known values.

	Row Boundaries
	--------------
	`mapFolding.oeis._dataBaskets.MetadataOEISid` supplies the starting row through `rowStart`,
	falling back to `offset`. When metadata supplies `rowLength`, the function evaluates `rowLength`
	for successive row numbers. Otherwise, requested row lengths begin at one and increase by one,
	regardless of the starting row number.

	`mapFolding.dataStructures.makeTriangle` groups consecutive known indices starting at
	`offset`. Grouping ends at the first missing sequence index. A
	partially known final row is retained. Grouping stops at the first empty row, including an empty
	row caused by a requested length of zero. The function does not check whether the sequence
	represents a triangle.
	Use `getTriangleTerms` to preserve known values beyond gaps in the flat indices.

	Parameters
	----------
	oeisID : OEISid
		Identifier of a sequence present in the package metadata. Lookup ignores letter case and
		leading or trailing whitespace.

	Returns
	-------
	triangleRows : 形Triangle
		Consecutive row numbers mapped to newly allocated lists of known sequence values.

	See Also
	--------
	`getTriangleTerms`
		Preserve available cells beyond gaps in the flat sequence indices.

	Unknown Sequences
	-----------------
	This function propagates `KeyError` from the metadata lookup if the normalized `oeisID` is
	absent from the package metadata.

	Examples
	--------
	`mapFolding.oeis.A400429oeisByFormula.checkFormulas` obtains the stored semi-meander rows.

		```python
		from mapFolding.oeis import getTriangleRows

		triangleRows = getTriangleRows('A400429')
		```
	"""
	metadata: MetadataOEISid = getMetadata(oeisID)
	rowStart: int = metadata.get('rowStart', metadata['offset'])
	rowLengths: Iterable[int] | None = None
	if 'rowLength' in metadata:
		rowLengths = map(metadata['rowLength'], count(rowStart))
	return makeTriangle(map(metadata['valuesKnown'].__getitem__,
		takewhile(metadata['valuesKnown'].__contains__, count(metadata['offset']))), rowLengths, rowStart)

def getTriangleTerms(oeisID: OEISid) -> 形TriangleSparse:
	"""Group available terms for `oeisID` into sparse triangle rows.

	(AI generated docstring)

	You can use this function to inspect known triangle cells even when some flat sequence indices
	are missing. This function uses `getMetadata` for the row layout and
	`mapFolding.dataStructures.lookupToTriangle` to preserve each available cell's column.

	Parameters
	----------
	oeisID : OEISid
		Registered sequence identifier, ignoring letter case and surrounding whitespace.

	Returns
	-------
	triangleTerms : 形TriangleSparse
		New dictionaries mapping row numbers to available columns, with columns starting at one.

	See Also
	--------
	`getTriangleRows`
		Read dense rows from the consecutive prefix of known sequence indices.

	Row Layout
	----------
	The first row uses `rowStart`, or `offset` when `rowStart` is absent. Flat indices start at
	`offset`. A supplied `rowLength` determines each complete row length; otherwise lengths start
	at one and increase by one. Rows through the largest stored flat index are retained, including
	empty rows caused by gaps. A partial final row keeps the original column keys.
	"""
	metadata: MetadataOEISid = getMetadata(oeisID)
	rowStart: int = metadata.get('rowStart', metadata['offset'])
	rowLengths: Iterable[int] | None = None
	if 'rowLength' in metadata:
		rowLengths = map(metadata['rowLength'], count(rowStart))
	return lookupToTriangle(metadata['valuesKnown'], rowLengths, rowStart, metadata['offset'])

def getTriangleDiagonal(oeisID: OEISid, 次diagonal: int, *, fromRight: bool = True) -> dict[int, int]:
	"""Select an available triangle diagonal for `oeisID` without shifting missing cells.

	(AI generated docstring)

	You can use this function to compare one diagonal with a formula or independently computed
	counts. This function obtains sparse rows from `getTriangleTerms` and selects cells with
	`mapFolding.dataStructures.getLookupDiagonal`.

	Parameters
	----------
	oeisID : OEISid
		Registered triangle sequence identifier, ignoring case and surrounding whitespace.
	次diagonal : int
		Offset from the selected row edge, starting at one. Zero selects no ordinary cell.
	fromRight : bool = True
		Measure from the complete row's right edge when `True`, or the left edge when `False`.

	Returns
	-------
	diagonal : dict[int, int]
		Available selected cells keyed by ascending row number. Missing cells are omitted.

	Complete Row Lengths
	--------------------
	Right-edge selection uses the metadata `rowLength`, or the default lengths starting at one.
	Missing cells therefore retain their positions relative to the complete row's edge.
	Selection propagates `ValueError` from the diagonal selector if `次diagonal` is negative.
	"""
	metadata: MetadataOEISid = getMetadata(oeisID)
	return getLookupDiagonal(getTriangleTerms(oeisID), 次diagonal, fromRight=fromRight,
		getGroupLength=metadata.get('rowLength', partial(add, 1 - metadata.get('rowStart', metadata['offset']))))

def getTriangleIndices(oeisID: OEISid, n: int) -> range:
	"""Locate the complete triangle row `n` in the flat sequence `oeisID`.

	(AI generated docstring)

	You can use this function to match a row calculation with b-file indices. This function reads
	the row layout from `getMetadata` and delegates index calculation to
	`mapFolding.dataStructures.makeGroupKeys`.

	Parameters
	----------
	oeisID : OEISid
		Registered triangle sequence identifier, ignoring case and surrounding whitespace.
	n : int
		Row number, no smaller than the first row specified by the metadata.

	Returns
	-------
	indices : range
		Flat indices for every column of the complete row, whether or not those terms are known.

	Row Layout
	----------
	Flat indices start at `offset`; row numbers start at `rowStart`, falling back to `offset`.
	A supplied `rowLength` determines row lengths. Otherwise lengths start at one and increase
	by one, independently of the first row number.
	Index calculation propagates `ValueError` if `n` precedes the first metadata row.
	"""
	metadata: MetadataOEISid = getMetadata(oeisID)
	return makeGroupKeys(n, keyGroupFirst=metadata.get('rowStart', metadata['offset']), keyFlatFirst=metadata['offset'],
		getGroupLength=metadata.get('rowLength'))

def updateBFile(oeisID: OEISid, sequence: Mapping[int, int]) -> Path:
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
		updateBFile('A400429', {nT: aOFn下k})
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
		writeText(contents + formatBFile(known), pathFilename)
	metadata['valuesKnown'].clear()
	metadata['valuesKnown'].update(sorted(known.items()))
	metadata['valueUnknown'] = max(known, default=metadata['offset'] - 1) + 1
	return pathFilename

def updateTriangle(oeisID: OEISid, triangle: Mapping[int, Sequence[int] | Mapping[int, int]]) -> Path:
	"""Add dense or sparse rows from `triangle` to the local sequence cache for `oeisID`.

	(AI generated docstring)

	You can use this function to save row computations without converting column positions to
	flat indices yourself. This function flattens `triangle` using
	`mapFolding.dataStructures.triangleToLookup`, then delegates persistence and shared-value
	updates to `updateBFile`.

	Parameters
	----------
	oeisID : OEISid
		Registered triangle sequence identifier, ignoring case and surrounding whitespace.
	triangle : Mapping[int, Sequence[int] | Mapping[int, int]]
		Row numbers mapped to dense cell sequences or sparse dictionaries with columns starting
		at one. Supplied columns must fit the complete row length defined by the metadata.

	Returns
	-------
	pathFilename : Path
		Local b-file path. Existing file and metadata values take precedence over supplied cells.

	Row Layout
	----------
	The metadata `offset` selects the first flat index; `rowStart` selects the first row, falling
	back to `offset`. A supplied `rowLength` sets each complete row length. Otherwise lengths
	start at one and increase by one. Missing rows and columns keep their flat index positions.
	"""
	metadata: MetadataOEISid = getMetadata(oeisID)
	return updateBFile(oeisID, triangleToLookup(triangle, keyGroupFirst=metadata.get('rowStart', metadata['offset']),
		keyFlatFirst=metadata['offset'], getGroupLength=metadata.get('rowLength')))


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

def _makeDictionaryOEIS() -> dict[str, MetadataOEISid]:
	"""I use this to assemble the metadata registry during module import.

	(AI generated docstring)

	This function loads descriptions, offsets, and b-file values for each identifier in
	`mapFolding.oeis._theSSOT.oeisIDsImplemented`. The result also supplies row settings for
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
"""Metadata for every implemented OEIS sequence."""
