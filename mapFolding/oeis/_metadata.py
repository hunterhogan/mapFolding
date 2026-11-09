from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import dissoc, get_in
from hunterMakesPy import errorL33T
from itertools import count, filterfalse, takewhile
from mapFolding.dataStructures import flattenTriangle, makeLookupDiagonal, makeTriangle
from mapFolding.kitFilesystem import readText, writeText
from mapFolding.oeis._beDRY import format_bFile, formatOEISid, makeValuesKnown, parse_bFile, parseEntry, parseOffset
from mapFolding.oeis._dataBaskets import MetadataOEISid, MetadataOEISidTriangle
from mapFolding.oeis._oeisdata import filterEntryAbsent, makePathFilename_bFile, makePathFilenameEntry
from mapFolding.oeis._theSSOT import settingsOEIS
from operator import add
from typing import TYPE_CHECKING
import sys

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

def updateValuesKnown(oeisID: OEISid, sequence: Mapping[int, int], pathTerms: Path = settingsOEIS.pathTerms) -> Path:
	"""Add terms absent from the known values of `oeisID` to the terms b-file and update shared known values.

	(AI generated docstring)

	You can use this function to retain computed terms between research runs. The terms b-file under
	`pathTerms` holds only computed terms, never the terms of the OEIS snapshot. This function does
	not submit data to OEIS.

	Parameters
	----------
	oeisID : OEISid
		Registered sequence identifier, ignoring case and surrounding whitespace.
	sequence : Mapping[int, int]
		Computed values keyed by flat sequence index. Known values and existing terms-file values take
		precedence over supplied values.
	pathTerms : Path = `mapFolding.oeis.settingsOEIS.pathTerms`
		Directory in the oeisdata directory layout. Supply a directory outside the package for terms
		that must remain private.

	Returns
	-------
	pathFilename : Path
		The b-file path under `pathTerms`. The path may remain absent when there are no terms to add.

	Merge and Persistence
	---------------------
	Known values and existing terms-file values take precedence over `sequence`. When new indices are
	added, comment lines are retained before all terms and `mapFolding.oeis.format_bFile` writes the
	terms in ascending index order. No rewrite occurs when there are no new indices.

	The shared `valuesKnown` dictionary is updated in place with terms-file values at indices it lacks,
	so a private terms directory also adds its existing terms. Existing references returned by
	`getValuesKnown` observe this update. `valueUnknown` becomes one greater than the largest known
	index, or `offset` when there are no terms. Gaps below that index remain possible.

	Examples
	--------
	The cell writer in `easyRun/triangleSemiMake.py` saves each computed triangle term.

		```python
			updateValuesKnown('A400429', {nT: aOFn下k})
		```
	"""
	oeisID = formatOEISid(oeisID)
	metadata: MetadataOEISid = getMetadata(oeisID)
	pathFilename: Path = makePathFilename_bFile(oeisID, pathTerms)
	contents: str = readText(pathFilename, default='')
	terms: dict[int, int] = {}
	if contents:
		terms = parse_bFile(contents)
	sequence = dissoc(sequence, *metadata['valuesKnown'], *terms)
	if sequence:
		terms.update(sequence)
		contents = ''.join(map('{}\n'.format, filter(lambda record: record.lstrip().startswith('#'), contents.splitlines())))
		writeText(contents + format_bFile(terms), pathFilename)
	metadata['valuesKnown'].update(dissoc(terms, *metadata['valuesKnown']))
	metadata['valueUnknown'] = max(metadata['valuesKnown'], default=metadata['offset'] - 1) + 1
	return pathFilename

def updateTriangle(oeisID: OEISid, triangle: 形Triangle, pathTerms: Path = settingsOEIS.pathTerms) -> Path:
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
	pathTerms : Path = `mapFolding.oeis.settingsOEIS.pathTerms`
		Directory in the oeisdata directory layout, passed to `updateValuesKnown`.

	Returns
	-------
	pathFilename : Path
		Terms b-file path. Known values and existing terms-file values take precedence over supplied cells.
	"""
	metadata: MetadataOEISidTriangle = getMetadataTriangle(oeisID)
	return updateValuesKnown(oeisID, flattenTriangle(triangle, metadata['rowLength'], metadata['rowStart'], metadata['offset']), pathTerms)

#================== Meat grinders =================================================================

def _getMetadataOEISid(oeisID: OEISid) -> MetadataOEISid:
	lookupField: dict[str, list[str]] = parseEntry(readText(makePathFilenameEntry(oeisID, settingsOEIS.pathSnapshot)))
	contents: str = readText(makePathFilename_bFile(oeisID, settingsOEIS.pathSnapshot), default='')
	if contents:
		valuesKnown: dict[int, int] = parse_bFile(contents)
	else:
		valuesKnown = makeValuesKnown(lookupField)
	contents = readText(makePathFilename_bFile(oeisID, settingsOEIS.pathTerms), default='')
	if contents:
		valuesKnown.update(dissoc(parse_bFile(contents), *valuesKnown))
	return MetadataOEISid(description=' '.join(lookupField['%N']), offset=parseOffset(lookupField), valuesKnown=valuesKnown, valueUnknown=max(valuesKnown) + 1)

boxOfOEISidAbsent: tuple[OEISid, ...] = filterEntryAbsent(settingsOEIS.boxOfIDs, settingsOEIS.pathSnapshot)
"""Implemented OEIS IDs without an entry in `settingsOEIS.pathSnapshot`."""
if boxOfOEISidAbsent:
	from mapFolding.oeis._oeisdataClone import checkoutSnapshot
	checkoutSnapshot(boxOfOEISidAbsent)
	boxOfOEISidAbsent = filterEntryAbsent(settingsOEIS.boxOfIDs, settingsOEIS.pathSnapshot)
	if boxOfOEISidAbsent:
		message: str = (f"I could not find OEIS snapshot entries for `{boxOfOEISidAbsent = }`. "
			"I will continue with the available metadata.\n")
		sys.stdout.write(message)

boxOfOEISid: tuple[OEISid, ...] = (*filterfalse(boxOfOEISidAbsent.__contains__, settingsOEIS.boxOfIDs),)
dictionaryOEIS: dict[OEISid, MetadataOEISid] = dict(zip(boxOfOEISid, map(_getMetadataOEISid, boxOfOEISid), strict=True))
"""Metadata loaded for implemented OEIS sequences with snapshot entries."""
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
