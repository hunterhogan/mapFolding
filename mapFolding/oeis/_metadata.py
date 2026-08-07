from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import compose, get_in
from hunterMakesPy import errorL33T
from itertools import filterfalse
from mapFolding.kitFilesystem import getCacheOrURL
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.oeis._dataBaskets import MetadataOEISid
from mapFolding.oeis._theSSOT import oeisIDsImplemented, pathCache
from more_itertools import take
from operator import methodcaller
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from pathlib import Path

#================== Sausage =======================================================================

@cache
def getMetadata(oeisID: OEISid) -> MetadataOEISid:
	"""Retrieve metadata for a specified OEIS sequence.

	(AI generated docstring)

	This function fetches the description, offset, known values, and the next unknown index for an
	OEIS sequence. It utilizes cached data when available or retrieves fresh data from the OEIS
	website. The metadata is structured in a `MetadataOEISid` dictionary.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to retrieve.

	Returns
	-------
	metadata : MetadataOEISid
		A dictionary containing the sequence's description, offset, known values, and next unknown index.
	"""
	return dictionaryOEIS[formatOEISid(oeisID)]

@cache
def getValuesKnown(oeisID: OEISid) -> dict[int, int]:
	"""Retrieve known sequence values for a specified OEIS sequence.

	(AI generated docstring)

	This function fetches the complete set of known values for an OEIS sequence by accessing cached
	data when available or retrieving fresh data from the OEIS website. The data is parsed from the
	standard OEIS b-file format.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to retrieve.

	Returns
	-------
	OEISsequence : dict[int, int]
		A dictionary mapping sequence indices to their corresponding values, or a fallback dictionary
		containing {-1: -1} if retrieval fails.
	"""
	# TODO Create humpy_cytoolz.get_in overloads.
	return get_in((formatOEISid(oeisID), 'valuesKnown'), dictionaryOEIS, {-1: -1})

#================== Meat grinders =================================================================

def _getMetadata_bFile(oeisID: OEISid) -> dict[int, int]:
	"""Retrieve known sequence values for a specified OEIS sequence.

	(AI generated docstring)

	This function fetches the complete set of known values for an OEIS sequence by accessing cached
	data when available or retrieving fresh data from the OEIS website. The data is parsed from the
	standard OEIS b-file format.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to retrieve.

	Returns
	-------
	OEISsequence : dict[int, int]
		A dictionary mapping sequence indices to their corresponding values, or a fallback dictionary
		containing {-1: -1} if retrieval fails.
	"""
	# TODO centralize b-file format.
	filename: str = f"b{oeisID[1:]}.txt"
	pathFilenameCache: Path = pathCache / filename
	url: str = f"https://oeis.org/{oeisID}/{filename}"

	oeisData: str = getCacheOrURL(pathFilenameCache, url)

	if not oeisData:
		message: str = f"Failed to retrieve OEIS sequence information for {oeisID = }."
		warnings.warn(message, stacklevel=0)

	n_aOFn: dict[int, int] = {}
	if oeisData:
		n_aOFn.update(map(compose(tuple[int, int], partial(map, int), partial(take, 2)), map(methodcaller('split'), filterfalse(methodcaller('startswith', '#'), oeisData.strip().splitlines()))))
	return n_aOFn

def _getMetadataAFile(oeisID: OEISid) -> tuple[str, int]:
	"""Retrieve the description and offset metadata for an OEIS sequence.

	(AI generated docstring)

	This function extracts the mathematical description and starting index offset from OEIS sequence
	metadata using the machine-readable text format. It employs the same caching mechanism as other
	retrieval functions to minimize network requests.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to retrieve.

	Returns
	-------
	description : str
		A human-readable string describing the sequence's mathematical meaning.
	offset : int
		The starting index of the sequence, typically 0 or 1 depending on mathematical context.

	Parsing Details
	---------------
	Descriptions are parsed from OEIS %N entries and offsets from %O entries in the machine-readable
	text format. If metadata cannot be retrieved, the function issues warning messages and returns
	fallback values.
	"""
	oeisID = formatOEISid(oeisID)
	pathFilenameCache: Path = pathCache / f"{oeisID}.txt"
	url: str = f"https://oeis.org/search?q=id:{oeisID}&fmt=text"

	oeisData: str = getCacheOrURL(pathFilenameCache, url)

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
	"""Construct metadata for every implemented OEIS sequence."""
	dictionary: dict[str, MetadataOEISid] = {}
	for oeisID in oeisIDsImplemented:
		valuesKnown: dict[int, int] = _getMetadata_bFile(oeisID)
		description, offset = _getMetadataAFile(oeisID)
		dictionary[oeisID] = MetadataOEISid(description=description, offset=offset, valuesKnown=valuesKnown, valueUnknown=max(valuesKnown) + 1)
	return dictionary

dictionaryOEIS: dict[str, MetadataOEISid] = _makeDictionaryOEIS()
"""Metadata for every implemented OEIS sequence."""
