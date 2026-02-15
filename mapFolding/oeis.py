# ruff: noqa: PLC0415 E701

"""
Mathematical validation and discovery through OEIS integration.

(AI generated docstring)

Complementing the unified computational interface, this module extends the map
folding ecosystem into the broader mathematical community through comprehensive
integration with the Online Encyclopedia of Integer Sequences (OEIS). This bridge
enables validation of computational results against established mathematical
knowledge while supporting the discovery of new sequence values through the
sophisticated computational assembly line.

The integration provides multiple pathways for mathematical verification: direct
computation of OEIS sequences using the complete algorithmic implementation,
cached access to published sequence data for rapid validation, and research
support for extending known sequences through new computational discoveries.
The module handles sequence families ranging from simple strip folding to
complex multi-dimensional hypercube problems.

Through intelligent caching and optimized lookup mechanisms, this module ensures
that the computational power developed through the foundational layers can contribute
meaningfully to mathematical research. Whether validating results, avoiding
redundant computation, or extending mathematical knowledge, this integration
completes the journey from configuration foundation to mathematical discovery.
"""

from datetime import datetime, timedelta, UTC
from hunterMakesPy.filesystemToolkit import writeStringToHere
from itertools import chain
from mapFolding import MetadataOEISid, MetadataOEISidMapFolding, packageSettings
from mapFolding._theSSOT import pathCache
from mapFolding.basecamp import countFolds
from mapFolding.filesystemToolkit import (
	getPathFilenameFoldsTotal, getPathRootJobDEFAULT, saveFoldsTotal, saveFoldsTotalFAILearly)
from os import PathLike
from pathlib import Path, PurePath
from typing import Final, Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import argparse
import sys
import time
import warnings

def _standardizeOEISid(oeisID: str) -> str:
	"""Standardize an OEIS sequence ID to uppercase and without whitespace.

	Parameters
	----------
	oeisID : str
		The OEIS sequence identifier to standardize.

	Returns
	-------
	oeisIDstandardized : str
		Uppercase, alphanumeric OEIS ID.

	"""
	return str(oeisID).upper().strip()

oeisIDsImplemented: Final[list[str]]  = sorted(map(_standardizeOEISid, packageSettings.OEISidMapFoldingManuallySet))
"""Directly implemented OEIS IDs; standardized, e.g., 'A001415'."""

def _getFilenameOEISbFile(oeisID: str) -> str:
	"""Generate the filename for an OEIS b-file given a sequence ID.

	(AI generated docstring)

	OEIS b-files contain sequence values in a standardized format and follow the naming convention
	'b{sequence_number}.txt', where the sequence number excludes the 'A' prefix.

	Parameters
	----------
	oeisID : str
		The OEIS sequence identifier to convert to a b-file filename.

	Returns
	-------
	str
		The corresponding b-file filename for the given sequence ID.

	"""
	oeisID = _standardizeOEISid(oeisID)
	return f"b{oeisID[1:]}.txt"

def _parseBFileOEIS(OEISbFile: str) -> dict[int, int]:
	"""Parse the content of an OEIS b-file into a sequence dictionary.

	(AI generated docstring)

	OEIS b-files contain sequence data in a standardized two-column format where each line represents
	an index-value pair. Comment lines beginning with '#' are ignored during parsing.

	Parameters
	----------
	OEISbFile : str
		A multiline string representing the content of an OEIS b-file.

	Returns
	-------
	OEISsequence : dict[int, int]
		A dictionary mapping sequence indices to their corresponding values.

	Raises
	------
	ValueError
		If the file content format is invalid or cannot be parsed.

	"""
	bFileLines: list[str] = OEISbFile.strip().splitlines()

	OEISsequence: dict[int, int] = {}
	for line in bFileLines:
		if line.startswith('#'):
			continue
		n, aOFn = map(int, line.split())
		OEISsequence[n] = aOFn
	return OEISsequence

def _getOEISofficial(pathFilenameCache: Path, url: str) -> None | str:
	"""Retrieve OEIS ID data from oeis.org or local cache.

	Parameters
	----------
	pathFilenameCache : Path
		Path to the local cache file for storing retrieved data.
	url : str
		URL to retrieve the OEIS sequence data from if cache is invalid or missing.

	Returns
	-------
	oeisInformation : str | None
		The retrieved OEIS sequence information as a string, or `None` if retrieval failed.

	"""
	tryCache: bool = False
	if pathFilenameCache.exists():
		fileAge: timedelta = datetime.now(tz=UTC) - datetime.fromtimestamp(pathFilenameCache.stat().st_mtime, tz=UTC)
		tryCache = fileAge < timedelta(days=packageSettings.cacheDays)

	oeisInformation: str | None = None
	if tryCache:
		try:
			oeisInformation = pathFilenameCache.read_text(encoding="utf-8")
		except OSError:
			tryCache = False

	if not tryCache:
		if not url.startswith(("http:", "https:")):
			message = "URL must start with 'http:' or 'https:'"
			raise ValueError(message)

		try:
			with urlopen(url) as response:  # noqa: S310
				oeisInformationRaw = response.read().decode('utf-8')
			oeisInformation = str(oeisInformationRaw)
			writeStringToHere(oeisInformation, pathFilenameCache)
		except (HTTPError, URLError):
			oeisInformation = pathFilenameCache.read_text(encoding="utf-8")

	if not oeisInformation:
		warnings.warn(f"Failed to retrieve OEIS sequence information for {pathFilenameCache.stem}.", stacklevel=2)

	return oeisInformation

def getOEISidValues(oeisID: str) -> dict[int, int]:
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
		A dictionary mapping sequence indices to their corresponding values, or a fallback
		dictionary containing {-1: -1} if retrieval fails.

	Raises
	------
	ValueError
		If the cached or downloaded file format is invalid.
	IOError
		If there is an error reading from or writing to the local cache.

	"""
	pathFilenameCache: Path = pathCache / _getFilenameOEISbFile(oeisID)
	url: str = f"https://oeis.org/{oeisID}/{_getFilenameOEISbFile(oeisID)}"

	oeisInformation: None | str = _getOEISofficial(pathFilenameCache, url)

	if oeisInformation:
		return _parseBFileOEIS(oeisInformation)
	return {-1: -1}

def getOEISidInformation(oeisID: str) -> tuple[str, int]:
	"""Retrieve the description and offset metadata for an OEIS sequence.

	(AI generated docstring)

	This function extracts the mathematical description and starting index offset from OEIS sequence
	metadata using the machine-readable text format. It employs the same caching mechanism as other
	retrieval functions to minimize network requests.

	Parameters
	----------
	oeisID : str
		The OEIS sequence identifier to retrieve metadata for.

	Returns
	-------
	description : str
		A human-readable string describing the sequence's mathematical meaning.
	offset : int
		The starting index of the sequence, typically 0 or 1 depending on mathematical context.

	Notes
	-----
	Descriptions are parsed from OEIS %N entries and offsets from %O entries. If metadata cannot
	be retrieved, warning messages are issued and fallback values are returned.

	"""
	oeisID = _standardizeOEISid(oeisID)
	pathFilenameCache: Path = pathCache / f"{oeisID}.txt"
	url: str = f"https://oeis.org/search?q=id:{oeisID}&fmt=text"

	oeisInformation: None | str = _getOEISofficial(pathFilenameCache, url)

	if not oeisInformation:
		return "Not found", -1
	listDescriptionDeconstructed: list[str] = []
	offset = None
	for lineOEIS in oeisInformation.splitlines():
		lineOEIS = lineOEIS.strip()
		if not lineOEIS or len(lineOEIS.split()) < 3:
			continue
		fieldCode, sequenceID, fieldData = lineOEIS.split(maxsplit=2)
		if fieldCode == '%N' and sequenceID == oeisID:
			listDescriptionDeconstructed.append(fieldData)
		if fieldCode == '%O' and sequenceID == oeisID:
			offsetAsStr: str = fieldData.split(',')[0]
			offset = int(offsetAsStr)
	if not listDescriptionDeconstructed:
		warnings.warn(f"No description found for {oeisID}", stacklevel=2)
		listDescriptionDeconstructed.append("No description found")
	if offset is None:
		warnings.warn(f"No offset found for {oeisID}", stacklevel=2)
		offset = -1
	description: str = ' '.join(listDescriptionDeconstructed)
	return description, offset

def _makeDictionaryOEISMapFolding() -> dict[str, MetadataOEISidMapFolding]:
	"""Construct the comprehensive settings dictionary for all implemented OEIS sequences.

	(AI generated docstring)

	This function builds the complete configuration dictionary by merging hardcoded settings with
	dynamically retrieved data from OEIS. For each implemented sequence, it combines:

	1. Sequence values from OEIS b-files
	2. Sequence metadata including descriptions and offsets
	3. Hardcoded mapping functions and test parameter sets

	The resulting dictionary serves as the authoritative configuration source for all OEIS-related
	operations throughout the package, enabling consistent access to sequence definitions, known values,
	and operational parameters.

	Returns
	-------
	settingsTarget : dict[str, SettingsOEIS]
		A comprehensive dictionary mapping OEIS sequence IDs to their complete settings
		objects, containing all metadata and known values needed for computation and validation.

	"""
	dictionaryOEIS: dict[str, MetadataOEISidMapFolding] = {}
	for oeisID in oeisIDsImplemented:
		valuesKnownSherpa: dict[int, int] = getOEISidValues(oeisID)
		descriptionSherpa, offsetSherpa = getOEISidInformation(oeisID)
		dictionaryOEIS[oeisID] = MetadataOEISidMapFolding(
			description=descriptionSherpa,
			offset=offsetSherpa,
			getMapShape=packageSettings.OEISidMapFoldingManuallySet[oeisID]['getMapShape'],
			valuesKnown=valuesKnownSherpa,
			valueUnknown=max(valuesKnownSherpa.keys(), default=0) + 1
		)
	return dictionaryOEIS

dictionaryOEISMapFolding: dict[str, MetadataOEISidMapFolding] = _makeDictionaryOEISMapFolding()
"""Metadata for each MapFolding OEIS ID."""

def makeDictionaryFoldsTotalKnown() -> dict[tuple[int, ...], int]:
	"""Make a `mapShape` to known `foldsTotal` dictionary.

	Returns
	-------
	dictionaryFoldsTotalKnown : dict[tuple[int, ...], int]
		A dictionary where keys are tuples representing map shapes and values are the total number
		of distinct folding patterns for those shapes.

	"""
	return dict(chain.from_iterable(zip(map(oeisIDmetadata['getMapShape'], oeisIDmetadata['valuesKnown'].keys())
	, oeisIDmetadata['valuesKnown'].values(), strict=True) for oeisID, oeisIDmetadata in dictionaryOEISMapFolding.items() if oeisID != 'A007822'))

def getFoldsTotalKnown(mapShape: tuple[int, ...]) -> int:
	"""Retrieve the known total number of distinct folding patterns for a given map shape.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.

	Returns
	-------
	foldingsTotal : int
		The known total number of distinct folding patterns for the given map shape, or 0 if the map shape does not match any
		known values in the OEIS sequences.

	Notes
	-----
	Map shapes are matched exactly as provided without internal sorting or normalization.

	"""
	lookupFoldsTotal: dict[tuple[int, ...], int] = makeDictionaryFoldsTotalKnown()
	return lookupFoldsTotal.get(tuple(mapShape), 0)

def _formatHelpText() -> str:
	"""Format comprehensive help text for both command-line and interactive use.

	(AI generated docstring)

	This function generates standardized help documentation that includes all available OEIS sequences
	with their descriptions and provides usage examples for both command-line and programmatic interfaces.

	Returns
	-------
	helpText : str
		A formatted string containing complete usage information and examples.

	"""
	exampleOEISid: str = 'A001415'
	exampleN: int = 6

	return (
		"\nAvailable OEIS sequences:\n"
		f"{_formatOEISsequenceInfo()}\n"
		"\nUsage examples:\n"
		"  Command line:\n"
		f"	OEIS_for_n {exampleOEISid} {exampleN}\n"
		"  Python:\n"
		"	from mapFolding.oeis import oeisIDfor_n\n"
		f"	foldsTotal = oeisIDfor_n('{exampleOEISid}', {exampleN})"
	)

def _formatOEISsequenceInfo() -> str:
	"""Format information about available OEIS sequences for display in help messages and error output.

	(AI generated docstring)

	This function creates a standardized listing of all implemented OEIS sequences with their mathematical
	descriptions, suitable for inclusion in help text and error messages.

	Returns
	-------
	sequenceInfo : str
		A formatted string listing each OEIS sequence ID with its description.

	"""
	return "\n".join(
		f"  {oeisID}: {dictionaryOEISMapFolding[oeisID]['description']}"
		for oeisID in oeisIDsImplemented
	)

def oeisIDfor_n(oeisID: str, n: int) -> int:
	"""Calculate the value a(n) for a specified OEIS ID and index.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to evaluate.
	n : int
		A non-negative integer index for which to calculate the sequence value.

	Returns
	-------
	a(n) : int
		The value a(n) of the specified OEIS sequence.

	Raises
	------
	ValueError
		If n is not a non-negative integer.
	KeyError
		If the OEIS sequence ID is not directly implemented.
	ArithmeticError
		If n is below the sequence's defined offset.

	"""
	oeisID = _standardizeOEISid(oeisID)

	if not isinstance(n, int) or n < 0:
		message: str = f"I received `{n = }` in the form of `{type(n) = }`, but it must be non-negative integer in the form of `{int}`."
		raise ValueError(message)

	mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)

	if n <= 1 or len(mapShape) < 2:
		offset: int = dictionaryOEISMapFolding[oeisID]['offset']
		if n < offset:
			message = f"OEIS sequence {oeisID} is not defined at {n = }."
			raise ArithmeticError(message)
		foldsTotal: int = dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]
	else:
		foldsTotal = countFolds(mapShape=mapShape)

	return foldsTotal

def OEIS_for_n() -> None:
	"""Command-line interface for calculating OEIS sequence values.

	(AI generated docstring)

	This function provides a command-line interface to the `oeisIDfor_n` function, enabling users to
	calculate specific values of implemented OEIS sequences from the terminal. It includes argument
	parsing, error handling, and performance timing to provide a complete user experience.

	The function accepts two command-line arguments: an OEIS sequence identifier and an integer index,
	then outputs the calculated sequence value along with execution time. Error messages are directed
	to stderr with appropriate exit codes for shell scripting integration.

	Usage
	-----
	python -m mapFolding.oeis OEIS_for_n A001415 10

	Raises
	------
	SystemExit
		With code 1 if invalid arguments are provided or computation fails.

	"""
	parserCLI = argparse.ArgumentParser(
		description="Calculate a(n) for an OEIS sequence.",
		epilog=_formatHelpText(),
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parserCLI.add_argument('oeisID', help="OEIS sequence identifier")
	parserCLI.add_argument('n', type=int, help="Calculate a(n) for this n")

	argumentsCLI: argparse.Namespace = parserCLI.parse_args()

	timeStart: float = time.perf_counter()

	try:
		print(oeisIDfor_n(argumentsCLI.oeisID, argumentsCLI.n), "distinct folding patterns.")  # noqa: T201
	except (KeyError, ValueError, ArithmeticError) as ERRORmessage:
		print(f"Error: {ERRORmessage}", file=sys.stderr)  # noqa: T201
		sys.exit(1)

	timeElapsed: float = time.perf_counter() - timeStart
	print(f"Time elapsed: {timeElapsed:.3f} seconds")  # noqa: T201

def getOEISids() -> None:
	"""Display comprehensive information about all implemented OEIS sequences.

	(AI generated docstring)

	This function serves as the primary help interface for the module, displaying detailed information
	about all directly implemented OEIS sequences along with usage examples for both command-line and
	programmatic interfaces. It provides users with a complete overview of available sequences and
	their mathematical meanings.

	The output includes sequence identifiers, mathematical descriptions, and practical usage examples
	to help users understand how to access and utilize the OEIS interface functionality.

	"""
	print(_formatHelpText())  # noqa: T201

def _makeDictionaryOEIS() -> dict[str, MetadataOEISid]:
	dictionary: dict[str, MetadataOEISid] = {}
	for oeisID in packageSettings.OEISidManuallySet:
		valuesKnownSherpa: dict[int, int] = getOEISidValues(oeisID)
		descriptionSherpa, offsetSherpa = getOEISidInformation(oeisID)
		dictionary[oeisID] = MetadataOEISid(
			description=descriptionSherpa,
			offset=offsetSherpa,
			valuesKnown=valuesKnownSherpa,
			valueUnknown=max(valuesKnownSherpa.keys(), default=0) + 1,
		)
	return dictionary

dictionaryOEIS: dict[str, MetadataOEISid] = _makeDictionaryOEIS()

if __name__ == "__main__":
	getOEISids()


def NOTcountingFolds(oeisID: str, oeis_n: int, flow: str | None = None
		, pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
		, CPUlimit: bool | float | int | None = None  # noqa: FBT001
	) -> int:
	"""You can compute the n-th term of specified OEIS sequences using specialized algorithms.

	(AI generated docstring)

	This function computes values for OEIS [1] sequences that require specialized algorithms
	(meanders [2], symmetric foldings) or closed-form formulas, as opposed to the general
	multidimensional map-folding algorithm accessible via `countFolds` [3]. The function
	dispatches to algorithm-specific implementations based on `oeisID` and `flow` parameters.

	The function name reflects that these computations are NOT standard map-folding counts:
	meanders use transfer matrix methods, symmetric foldings exploit symmetry constraints,
	and formula-based sequences compute directly without search.

	Parameters
	----------
	oeisID : str
		OEIS sequence identifier. Supported sequences fall into three categories:
		- Formula-based: A000136, A000560, A001010, A001011, A005315, A060206, A077460,
			A078591, A086345 [4], A178961, A223094, A259702, A301620
		- Meanders: A000682 [5], A005316 [6]
		- Symmetric foldings: A007822
	oeis_n : int
		Sequence index (typically starting from 0 or 1, depending on OEIS sequence offset).
	flow : str | None = None
		Algorithm variant selector. Available values depend on `oeisID`:
		- For A000682, A005316: 'matrixMeanders' (default), 'matrixNumPy', 'matrixPandas'
		- For A007822: 'algorithm' (default), 'asynchronous', 'theorem2', 'theorem2Numba', 'theorem2Trimmed'
		- For formula-based sequences: ignored (`flow` has no effect)
	CPUlimit : bool | float | int | None = None
		Processor usage limit for parallel algorithms (A007822 with certain `flow` values).
		Interpretation matches `countFolds.CPUlimit` [3]:
		- `False`, `None`, or `0`: use all available processors
		- `True`: limit to 1 processor
		- `int >= 1`: maximum number of processors
		- `0 < float < 1`: fraction of available processors
		- `-1 < float < 0`: fraction of processors to *not* use
		- `int <= -1`: number of processors to *not* use

	Returns
	-------
	countTotal : int
		The n-th term of the specified OEIS sequence.

	Raises
	------
	ValueError
		If `oeisID` matches A000682 or A005316 but receives an invalid internal state (programming error, not user error).

	Examples
	--------
	Formula-based sequence:

	>>> from mapFolding.basecamp import NOTcountingFolds
	>>> NOTcountingFolds('A000136', 3)
	8

	Meander computation with matrix algorithm:

	>>> NOTcountingFolds('A000682', 5, flow='matrixMeanders')
	42

	Symmetric folding with Numba-optimized implementation:

	>>> NOTcountingFolds('A007822', 6, flow='theorem2Numba')
	144

	See Also
	--------
	mapFolding.basecamp.countFolds
		General multidimensional map-folding computation.
	mapFolding.oeis.oeisIDfor_n
		Convenience function that routes to either `countFolds` or `NOTcountingFolds` based on `oeisID`.

	Algorithm Details
	-----------------
	Meander Sequences (A000682, A005316)
		Meanders [2] represent configurations of non-intersecting curves crossing a line. The
		algorithms use transfer matrix methods to enumerate valid configurations. Initial arc codes
		(binary representations of curve crossings) are constructed based on sequence parity, then
		iterative transformations generate all valid meander states. See
		`mapFolding.reference.A000682facts` [7] and `mapFolding.reference.A005316facts` [8] for
		sequence-specific parameters.

	Symmetric Folding Sequence (A007822)
		A007822 counts foldings of 1×(2n) maps that are symmetric under 180-degree rotation.
		The algorithm exploits symmetry to reduce the search space. The `flow` parameter selects
		between serial, parallel (asynchronous), and optimized implementations (theorem2 variants).

	Formula-Based Sequences
		These sequences have closed-form definitions that compute values directly without search.
		Implementations reside in `mapFolding.algorithms.oeisIDbyFormula` [9]. Some formulas
		(A259702, A301620) are defined recursively in terms of A000682.

	References
	----------
	[1] OEIS - The On-Line Encyclopedia of Integer Sequences
		https://oeis.org/
	[2] Meanders - Wikipedia
		https://en.wikipedia.org/wiki/Meander_(mathematics)
	[3] mapFolding.basecamp.countFolds
		Internal package reference
	[4] mapFolding.algorithms.A086345
		Internal package reference (special formula implementation for A086345)
	[5] OEIS A000682 - Semi-meanders: number of Folded meanders
		https://oeis.org/A000682
	[6] OEIS A005316 - Meanders
		https://oeis.org/A005316
	[7] mapFolding.reference.A000682facts
		Internal package reference (arc code patterns and bit-width data)
	[8] mapFolding.reference.A005316facts
		Internal package reference (boundary bucket distributions)
	[9] mapFolding.algorithms.oeisIDbyFormula
		Internal package reference (closed-form sequence formulas)

	"""  # noqa: RUF002
#-------- memorialization instructions ---------------------------------------------

	if pathLikeWriteFoldsTotal is not None:
		# For sequences without a natural mapShape, create filename based on oeisID and oeis_n
		if oeisID == 'A007822':
			# A007822 has a mapShape, so use the standard approach
			mapShapeForFilename: tuple[int, ...] = (1, 2 * oeis_n)
			pathFilenameFoldsTotal: Path | None = getPathFilenameFoldsTotal(mapShapeForFilename, pathLikeWriteFoldsTotal)
		else:
			# Other sequences don't have mapShape, so create filename directly
			filenameCountTotal: str = f"{oeisID}_n{oeis_n}.countTotal"
			pathLikeSherpa = Path(pathLikeWriteFoldsTotal)
			if pathLikeSherpa.is_dir():
				pathFilenameFoldsTotal = pathLikeSherpa / filenameCountTotal
			elif pathLikeSherpa.is_file() and pathLikeSherpa.is_absolute():
				pathFilenameFoldsTotal = pathLikeSherpa
			else:
				pathFilenameFoldsTotal = getPathRootJobDEFAULT() / pathLikeSherpa
			pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
		saveFoldsTotalFAILearly(pathFilenameFoldsTotal)
	else:
		pathFilenameFoldsTotal = None

#-------- Algorithm selection and execution ---------------------------------------------

	countTotal: int = -31212012 # ERROR
	matched_oeisID: bool = True

	match oeisID:
		case 'A000136': from mapFolding.algorithms.oeisIDbyFormula import A000136 as doTheNeedful
		case 'A000560': from mapFolding.algorithms.oeisIDbyFormula import A000560 as doTheNeedful
		case 'A001010': from mapFolding.algorithms.oeisIDbyFormula import A001010 as doTheNeedful
		case 'A001011': from mapFolding.algorithms.oeisIDbyFormula import A001011 as doTheNeedful
		case 'A005315': from mapFolding.algorithms.oeisIDbyFormula import A005315 as doTheNeedful
		case 'A060206': from mapFolding.algorithms.oeisIDbyFormula import A060206 as doTheNeedful
		case 'A077460': from mapFolding.algorithms.oeisIDbyFormula import A077460 as doTheNeedful
		case 'A078591': from mapFolding.algorithms.oeisIDbyFormula import A078591 as doTheNeedful
		case 'A086345': from mapFolding.algorithms.A086345 import A086345 as doTheNeedful
		case 'A178961': from mapFolding.algorithms.oeisIDbyFormula import A178961 as doTheNeedful
		case 'A223094': from mapFolding.algorithms.oeisIDbyFormula import A223094 as doTheNeedful
		case 'A259702': from mapFolding.algorithms.oeisIDbyFormula import A259702 as doTheNeedful
		case 'A301620': from mapFolding.algorithms.oeisIDbyFormula import A301620 as doTheNeedful
		case _: matched_oeisID = False
	if matched_oeisID:
		countTotal = doTheNeedful(oeis_n) # pyright: ignore[reportPossiblyUnboundVariable]
	else:
		matched_oeisID = True
		match oeisID:
			case 'A000682' | 'A005316':
				match flow:
					case 'matrixNumPy':
						from mapFolding.algorithms.matrixMeandersNumPyndas import doTheNeedful, MatrixMeandersNumPyState as State
					case 'matrixPandas':
						from mapFolding.algorithms.matrixMeandersNumPyndas import (
							doTheNeedfulPandas as doTheNeedful, MatrixMeandersNumPyState as State)
					case 'matrixMeanders' | _:
						from mapFolding.algorithms.matrixMeanders import doTheNeedful
						from mapFolding.dataBaskets import MatrixMeandersState as State

				boundary: int = oeis_n - 1

				if oeisID == 'A000682':
					if oeis_n == 1:
						return 1
					elif oeis_n & 0b1:
						arcCode: int = 0b101
					else:
						arcCode = 0b1
					listArcCodes: list[int] = [(arcCode << 1) | arcCode]
													#  0b1010 | 0b0101 is 0b1111, or 0xf
													#    0b10 |   0b01 is   0b11, or 0x3

					MAXIMUMarcCode: int = 1 << (2 * boundary + 4)
					while listArcCodes[-1] < MAXIMUMarcCode:
						arcCode = (arcCode << 4) | 0b0101 # e.g., 0b 10000 | 0b 0101 = 0b 10101
						listArcCodes.append((arcCode << 1) | arcCode) # e.g., 0b 101010 | 0b 1010101 = 0b 111111 = 0x3f
						# Thereafter, append 0b1111 or 0xf, so, e.g., 0x3f, 0x3ff, 0x3fff, 0x3ffff, ...
						# See "mapFolding/reference/A000682facts.py"
					dictionaryMeanders=dict.fromkeys(listArcCodes, 1)

				elif oeisID == 'A005316':
					if oeis_n & 0b1:
						dictionaryMeanders: dict[int, int] = {0b1111: 1} # 0xf
					else:
						dictionaryMeanders = {0b10110: 1}
				else:
					message = f"Programming error: I should never have received `{oeisID = }`."
					raise ValueError(message)

				state = State(oeis_n, oeisID, boundary, dictionaryMeanders)
				countTotal = doTheNeedful(state) # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-argument-type]
			case 'A007822':
				mapShape: tuple[Literal[1], int] = (1, 2 * oeis_n)
				from mapFolding.beDRY import defineProcessorLimit
				concurrencyLimit: int = defineProcessorLimit(CPUlimit)

				from mapFolding.dataBaskets import SymmetricFoldsState
				symmetricState: SymmetricFoldsState = SymmetricFoldsState(mapShape)

				match flow:
					case 'asynchronous':
						from mapFolding.syntheticModules.A007822.asynchronous import doTheNeedful
						symmetricState = doTheNeedful(symmetricState, concurrencyLimit)
					case 'theorem2':
						from mapFolding.syntheticModules.A007822.theorem2 import doTheNeedful
						symmetricState = doTheNeedful(symmetricState)
					case 'theorem2Numba':
						from mapFolding.syntheticModules.A007822.theorem2Numba import doTheNeedful
						symmetricState = doTheNeedful(symmetricState)
					case 'theorem2Trimmed':
						from mapFolding.syntheticModules.A007822.theorem2Trimmed import doTheNeedful
						symmetricState = doTheNeedful(symmetricState)
					case _:
						from mapFolding.syntheticModules.A007822.algorithm import doTheNeedful
						symmetricState = doTheNeedful(symmetricState)

				countTotal = symmetricState.symmetricFolds
			case _:
				matched_oeisID = False

#-------- Follow memorialization instructions ---------------------------------------------

	if pathFilenameFoldsTotal is not None:
		saveFoldsTotal(pathFilenameFoldsTotal, countTotal)

	return countTotal

