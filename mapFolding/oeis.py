"""
Mathematical validation and discovery through OEIS integration.

(AI generated docstring)

Complementing the unified computational interface, this module extends the map
folding ecosystem into the broader mathematical community through comprehensive
integration with the Online Encyclopedia of Integer Sequences (OEIS). This bridge
enables validation of computational results against established mathematical
knowledge while supporting the discovery of new sequence values through the
sophisticated computational pipeline.

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

from collections.abc import Callable
from datetime import datetime, timedelta, UTC
from functools import cache
from hunterMakesPy import writeStringToHere
from mapFolding import countFolds, packageSettings
from pathlib import Path
from typing import Any, Final, TypedDict
from urllib.request import urlopen
import argparse
import random
import sys
import time
import warnings

cacheDays = 30
"""Number of days to retain cached OEIS data before refreshing from the online source."""

pathCache: Path = packageSettings.pathPackage / ".cache"
"""Local directory path for storing cached OEIS sequence data and metadata."""

class SettingsOEIS(TypedDict):
	"""Complete configuration settings for a single OEIS sequence implementation.

	(AI generated docstring)

	This `TypedDict` defines the structure for storing all metadata, known values, and operational parameters
	needed to work with an OEIS sequence within the map folding context.

	"""

	description: str
	getMapShape: Callable[[int], tuple[int, ...]]
	offset: int
	valuesBenchmark: list[int]
	valuesKnown: dict[int, int]
	valuesTestParallelization: list[int]
	valuesTestValidation: list[int]
	valueUnknown: int

class SettingsOEIShardcodedValues(TypedDict):
	"""Hardcoded configuration values for OEIS sequences defined within the module.

	(AI generated docstring)

	This `TypedDict` contains the static configuration data that is embedded in the source code
	rather than retrieved from external sources.

	"""

	getMapShape: Callable[[int], tuple[int, ...]]
	valuesBenchmark: list[int]
	valuesTestParallelization: list[int]
	valuesTestValidation: list[int]

settingsOEIShardcodedValues: dict[str, SettingsOEIShardcodedValues] = {
	'A000136': {
		'getMapShape': lambda n: tuple(sorted([1, n])),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [*range(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],  # noqa: S311
	},
	'A001415': {
		'getMapShape': lambda n: tuple(sorted([2, n])),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [*range(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],  # noqa: S311
	},
	'A001416': {
		'getMapShape': lambda n: tuple(sorted([3, n])),
		'valuesBenchmark': [9],
		'valuesTestParallelization': [*range(3, 5)],
		'valuesTestValidation': [random.randint(2, 6)],  # noqa: S311
	},
	'A001417': {
		'getMapShape': lambda n: tuple(2 for _dimension in range(n)),
		'valuesBenchmark': [6],
		'valuesTestParallelization': [*range(2, 4)],
		'valuesTestValidation': [random.randint(2, 4)],  # noqa: S311
	},
	'A195646': {
		'getMapShape': lambda n: tuple(3 for _dimension in range(n)),
		'valuesBenchmark': [3],
		'valuesTestParallelization': [*range(2, 3)],
		'valuesTestValidation': [2],
	},
	'A001418': {
		'getMapShape': lambda n: (n, n),
		'valuesBenchmark': [5],
		'valuesTestParallelization': [*range(2, 4)],
		'valuesTestValidation': [random.randint(2, 4)],  # noqa: S311
	},
}
"""
Registry of hardcoded OEIS sequence configurations implemented in this module.

Each key is a standardized OEIS sequence identifier (e.g., 'A001415'), and each value contains
the static configuration needed to work with that sequence, including the mapping function from
sequence index to map shape and various test parameter sets.
"""

oeisIDsImplemented: Final[list[str]]  = sorted([oeisID.upper().strip() for oeisID in settingsOEIShardcodedValues])
"""Directly implemented OEIS IDs; standardized, e.g., 'A001415'."""

def validateOEISid(oeisIDcandidate: str) -> str:
	"""Validate an OEIS sequence ID against implemented sequences.

	(AI generated docstring)

	If the provided ID is recognized within the application's implemented OEIS sequences, the function returns the
	verified ID in uppercase. Otherwise, a KeyError is raised indicating that the sequence is not directly supported.

	Parameters
	----------
	oeisIDcandidate : str
		The OEIS sequence identifier to validate.

	Returns
	-------
	str
		The validated and possibly modified OEIS sequence ID, if recognized.

	Raises
	------
	KeyError
		If the provided sequence ID is not directly implemented.

	"""
	if oeisIDcandidate in oeisIDsImplemented:
		return oeisIDcandidate
	else:
		oeisIDcleaned: str = str(oeisIDcandidate).upper().strip()
		if oeisIDcleaned in oeisIDsImplemented:
			return oeisIDcleaned
		else:
			message = (
				f"OEIS ID {oeisIDcandidate} is not directly implemented.\n"
				f"Available sequences:\n{_formatOEISsequenceInfo()}"
			)
			raise KeyError(
				message
			)

def getFilenameOEISbFile(oeisID: str) -> str:
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
	oeisID = validateOEISid(oeisID)
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

def getOEISofficial(pathFilenameCache: Path, url: str) -> None | str:
	"""Retrieve OEIS sequence data from cache or online source with intelligent caching.

	(AI generated docstring)

	This function implements a caching strategy that prioritizes local cached data when it exists and
	has not expired. Fresh data is retrieved from the OEIS website when the cache is stale or missing,
	and the cache is updated for future use.

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

	Notes
	-----
	Cache expiration is controlled by the module-level `cacheDays` variable. The function validates
	URL schemes and issues warnings for failed retrievals.

	"""
	tryCache: bool = False
	if pathFilenameCache.exists():
		fileAge: timedelta = datetime.now(tz=UTC) - datetime.fromtimestamp(pathFilenameCache.stat().st_mtime, tz=UTC)
		tryCache = fileAge < timedelta(days=cacheDays)

	oeisInformation: str | None = None
	if tryCache:
		try:
			oeisInformation = pathFilenameCache.read_text(encoding="utf-8")
		except OSError:
			tryCache = False

	if not tryCache:
		if not url.startswith(("http:", "https:")):
			raise ValueError("URL must start with 'http:' or 'https:'")
		with urlopen(url) as response:
			oeisInformationRaw = response.read().decode('utf-8')
		oeisInformation = str(oeisInformationRaw)
		writeStringToHere(oeisInformation, pathFilenameCache)

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
	pathFilenameCache: Path = pathCache / getFilenameOEISbFile(oeisID)
	url: str = f"https://oeis.org/{oeisID}/{getFilenameOEISbFile(oeisID)}"

	oeisInformation: None | str = getOEISofficial(pathFilenameCache, url)

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
	oeisID = validateOEISid(oeisID)
	pathFilenameCache: Path = pathCache / f"{oeisID}.txt"
	url: str = f"https://oeis.org/search?q=id:{oeisID}&fmt=text"

	oeisInformation: None | str = getOEISofficial(pathFilenameCache, url)

	if not oeisInformation:
		return "Not found", -1
	listDescriptionDeconstructed: list[str] = []
	offset = None
	for lineOEIS in oeisInformation.splitlines():
		lineOEIS = lineOEIS.strip()  # noqa: PLW2901
		if not lineOEIS or len(lineOEIS.split()) < 3:  # noqa: PLR2004
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

def makeSettingsOEIS() -> dict[str, SettingsOEIS]:
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
	settingsTarget: dict[str, SettingsOEIS] = {}
	for oeisID in oeisIDsImplemented:
		valuesKnownSherpa: dict[int, int] = getOEISidValues(oeisID)
		descriptionSherpa, offsetSherpa = getOEISidInformation(oeisID)
		settingsTarget[oeisID] = SettingsOEIS(
			description=descriptionSherpa,
			offset=offsetSherpa,
			getMapShape=settingsOEIShardcodedValues[oeisID]['getMapShape'],
			valuesBenchmark=settingsOEIShardcodedValues[oeisID]['valuesBenchmark'],
			valuesTestParallelization=settingsOEIShardcodedValues[oeisID]['valuesTestParallelization'],
			valuesTestValidation=settingsOEIShardcodedValues[oeisID]['valuesTestValidation'] + list(range(offsetSherpa, 2)),
			valuesKnown=valuesKnownSherpa,
			valueUnknown=max(valuesKnownSherpa.keys(), default=0) + 1
		)
	return settingsTarget

settingsOEIS: dict[str, SettingsOEIS] = makeSettingsOEIS()
"""
Complete settings and metadata for all implemented OEIS sequences.

This dictionary contains the comprehensive configuration for each OEIS sequence supported by the module,
including known values retrieved from OEIS, mathematical descriptions, offset information, and all
operational parameters needed for computation and testing. The dictionary is populated by combining
hardcoded configurations with dynamically retrieved OEIS data during module initialization.
"""

@cache
def makeDictionaryFoldsTotalKnown() -> dict[tuple[int, ...], int]:
	"""Create a cached lookup dictionary mapping map shapes to their known folding totals.

	(AI generated docstring)

	This function processes all known sequence values from implemented OEIS sequences and creates
	a unified dictionary that maps map dimension tuples to their corresponding folding totals. The
	resulting dictionary enables rapid lookup of known values without requiring knowledge of which
	specific OEIS sequence contains the data.

	Returns
	-------
	dictionaryMapDimensionsToFoldsTotalKnown : dict[tuple[int, ...], int]
		A dictionary where keys are tuples representing map shapes and values are the total number
		of distinct folding patterns for those shapes.

	"""
	dictionaryMapDimensionsToFoldsTotalKnown: dict[tuple[int, ...], int] = {}

	for settings in settingsOEIS.values():
		sequence = settings['valuesKnown']

		for n, foldingsTotal in sequence.items():
			mapShape = settings['getMapShape'](n)
			mapShape = tuple(mapShape)
			dictionaryMapDimensionsToFoldsTotalKnown[mapShape] = foldingsTotal
	return dictionaryMapDimensionsToFoldsTotalKnown

def getFoldsTotalKnown(mapShape: tuple[int, ...]) -> int:
	"""Retrieve the known total number of distinct folding patterns for a given map shape.

	(AI generated docstring)

	This function provides rapid access to precalculated folding totals from OEIS sequences without
	requiring computation. It serves as a validation reference for algorithm results and enables
	quick lookup of known values across all implemented sequences.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.

	Returns
	-------
	foldingsTotal : int
		The known total number of distinct folding patterns for the given map shape,
		or -1 if the map shape does not match any known values in the OEIS sequences.

	Notes
	-----
	The function uses a cached dictionary for efficient retrieval without repeatedly processing
	OEIS data. Map shapes are matched exactly as provided without internal sorting or normalization.

	"""
	lookupFoldsTotal = makeDictionaryFoldsTotalKnown()
	return lookupFoldsTotal.get(tuple(mapShape), -1)

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
	exampleOEISid: str = oeisIDsImplemented[0]
	exampleN: int = settingsOEIS[exampleOEISid]['valuesTestValidation'][-1]

	return (
		"\nAvailable OEIS sequences:\n"
		f"{_formatOEISsequenceInfo()}\n"
		"\nUsage examples:\n"
		"  Command line:\n"
		f"	OEIS_for_n {exampleOEISid} {exampleN}\n"
		"  Python:\n"
		"	from mapFolding import oeisIDfor_n\n"
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
		f"  {oeisID}: {settingsOEIS[oeisID]['description']}"
		for oeisID in oeisIDsImplemented
	)

def oeisIDfor_n(oeisID: str, n: int | Any) -> int:
	"""Calculate the value a(n) for a specified OEIS sequence and index.

	(AI generated docstring)

	This function serves as the primary interface for computing OEIS sequence values within the map folding
	context. For small values or values within the known range, it returns cached OEIS data. For larger
	values, it computes the result using the map folding algorithm with the appropriate map shape derived
	from the sequence definition.

	Parameters
	----------
	oeisID : str
		The identifier of the OEIS sequence to evaluate.
	n : int | Any
		A non-negative integer index for which to calculate the sequence value.

	Returns
	-------
	int
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
	oeisID = validateOEISid(oeisID)

	if not isinstance(n, int) or n < 0:
		message = f"I received `{n = }` in the form of `{type(n) = }`, but it must be non-negative integer in the form of `{int}`."
		raise ValueError(message)

	mapShape: tuple[int, ...] = settingsOEIS[oeisID]['getMapShape'](n)

	if n <= 1 or len(mapShape) < 2:  # noqa: PLR2004
		offset: int = settingsOEIS[oeisID]['offset']
		if n < offset:
			message = f"OEIS sequence {oeisID} is not defined at {n = }."
			raise ArithmeticError(message)
		foldsTotal: int = settingsOEIS[oeisID]['valuesKnown'][n]
		return foldsTotal
	return countFolds(mapShape)

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

def clearOEIScache() -> None:
	"""Delete all cached OEIS sequence files from the local cache directory.

	(AI generated docstring)

	This function removes all cached OEIS data files, including both sequence value files (b-files)
	and metadata files, forcing fresh retrieval from the OEIS website on the next access. This is
	useful for clearing stale cache data or troubleshooting network-related issues.

	The function safely handles missing files and provides user feedback about the cache clearing
	operation. If the cache directory does not exist, an informative message is displayed.

	"""
	if not pathCache.exists():
		print(f"Cache directory, {pathCache}, not found - nothing to clear.")  # noqa: T201
		return
	for oeisID in settingsOEIS:
		( pathCache / f"{oeisID}.txt" ).unlink(missing_ok=True)
		( pathCache / getFilenameOEISbFile(oeisID) ).unlink(missing_ok=True)
	print(f"Cache cleared from {pathCache}")  # noqa: T201

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

if __name__ == "__main__":
	getOEISids()
