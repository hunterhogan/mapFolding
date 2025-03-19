"""
Interface to The Online Encyclopedia of Integer Sequences (OEIS) for map folding sequences.

This module provides a comprehensive interface for accessing and utilizing integer sequences
from the OEIS that relate to map folding problems. It implements functionality to:

1. Retrieve sequence data from OEIS with local caching for performance
2. Map sequence indices to corresponding map shapes based on sequence definitions
3. Provide a command-line interface for sequence lookups
4. Execute map folding computations for sequence terms not available in OEIS

The module maintains a registry of implemented OEIS sequences (A001415-A001418, A195646)
with their metadata, known values, and functions to convert between sequence indices and
map dimensions. This allows the package to validate results against established mathematical
literature and extend sequences beyond their currently known terms.
"""
from collections.abc import Callable
from datetime import datetime, timedelta
from mapFolding.theSSOT import thePathPackage
from pathlib import Path
from typing import Any, Final, TYPE_CHECKING
import argparse
import pathlib
import random
import sys
import time
import urllib.request
import urllib.response
import warnings

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict

cacheDays = 7

"""
Section: make `settingsOEIS`"""

pathCache: Path = thePathPackage / ".cache"

class SettingsOEIS(TypedDict):
	description: str
	getMapShape: Callable[[int], tuple[int, ...]]
	offset: int
	valuesBenchmark: list[int]
	valuesKnown: dict[int, int]
	valuesTestParallelization: list[int]
	valuesTestValidation: list[int]
	valueUnknown: int

class SettingsOEIShardcodedValues(TypedDict):
	getMapShape: Callable[[int], tuple[int, ...]]
	valuesBenchmark: list[int]
	valuesTestParallelization: list[int]
	valuesTestValidation: list[int]

settingsOEIShardcodedValues: dict[str, SettingsOEIShardcodedValues] = {
	'A001415': {
		'getMapShape': lambda n: (2, n) if n >= 2 else (n, 2),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [*range(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],
	},
	'A001416': {
		'getMapShape': lambda n: (3, n) if n >= 3 else (n, 3),
		'valuesBenchmark': [9],
		'valuesTestParallelization': [*range(3, 5)],
		'valuesTestValidation': [random.randint(2, 6)],
	},
	'A001417': {
		'getMapShape': lambda n: tuple(2 for _dimension in range(n)),
		'valuesBenchmark': [6],
		'valuesTestParallelization': [*range(2, 4)],
		'valuesTestValidation': [random.randint(2, 4)],
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
		'valuesTestValidation': [random.randint(2, 4)],
	},
}

oeisIDsImplemented: Final[list[str]]  = sorted([oeisID.upper().strip() for oeisID in settingsOEIShardcodedValues.keys()])
"""Directly implemented OEIS IDs; standardized, e.g., 'A001415'."""

def validateOEISid(oeisIDcandidate: str) -> str:
	"""
	Validates an OEIS sequence ID against implemented sequences.

	If the provided ID is recognized within the application's implemented
	OEIS sequences, the function returns the verified ID in uppercase.
	Otherwise, a KeyError is raised indicating that the sequence is not
	directly supported.

	Parameters:
		oeisIDcandidate: The OEIS sequence identifier to validate.

	Returns:
		oeisID: The validated and possibly modified OEIS sequence ID, if recognized.

	Raises:
		KeyError: If the provided sequence ID is not directly implemented.
	"""
	if oeisIDcandidate in oeisIDsImplemented:
		return oeisIDcandidate
	else:
		oeisIDcleaned: str = str(oeisIDcandidate).upper().strip()
		if oeisIDcleaned in oeisIDsImplemented:
			return oeisIDcleaned
		else:
			raise KeyError(
				f"OEIS ID {oeisIDcandidate} is not directly implemented.\n"
				f"Available sequences:\n{_formatOEISsequenceInfo()}"
			)

def getFilenameOEISbFile(oeisID: str) -> str:
	oeisID = validateOEISid(oeisID)
	return f"b{oeisID[1:]}.txt"

def _parseBFileOEIS(OEISbFile: str, oeisID: str) -> dict[int, int]:
	"""
	Parses the content of an OEIS b-file for a given sequence ID.

	This function processes a multiline string representing an OEIS b-file and
	creates a dictionary mapping integer indices to their corresponding sequence
	values. The first line of the b-file is expected to contain a comment that
	matches the given sequence ID. If it does not match, a ValueError is raised.

	Parameters:
		OEISbFile: A multiline string representing an OEIS b-file.
		oeisID: The expected OEIS sequence identifier.
	Returns:
		OEISsequence: A dictionary where each key is an integer index `n` and
		each value is the sequence value `a(n)` corresponding to that index.
	Raises:
		ValueError: If the first line of the file does not indicate the expected
		sequence ID or if the content format is invalid.
	"""
	bFileLines: list[str] = OEISbFile.strip().splitlines()
	if not bFileLines.pop(0).startswith(f"# {oeisID}"):
		warnings.warn(f"Content does not match sequence {oeisID}")
		return {-1: -1}

	OEISsequence: dict[int, int] = {}
	for line in bFileLines:
		if line.startswith('#'):
			continue
		n, aOFn = map(int, line.split())
		OEISsequence[n] = aOFn
	return OEISsequence

def getOEISofficial(pathFilenameCache: pathlib.Path, url: str) -> None | str:
	tryCache: bool = False
	if pathFilenameCache.exists():
		fileAge: timedelta = datetime.now() - datetime.fromtimestamp(pathFilenameCache.stat().st_mtime)
		tryCache = fileAge < timedelta(days=cacheDays)

	oeisInformation: str | None = None
	if tryCache:
		try:
			oeisInformation = pathFilenameCache.read_text()
		except OSError:
			tryCache = False

	if not tryCache:
		httpResponse: urllib.response.addinfourl = urllib.request.urlopen(url)
		oeisInformation = httpResponse.read().decode('utf-8')
		pathFilenameCache.parent.mkdir(parents=True, exist_ok=True)
		pathFilenameCache.write_text(oeisInformation)

	if not oeisInformation:
		warnings.warn(f"Failed to retrieve OEIS sequence information for {pathFilenameCache.stem}.")

	return oeisInformation

def getOEISidValues(oeisID: str) -> dict[int, int]:
	"""
	Retrieves the specified OEIS sequence as a dictionary mapping integer indices
	to their corresponding values.
	This function checks for a cached local copy of the sequence data, using it if
	it has not expired. Otherwise, it fetches the sequence data from the OEIS
	website and writes it to the cache. The parsed data is returned as a dictionary
	mapping each index to its sequence value.

	Parameters:
		oeisID: The identifier of the OEIS sequence to retrieve.
	Returns:
		OEISsequence: A dictionary where each key is an integer index, `n`, and each
		value is the corresponding "a(n)" from the OEIS entry.
	Raises:
		ValueError: If the cached or downloaded file format is invalid.
		IOError: If there is an error reading from or writing to the local cache.
	"""

	pathFilenameCache: Path = pathCache / getFilenameOEISbFile(oeisID)
	url: str = f"https://oeis.org/{oeisID}/{getFilenameOEISbFile(oeisID)}"

	oeisInformation: None | str = getOEISofficial(pathFilenameCache, url)

	if oeisInformation:
		return _parseBFileOEIS(oeisInformation, oeisID)
	return {-1: -1}

def getOEISidInformation(oeisID: str) -> tuple[str, int]:
	oeisID = validateOEISid(oeisID)
	pathFilenameCache: Path = pathCache / f"{oeisID}.txt"
	url: str = f"https://oeis.org/search?q=id:{oeisID}&fmt=text"

	oeisInformation: None | str = getOEISofficial(pathFilenameCache, url)

	if not oeisInformation:
		return "Not found", -1

	listDescriptionDeconstructed: list[str] = []
	offset = None
	for ImaStr in oeisInformation.splitlines():
		ImaStr = ImaStr.strip() + "I am writing code to string parse machine readable data because in 2025, people can't even spell enteroperableity."
		secretCode, title, secretData =ImaStr.split(maxsplit=2)
		if secretCode == '%N' and title == oeisID:
			listDescriptionDeconstructed.append(secretData)
		if secretCode == '%O' and title == oeisID:
			offsetAsStr: str = secretData.split(',')[0]
			offset = int(offsetAsStr)
	if not listDescriptionDeconstructed:
		warnings.warn(f"No description found for {oeisID}")
		listDescriptionDeconstructed.append("No description found")
	if offset is None:
		warnings.warn(f"No offset found for {oeisID}")
		offset = -1
	description: str = ' '.join(listDescriptionDeconstructed)
	return description, offset

def makeSettingsOEIS() -> dict[str, SettingsOEIS]:
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
"""All values and settings for `oeisIDsImplemented`."""

"""
Section: private functions"""

def _formatHelpText() -> str:
	"""Format standardized help text for both CLI and interactive use."""
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
	"""Format information about available OEIS sequences for display or error messages."""
	return "\n".join(
		f"  {oeisID}: {settingsOEIS[oeisID]['description']}"
		for oeisID in oeisIDsImplemented
	)

"""
Section: public functions"""

def oeisIDfor_n(oeisID: str, n: int | Any) -> int:
	"""
	Calculate a(n) of a sequence from "The On-Line Encyclopedia of Integer Sequences" (OEIS).

	Parameters:
		oeisID: The ID of the OEIS sequence.
		n: A non-negative integer for which to calculate the sequence value.

	Returns:
		sequenceValue: a(n) of the OEIS sequence.

	Raises:
		ValueError: If n is negative.
		KeyError: If the OEIS sequence ID is not directly implemented.
	"""
	oeisID = validateOEISid(oeisID)

	if not isinstance(n, int) or n < 0:
		raise ValueError("`n` must be non-negative integer.")

	mapShape: tuple[int, ...] = settingsOEIS[oeisID]['getMapShape'](n)

	if n <= 1 or len(mapShape) < 2:
		offset: int = settingsOEIS[oeisID]['offset']
		if n < offset:
			raise ArithmeticError(f"OEIS sequence {oeisID} is not defined at n={n}.")
		foldsTotal: int = settingsOEIS[oeisID]['valuesKnown'][n]
		return foldsTotal
	from mapFolding.basecamp import countFolds
	return countFolds(mapShape)

def OEIS_for_n() -> None:
	"""Command-line interface for oeisIDfor_n."""
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
		print(oeisIDfor_n(argumentsCLI.oeisID, argumentsCLI.n), "distinct folding patterns.")
	except (KeyError, ValueError, ArithmeticError) as ERRORmessage:
		print(f"Error: {ERRORmessage}", file=sys.stderr)
		sys.exit(1)

	timeElapsed: float = time.perf_counter() - timeStart
	print(f"Time elapsed: {timeElapsed:.3f} seconds")

def clearOEIScache() -> None:
	"""Delete all cached OEIS sequence files."""
	if not pathCache.exists():
		print(f"Cache directory, {pathCache}, not found - nothing to clear.")
		return
	for oeisID in settingsOEIS:
		( pathCache / f"{oeisID}.txt" ).unlink(missing_ok=True)
		( pathCache / getFilenameOEISbFile(oeisID) ).unlink(missing_ok=True)
	print(f"Cache cleared from {pathCache}")

def getOEISids() -> None:
	"""Print all available OEIS sequence IDs that are directly implemented."""
	print(_formatHelpText())

if __name__ == "__main__":
	getOEISids()
