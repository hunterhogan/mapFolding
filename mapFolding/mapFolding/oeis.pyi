import pathlib
from collections.abc import Callable as Callable
from mapFolding import countFolds as countFolds
from mapFolding.theSSOT import pathPackage as pathPackage
from pathlib import Path
from typing import Any, Final, TypedDict

cacheDays: int
pathCache: Path

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

settingsOEIShardcodedValues: dict[str, SettingsOEIShardcodedValues]
oeisIDsImplemented: Final[list[str]]

def validateOEISid(oeisIDcandidate: str) -> str:
    """
\tValidates an OEIS sequence ID against implemented sequences.

\tIf the provided ID is recognized within the application's implemented
\tOEIS sequences, the function returns the verified ID in uppercase.
\tOtherwise, a KeyError is raised indicating that the sequence is not
\tdirectly supported.

\tParameters:
\t\toeisIDcandidate: The OEIS sequence identifier to validate.

\tReturns:
\t\toeisID: The validated and possibly modified OEIS sequence ID, if recognized.

\tRaises:
\t\tKeyError: If the provided sequence ID is not directly implemented.
\t"""
def getFilenameOEISbFile(oeisID: str) -> str: ...
def _parseBFileOEIS(OEISbFile: str, oeisID: str) -> dict[int, int]:
    """
\tParses the content of an OEIS b-file for a given sequence ID.

\tThis function processes a multiline string representing an OEIS b-file and
\tcreates a dictionary mapping integer indices to their corresponding sequence
\tvalues. The first line of the b-file is expected to contain a comment that
\tmatches the given sequence ID. If it does not match, a ValueError is raised.

\tParameters:
\t\tOEISbFile: A multiline string representing an OEIS b-file.
\t\toeisID: The expected OEIS sequence identifier.
\tReturns:
\t\tOEISsequence: A dictionary where each key is an integer index `n` and
\t\teach value is the sequence value `a(n)` corresponding to that index.
\tRaises:
\t\tValueError: If the first line of the file does not indicate the expected
\t\tsequence ID or if the content format is invalid.
\t"""
def getOEISofficial(pathFilenameCache: pathlib.Path, url: str) -> None | str: ...
def getOEISidValues(oeisID: str) -> dict[int, int]:
    '''
\tRetrieves the specified OEIS sequence as a dictionary mapping integer indices
\tto their corresponding values.
\tThis function checks for a cached local copy of the sequence data, using it if
\tit has not expired. Otherwise, it fetches the sequence data from the OEIS
\twebsite and writes it to the cache. The parsed data is returned as a dictionary
\tmapping each index to its sequence value.

\tParameters:
\t\toeisID: The identifier of the OEIS sequence to retrieve.
\tReturns:
\t\tOEISsequence: A dictionary where each key is an integer index, `n`, and each
\t\tvalue is the corresponding "a(n)" from the OEIS entry.
\tRaises:
\t\tValueError: If the cached or downloaded file format is invalid.
\t\tIOError: If there is an error reading from or writing to the local cache.
\t'''
def getOEISidInformation(oeisID: str) -> tuple[str, int]: ...
def makeSettingsOEIS() -> dict[str, SettingsOEIS]: ...

settingsOEIS: dict[str, SettingsOEIS]

def _formatHelpText() -> str:
    """Format standardized help text for both CLI and interactive use."""
def _formatOEISsequenceInfo() -> str:
    """Format information about available OEIS sequences for display or error messages."""
def oeisIDfor_n(oeisID: str, n: int | Any) -> int:
    '''
\tCalculate a(n) of a sequence from "The On-Line Encyclopedia of Integer Sequences" (OEIS).

\tParameters:
\t\toeisID: The ID of the OEIS sequence.
\t\tn: A non-negative integer for which to calculate the sequence value.

\tReturns:
\t\tsequenceValue: a(n) of the OEIS sequence.

\tRaises:
\t\tValueError: If n is negative.
\t\tKeyError: If the OEIS sequence ID is not directly implemented.
\t'''
def OEIS_for_n() -> None:
    """Command-line interface for oeisIDfor_n."""
def clearOEIScache() -> None:
    """Delete all cached OEIS sequence files."""
def getOEISids() -> None:
    """Print all available OEIS sequence IDs that are directly implemented."""
