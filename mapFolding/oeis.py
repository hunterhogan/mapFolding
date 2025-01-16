"""Everything implementing the The Online Encyclopedia of Integer Sequences (OEIS);
_only_ things that implement _only_ the OEIS."""
from datetime import datetime, timedelta
from mapFolding import countFolds
from typing import TYPE_CHECKING, List, Callable, Dict, Final
import pathlib
import random
import sys
import time
import urllib.request
import urllib.response

if TYPE_CHECKING:
    from typing import TypedDict
else:
    TypedDict = dict

"""
Section: make `settingsOEIS`"""
class SettingsOEIS(TypedDict):
    # I would prefer to load description dynamically from OEIS, but it's a pita for me
    # to learn how to efficiently implement right now.
    description: str
    getDimensions: Callable[[int], List[int]]
    valuesBenchmark: List[int]
    valuesKnown: Dict[int, int]
    valuesTestValidation: List[int]
    valueUnknown: int

settingsOEIShardcodedValues = {
    'A001415': {
        'description': 'Number of ways of folding a 2 X n strip of stamps.',
        'getDimensions': lambda n: sorted([2, n]),
        'valuesBenchmark': [14],
        'valuesTestValidation': [0, 1, random.randint(2, 9)],
    },
    'A001416': {
        'description': 'Number of ways of folding a 3 X n strip of stamps.',
        'getDimensions': lambda n: sorted([3, n]),
        'valuesBenchmark': [9],
        'valuesTestValidation': [0, 1, random.randint(2, 6)],
    },
    'A001417': {
        'description': 'Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.',
        'getDimensions': lambda n: [2] * n,
        'valuesBenchmark': [6],
        'valuesTestValidation': [0, 1, random.randint(2, 4)],
    },
    'A195646': {
        'description': 'Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.',
        'getDimensions': lambda n: [3] * n,
        'valuesBenchmark': [3],
        'valuesTestValidation': [0, 1, 2],
    },
    'A001418': {
        'description': 'Number of ways of folding an n X n sheet of stamps.',
        'getDimensions': lambda n: [n, n],
        'valuesBenchmark': [5],
        # offset 1: hypothetically, if I were to load the offset from OEIS, I could use it to
        # determine if a sequence is defined at n=0, which would affect, for example, the valuesTestValidation.
        'valuesTestValidation': [1, random.randint(2, 4)],
    },
}

oeisIDsImplemented: Final[List[str]]  = sorted([oeisID.upper().strip() for oeisID in settingsOEIShardcodedValues.keys()])
"""Directly implemented OEIS IDs; standardized, e.g., 'A001415'."""

def _validateOEISid(oeisIDcandidate: str):
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
        oeisIDcleaned = str(oeisIDcandidate).upper().strip()
        if oeisIDcleaned in oeisIDsImplemented:
            return oeisIDcleaned
        else:
            raise KeyError(
                f"OEIS ID {oeisIDcandidate} is not directly implemented.\n"
                f"Available sequences:\n{_formatOEISsequenceInfo()}"
            )

def _getFilenameOEISbFile(oeisID: str) -> str:
    oeisID = _validateOEISid(oeisID)
    return f"b{oeisID[1:]}.txt"

def _parseBFileOEIS(OEISbFile: str, oeisID: str) -> Dict[int, int]:
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
    bFileLines = OEISbFile.strip().splitlines()
    # The first line has the sequence ID
    if not bFileLines.pop(0).startswith(f"# {oeisID}"):
        raise ValueError(f"Content does not match sequence {oeisID}")

    OEISsequence = {}
    for line in bFileLines:
        if line.startswith('#'):
            continue
        n, aOFn = map(int, line.split())
        OEISsequence[n] = aOFn
    return OEISsequence

try:
    _pathCache = pathlib.Path(__file__).parent / ".cache"
except NameError:
    _pathCache = pathlib.Path.home() / ".mapFoldingCache"

def _getOEISidValues(oeisID: str) -> Dict[int, int]:
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

    pathFilenameCache = _pathCache / _getFilenameOEISbFile(oeisID)
    cacheDays = 7

    tryCache = False
    if pathFilenameCache.exists():
        fileAge = datetime.now() - datetime.fromtimestamp(pathFilenameCache.stat().st_mtime)
        tryCache = fileAge < timedelta(days=cacheDays)

    if tryCache:
        try:
            OEISbFile = pathFilenameCache.read_text()
            return _parseBFileOEIS(OEISbFile, oeisID)
        except (ValueError, IOError):
            tryCache = False

    urlOEISbFile = f"https://oeis.org/{oeisID}/{_getFilenameOEISbFile(oeisID)}"
    httpResponse: urllib.response.addinfourl = urllib.request.urlopen(urlOEISbFile)
    OEISbFile = httpResponse.read().decode('utf-8')

    if not tryCache:
        pathFilenameCache.parent.mkdir(parents=True, exist_ok=True)
        pathFilenameCache.write_text(OEISbFile)

    return _parseBFileOEIS(OEISbFile, oeisID)

def makeSettingsOEIS() -> Dict[str, SettingsOEIS]:
    """
    Creates a dictionary mapping OEIS IDs to their corresponding settings.

    This function initializes settings for each implemented OEIS sequence by combining
    hardcoded values with dynamically retrieved OEIS sequence values.

    Returns:
        Dict[str, SettingsOEIS]: A dictionary where:
            - Keys are OEIS sequence IDs (str)
            - Values are SettingsOEIS objects containing:
                - description: Text description of the sequence
                - getDimensions: Function to get dimensions
                - valuesBenchmark: Benchmark values
                - valuesKnown: Known values from OEIS
                - valuesTestValidation: Values for test validation
                - valueUnknown: First unknown value in sequence

    Note:
        Relies on global variables:
        - oeisIDsImplemented: List of implemented OEIS sequence IDs
        - settingsOEIShardcodedValues: Dictionary of hardcoded settings per sequence
    """
    settingsTarget = {}
    for oeisID in oeisIDsImplemented:
        valuesKnownSherpa = _getOEISidValues(oeisID)
        settingsTarget[oeisID] = SettingsOEIS(
            description = settingsOEIShardcodedValues[oeisID]['description'],
            getDimensions = settingsOEIShardcodedValues[oeisID]['getDimensions'],
            valuesBenchmark = settingsOEIShardcodedValues[oeisID]['valuesBenchmark'],
            valuesKnown = valuesKnownSherpa,
            valuesTestValidation = settingsOEIShardcodedValues[oeisID]['valuesTestValidation'],
            valueUnknown = max(valuesKnownSherpa.keys(), default=0) + 1
        )
    return settingsTarget

settingsOEIS: Dict[str, SettingsOEIS] = makeSettingsOEIS()
"""All values and settings for `oeisIDsImplemented`."""

"""
Section: private functions"""

def _formatHelpText() -> str:
    """Format standardized help text for both CLI and interactive use."""
    exampleOEISid = oeisIDsImplemented[0]
    exampleN = settingsOEIS[exampleOEISid]['valuesTestValidation'][-1]

    return (
        "\nAvailable OEIS sequences:\n"
        f"{_formatOEISsequenceInfo()}\n"
        "\nUsage examples:\n"
        "  Command line:\n"
        f"    OEIS_for_n {exampleOEISid} {exampleN}\n"
        "  Python:\n"
        "    from mapFolding import oeisIDfor_n\n"
        f"    foldsTotal = oeisIDfor_n('{exampleOEISid}', {exampleN})"
    )

def _formatOEISsequenceInfo() -> str:
    """Format information about available OEIS sequences for display or error messages."""
    return "\n".join(
        f"  {oeisID}: {settingsOEIS[oeisID]['description']}"
        for oeisID in oeisIDsImplemented
    )

"""
Section: public functions"""

def oeisIDfor_n(oeisID: str, n: int) -> int:
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
    oeisID = _validateOEISid(oeisID)

    if not isinstance(n, int) or n < 0:
        raise ValueError("`n` must be non-negative integer.")

    listDimensions = settingsOEIS[oeisID]['getDimensions'](n)

    if n <= 1 or len(listDimensions) < 2:
        foldsTotal = settingsOEIS[oeisID]['valuesKnown'].get(n, None)
        if foldsTotal is not None:
            return foldsTotal
        else:
            raise ArithmeticError(f"OEIS sequence {oeisID} is not defined at n={n}.")

    return countFolds(listDimensions)

def OEIS_for_n() -> None:
    """Command-line interface for oeisIDfor_n."""
    import argparse

    parserCLI = argparse.ArgumentParser(
        description="Calculate a(n) for an OEIS sequence.",
        epilog=_formatHelpText(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parserCLI.add_argument('oeisID', help="OEIS sequence identifier")
    parserCLI.add_argument('n', type=int, help="Calculate a(n) for this n")

    argumentsCLI = parserCLI.parse_args()

    timeStart = time.perf_counter()

    try:
        print(oeisIDfor_n(argumentsCLI.oeisID, argumentsCLI.n), "distinct folding patterns.")
    except (KeyError, ValueError, ArithmeticError) as ERRORmessage:
        print(f"Error: {ERRORmessage}", file=sys.stderr)
        sys.exit(1)

    timeElapsed = time.perf_counter() - timeStart
    print(f"Time elapsed: {timeElapsed:.3f} seconds")

def clearOEIScache() -> None:
    """Delete all cached OEIS sequence files."""
    if not _pathCache.exists():
        print(f"Cache directory, {_pathCache}, not found - nothing to clear.")
        return
    else:
        for oeisID in settingsOEIS:
            pathFilenameCache = _pathCache / _getFilenameOEISbFile(oeisID)
            pathFilenameCache.unlink(missing_ok=True)

    print(f"Cache cleared from {_pathCache}")

def getOEISids() -> None:
    """Print all available OEIS sequence IDs that are directly implemented."""
    print(_formatHelpText())

if __name__ == "__main__":
    getOEISids()
