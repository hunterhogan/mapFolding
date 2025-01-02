from datetime import datetime, timedelta
from mapFolding import foldings
import pathlib
import random
import typing
import urllib.request
import urllib.response

if typing.TYPE_CHECKING:
    from typing import TypedDict
else:
    TypedDict = dict

class SettingsOEISsequence(TypedDict):
    # I would prefer to load description dynamically from OEIS, but it's a pita for me
    # to learn how to efficiently implement right now.
    description: str 
    getDimensions: typing.Callable[[int], typing.List[int]]
    valuesBenchmark: typing.List[int]
    valuesKnown: typing.Dict[int, int]
    valuesTestValidation: typing.List[int]
    valueUnknown: int

try:
    _pathCache = pathlib.Path(__file__).parent / ".cache"
except NameError:
    _pathCache = pathlib.Path.home() / ".mapFoldingCache"

_formatFilenameCache = "{oeisID}.txt"

# NOTE: not DRY, and I'm annoyed and frustrated. I cannot figure out how to not duplicate this information here and in the dictionary.
# I suspect there is a better paradigm to accomplish this.
# 1. I am acting as if `oeisID` is a member of OEISsequenceID, but that isn't really true.
# 2. I would like to define `oeisID` as always being upper case, but I don't
# have an obvious way to do that in this system.
OEISsequenceID = typing.Literal['A001415', 'A001416', 'A001417', 'A195646', 'A001418']

settingsOEISsequences: typing.Dict[OEISsequenceID, SettingsOEISsequence] = {
    'A001415': {
        'description': 'Number of ways of folding a 2 X n strip of stamps.',
        'getDimensions': lambda n: sorted([2, n]),
        'valuesBenchmark': [12],
        'valuesTestValidation': [0, 1, random.randint(2, 9)],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
    'A001416': {
        'description': 'Number of ways of folding a 3 X n strip of stamps.',
        'getDimensions': lambda n: sorted([3, n]),
        'valuesBenchmark': [8],
        'valuesTestValidation': [0, 1, random.randint(2, 6)],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
    'A001417': {
        'description': 'Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.',
        'getDimensions': lambda n: [2] * n,
        'valuesBenchmark': [5],
        'valuesTestValidation': [0, 1, random.randint(2, 4)],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
    'A195646': {
        'description': 'Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.',
        'getDimensions': lambda n: [3] * n,
        'valuesBenchmark': [3],
        'valuesTestValidation': [0, 1, 2],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
    'A001418': {
        'description': 'Number of ways of folding an n X n sheet of stamps.',
        'getDimensions': lambda n: [n, n],
        'valuesBenchmark': [5],
        # offset 1: hypothetically, if I were to load the offset from OEIS, I could use it to
        # determine if a sequence is defined at n=0, which would affect, for example, the valuesTestValidation.
        'valuesTestValidation': [1, random.randint(2, 4)],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
}

def oeisSequence_aOFn(oeisID: OEISsequenceID, n: int) -> int:
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

    listDimensions = settingsOEISsequences[oeisID]['getDimensions'](n)

    if n <= 1 or len(listDimensions) < 2:
        foldingsTotal = settingsOEISsequences[oeisID]['valuesKnown'].get(n, None)
        if foldingsTotal is not None:
            return foldingsTotal
        else:
            raise ArithmeticError(f"OEIS sequence {oeisID} is not defined at n={n}.")

    return foldings(listDimensions)

def _validateOEISid(oeisID: typing.Union[str, OEISsequenceID]) -> OEISsequenceID:
    """
    Validates an OEIS sequence ID against implemented sequences.

    If the provided ID is recognized within the application's implemented
    OEIS sequences, the function returns the verified ID in uppercase.
    Otherwise, a KeyError is raised indicating that the sequence is not
    directly supported.

    Parameters:
        oeisID: The OEIS sequence identifier to validate.

    Returns:
        oeisID: The validated and possibly modified OEIS sequence ID, if recognized.

    Raises:
        KeyError: If the provided sequence ID is not directly implemented.
    """
    if oeisID in typing.get_args(OEISsequenceID):
        return oeisID # type: ignore # mypy doesn't understand that oeisID is now a valid OEISsequenceID
                        # and/or I don't know how to tell it that it is
    else:
        oeisIDcleaned = oeisID.upper().strip()
        if oeisIDcleaned in settingsOEISsequences:
            return oeisIDcleaned
        else:
            raise KeyError(f"Sequence {oeisID} is not directly implemented in mapFoldings. The directly implemented sequences are {typing.get_args(OEISsequenceID)}.\nFor maps with at least two getDimensions, try `mapFolding.foldings()`.")

def _parseBFileOEIS(OEISbFile: str, oeisID: OEISsequenceID) -> typing.Dict[int, int]:
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

def _getOEISidValues(oeisID: OEISsequenceID) -> typing.Dict[int, int]:
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
        OEISsequence: A dictionary where each key is an integer index and each
        value is the corresponding sequence term from the specified OEIS entry.
    Raises:
        ValueError: If the cached or downloaded file format is invalid.
        IOError: If there is an error reading from or writing to the local cache.
    """

    pathFilenameCache = _pathCache / _formatFilenameCache.format(oeisID=oeisID)
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

    # urlOEISbFile = _format_urlOEISbFile.format(oeisID=oeisID)
    urlOEISbFile = f"https://oeis.org/{oeisID}/b{oeisID[1:]}.txt"
    httpResponse: urllib.response.addinfourl = urllib.request.urlopen(urlOEISbFile)
    OEISbFile = httpResponse.read().decode('utf-8')

    # Ensure cache directory exists
    if not tryCache:
        pathFilenameCache.parent.mkdir(parents=True, exist_ok=True)
        pathFilenameCache.write_text(OEISbFile)

    return _parseBFileOEIS(OEISbFile, oeisID)

for oeisID in settingsOEISsequences:
    settingsOEISsequences[oeisID]['valuesKnown'] = _getOEISidValues(oeisID)
    settingsOEISsequences[oeisID]['valueUnknown'] = max(settingsOEISsequences[oeisID]['valuesKnown'].values()) + 1

def getOEISids() -> None:
    """Print all available OEIS sequence IDs that are directly implemented."""
    print("\nAvailable OEIS sequences:")
    for oeisID in typing.get_args(OEISsequenceID):
        print(f"  {oeisID}: {settingsOEISsequences[oeisID]['description']}")
    print("\nUsage example:")
    print("  from mapFolding import oeisSequence_aOFn")
    print("  foldingsTotal = oeisSequence_aOFn('A001415', 5)")

if __name__ == "__main__":
    getOEISids()
