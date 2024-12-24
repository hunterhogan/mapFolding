import pathlib
import random
import urllib.request
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, get_args

if TYPE_CHECKING:
    from typing import TypedDict
else:
    TypedDict = dict

class SettingsOEISsequence(TypedDict):
    # I would prefer to load description dynamically from OEIS, but it's a pita for me and my skill set right now.
    description: str 
    dimensions: Callable[[int], List[int]]
    benchmarkValues: List[int]
    testValuesValidation: List[int]
    valuesKnown: Dict[int, int]
    valueUnknown: int

try:
    _pathCache = pathlib.Path(__file__).parent / ".cache"
except NameError:
    _pathCache = pathlib.Path.home() / ".mapFoldingCache"

_formatFilenameCache = "{oeisID}.txt"

# NOTE: not DRY, and I'm annoyed and frustrated. I cannot figure out how to not duplicate this information here and in the dictionary.
OEISsequenceID = Literal['A001415', 'A001416', 'A001417', 'A195646', 'A001418'] 

settingsOEISsequences: Dict[OEISsequenceID, SettingsOEISsequence] = {
    'A001415': {
        'description': 'Number of ways of folding a 2 X n strip of stamps.',
        'dimensions': lambda n: [2, n],
        'benchmarkValues': [12],
        'testValuesValidation': [0, 1, random.randint(2, 9)],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
    'A001416': {
        'description': 'Number of ways of folding a 3 X n strip of stamps.',
        'dimensions': lambda n: [3, n],
        'benchmarkValues': [8],
        'testValuesValidation': [0, 1, random.randint(2, 6)],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
    'A001417': {
        'description': 'Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.',
        'dimensions': lambda n: [2] * n,
        'benchmarkValues': [5],
        'testValuesValidation': [0, 1, random.randint(2, 4)],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
    'A195646': {
        'description': 'Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.',
        'dimensions': lambda n: [3] * n,
        'benchmarkValues': [3],
        'testValuesValidation': [0, 1, 2],
        'valueUnknown': -1,
        'valuesKnown': {-1:-1},
    },
    'A001418': { # offset 1: hypothetically, if I were to load the offset from OEIS, I could use it to determine if a sequence is defined at n=0.
        'description': 'Number of ways of folding an n X n sheet of stamps.',
        'dimensions': lambda n: [n, n],
        'benchmarkValues': [5],
        'testValuesValidation': [1, random.randint(2, 4)],
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
    listDimensions = settingsOEISsequences[oeisID]['dimensions'](n)

    if n <= 1 or len(listDimensions) < 2:
        foldingsTotal = settingsOEISsequences[oeisID]['valuesKnown'].get(n, None)
        if foldingsTotal is not None:
            return foldingsTotal
        else:
            raise ArithmeticError(f"Sequence {oeisID} is not defined at {n=}.")

    from mapFolding import foldings
    return foldings(listDimensions)

def _validateOEISid(oeisID):
    if oeisID in get_args(OEISsequenceID):
        return oeisID
    else:
        oeisIDcleaned = oeisID.upper().strip() 
        if oeisIDcleaned in settingsOEISsequences:
            return oeisIDcleaned
        else:
            raise KeyError(f"Sequence {oeisID} is not directly implemented in mapFoldings. The directly implemented sequences are {get_args(OEISsequenceID)}. Or, for maps with at least two dimensions, try `mapFolding.foldings()`.")

def _parseBFileOEIS(bFileOEIS: str, oeisID: OEISsequenceID) -> Dict[int, int]:
    bFileLines = bFileOEIS.strip().splitlines()
    # Remove first line with the sequence ID
    if not bFileLines.pop(0).startswith(f"# {oeisID}"):
        raise ValueError(f"Content does not match sequence {oeisID}")

    OEISsequence = {}
    for line in bFileLines:
        if line.startswith('#'):
            continue
        n, aOFn = map(int, line.split())
        OEISsequence[n] = aOFn
    return OEISsequence

def _getOEISsequence(oeisID: OEISsequenceID) -> Dict[int, int]:
    """Fetch and parse an OEIS sequence from cache or URL."""
    pathFilenameCache = _pathCache / _formatFilenameCache.format(oeisID=oeisID)
    cacheDays = 7

    tryCache = False
    if pathFilenameCache.exists():
        fileAge = datetime.now() - datetime.fromtimestamp(pathFilenameCache.stat().st_mtime)
        tryCache = fileAge < timedelta(days=cacheDays)

    if tryCache:
        try:
            bFileOEIS = pathFilenameCache.read_text()
            return _parseBFileOEIS(bFileOEIS, oeisID)
        except (ValueError, IOError):
            tryCache = False

    url = f"https://oeis.org/{oeisID}/b{oeisID[1:]}.txt"
    httpResponse = urllib.request.urlopen(url)
    bFileOEIS = httpResponse.read().decode('utf-8')

    # Ensure cache directory exists
    if not tryCache:
        pathFilenameCache.parent.mkdir(parents=True, exist_ok=True)
        pathFilenameCache.write_text(bFileOEIS)

    return _parseBFileOEIS(bFileOEIS, oeisID)

for oeisID in settingsOEISsequences:
    settingsOEISsequences[oeisID]['valuesKnown'] = _getOEISsequence(oeisID)
    settingsOEISsequences[oeisID]['valueUnknown'] = max(settingsOEISsequences[oeisID]['valuesKnown'].values()) + 1

def getOEISids() -> None:
    """Print all available OEIS sequence IDs that are directly implemented."""
    print("\nAvailable OEIS sequences:")
    for oeisID in get_args(OEISsequenceID):
        print(f"  {oeisID}: {settingsOEISsequences[oeisID]['description']}")
    print("\nUsage example:")
    print("  from mapFolding import oeisSequence_aOFn")
    print("  foldingsTotal = oeisSequence_aOFn('A001415', 5)")

if __name__ == "__main__":
    getOEISids()
