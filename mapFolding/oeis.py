import pathlib
import random
import urllib.request
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, get_args, Tuple

from mapFolding import foldings

if TYPE_CHECKING:
    from typing import TypedDict
else:
    TypedDict = dict

class SettingsOEISsequence(TypedDict):
    description: str
    dimensions: Callable[[int], List[int]]
    benchmarkValues: List[int]
    testValuesValidation: List[int]
    valuesKnown: Dict[int, int]
    valueUnknown: List[int]

try:
    pathCache = pathlib.Path(__file__).parent / ".cache"
except NameError:
    pathCache = pathlib.Path.home() / ".mapFoldingCache"

OEISsequenceID = Literal['A001415', 'A001416', 'A001417', 'A195646', 'A001418']

settingsOEISsequences: Dict[OEISsequenceID, SettingsOEISsequence] = {
    'A001415': {
        'description': 'Number of ways of folding a 2 X n strip of stamps.',
        'dimensions': lambda n: [2, n],
        'benchmarkValues': [11],
        'testValuesValidation': [0, 1, random.randint(2, 9)],
        'valueUnknown': [2, 19],
        'valuesKnown': {},  # Placeholder
    },
    'A001416': {
        'description': 'Number of ways of folding a 3 X n strip of stamps.',
        'dimensions': lambda n: [3, n],
        'benchmarkValues': [8],
        'testValuesValidation': [0, 1, random.randint(2, 6)],
        'valueUnknown': [3, 15],
        'valuesKnown': {},  # Placeholder
    },
    'A001417': {
        'description': 'Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.',
        'dimensions': lambda n: [2] * n,
        'benchmarkValues': [5],
        'testValuesValidation': [0, 1, random.randint(2, 4)],
        'valueUnknown': [2, 2, 2, 2, 2, 2, 2, 2],
        'valuesKnown': {},  # Placeholder
    },
    'A195646': {
        'description': 'Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.',
        'dimensions': lambda n: [3] * n,
        'benchmarkValues': [3],
        'testValuesValidation': [0, 1, 2],
        'valueUnknown': [3, 3, 3, 3],
        'valuesKnown': {},  # Placeholder
    },
    'A001418': { # offset 1
        'description': 'Number of ways of folding an n X n sheet of stamps.',
        'dimensions': lambda n: [n, n],
        'benchmarkValues': [5],
        'testValuesValidation': [*range(1, 5)],
        'valueUnknown': [8, 8],
        'valuesKnown': {},  # Placeholder
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
    if oeisID not in get_args(OEISsequenceID):
        oeisID = oeisID.upper().strip() # type: ignore
    if oeisID not in settingsOEISsequences:
        raise KeyError(f"Sequence {oeisID} is not directly implemented in mapFoldings. The directly implemented sequences are {get_args(OEISsequenceID)}. Or, for maps with at least two dimensions, try `mapFolding.foldings()`.")
    if n < 0 or not isinstance(n, int):
        raise ValueError("`n` must be non-negative integer.")
    elif n == 0:
        foldingsTotal = settingsOEISsequences[oeisID]['valuesKnown'].get(n, None)
        if foldingsTotal is not None:
            return foldingsTotal
        else:
            raise ArithmeticError(f"Sequence {oeisID} is not defined at {n=}.")

    listDimensions = settingsOEISsequences[oeisID]['dimensions'](n)
    if len(listDimensions) < 2:
        foldingsTotal = settingsOEISsequences[oeisID]['valuesKnown'].get(n, None)
        if foldingsTotal is not None:
            return foldingsTotal
        else:
            raise ArithmeticError(f"Sequence {oeisID} is not defined at {n=}.")
    return foldings(listDimensions)

def _parseContent(bFileOEIS: str, oeisID: OEISsequenceID) -> Dict[int, int]:
    bFileLines = bFileOEIS.strip().splitlines()
    # Remove first line with sequence ID
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
    pathFilenameCache = pathCache / f"{oeisID}.txt"
    cacheDays = 7

    tryCache = False
    if pathFilenameCache.exists():
        fileAge = datetime.now() - datetime.fromtimestamp(pathFilenameCache.stat().st_mtime)
        tryCache = fileAge < timedelta(days=cacheDays)
    
    if tryCache:
        try:
            bFileOEIS = pathFilenameCache.read_text()
            return _parseContent(bFileOEIS, oeisID)
        except (ValueError, IOError):
            tryCache = False
    
    url = f"https://oeis.org/{oeisID}/b{oeisID[1:]}.txt"
    httpResponse = urllib.request.urlopen(url)
    bFileOEIS = httpResponse.read().decode('utf-8')
    
    # Ensure cache directory exists
    if not tryCache:
        pathFilenameCache.parent.mkdir(parents=True, exist_ok=True)
        pathFilenameCache.write_text(bFileOEIS)
    
    return _parseContent(bFileOEIS, oeisID)

for oeisID in settingsOEISsequences:
    settingsOEISsequences[oeisID]['valuesKnown'] = _getOEISsequence(oeisID)

dimensionsFoldingsTotalLookup: Dict[Tuple, int] = {}
for oeisID, settings in settingsOEISsequences.items():
    sequence = settings['valuesKnown']
    
    # Get all known dimensions and map to their folding counts
    for n, foldingsTotal in sequence.items():
        dimensions = settings['dimensions'](n)
        dimensions.sort()
        dimensionsFoldingsTotalLookup[tuple(dimensions)] = foldingsTotal
