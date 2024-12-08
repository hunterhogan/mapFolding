from datetime import datetime, timedelta
from mapFolding import foldings
from typing import Callable, Dict, List, TYPE_CHECKING, Literal, get_args
import math
import os
import pathlib
import random
import time
import urllib.request
import warnings

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
    'A001418': {
        'description': 'Number of ways of folding an n X n sheet of stamps.',
        'dimensions': lambda n: [n, n],
        'benchmarkValues': [5],
        'testValuesValidation': [*range(1, 4)],
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
    if n < 0:
        raise ValueError("n must be non-negative.")
    if oeisID not in settingsOEISsequences:
        raise KeyError(f"Sequence {oeisID} is not directly implemented in mapFoldings. Use `mapFolding.foldings()` instead.")

    listDimensions = settingsOEISsequences[oeisID]['dimensions'](n)
    return foldings(listDimensions) if n > 0 else 1

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

dimensionsFoldingsTotalLookup: Dict[tuple, int] = {}
for oeisID, settings in settingsOEISsequences.items():
    sequence = settings['valuesKnown']
    
    # Get all known dimensions and map to their folding counts
    for n, foldingsTotal in sequence.items():
        dimensions = settings['dimensions'](n)
        dimensions.sort()
        dimensionsFoldingsTotalLookup[tuple(dimensions)] = foldingsTotal
