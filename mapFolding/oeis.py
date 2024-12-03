import pathlib
import urllib.request
from datetime import datetime, timedelta
from typing import Dict

from mapFolding import foldings

try:
    from .oeisSettings import pathCache, settingsOEISsequences
except ImportError:
    from oeisSettings import pathCache, settingsOEISsequences


def oeisSequence_aOFn(oeisID: str, n: int) -> int:
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

def getOEISsequence(oeisID: str) -> Dict[int, int]:
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

def _parseContent(bFileOEIS: str, oeisID: str) -> Dict[int, int]:
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

dimensionsFoldingsTotalLookup = {}
for oeisID, settings in settingsOEISsequences.items():
    sequence = getOEISsequence(oeisID)
    
    # Get all known dimensions and map to their folding counts
    for n, foldingsTotal in sequence.items():
        dimensions = settings['dimensions'](n)
        dimensions.sort()
        dimensionsFoldingsTotalLookup[tuple(dimensions)] = foldingsTotal
