from typing import Dict, Tuple

from mapFolding import settingsOEISsequences

# TODO convert this to a function; draft signature:
# def getFoldingsTotalKnown(listDimensions: List[int]) -> int:
dimensionsFoldingsTotalLookup: Dict[Tuple, int] = {}
for oeisID, settings in settingsOEISsequences.items():
    sequence = settings['valuesKnown']
    
    # Get all known dimensions and their folding counts
    for n, foldingsTotal in sequence.items():
        dimensions = settings['dimensions'](n)
        dimensions.sort()
        dimensionsFoldingsTotalLookup[tuple(dimensions)] = foldingsTotal
