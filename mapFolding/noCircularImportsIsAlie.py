from typing import Dict, List, Tuple, get_args

from mapFolding import parseListDimensions
from mapFolding.oeis import OEISsequenceID

def getFoldingsTotalKnown(listDimensions: List[int]) -> int:
    """
    Look up the known total number of foldings for given map dimensions.
    
    Parameters:
        listDimensions: List of dimensions to look up

    Returns:
        Total number of foldings if known, raises KeyError if not found
    """
    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listPositive = [dimension for dimension in listNonNegative if dimension > 0]

    if len(listPositive) < 2:
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. Other functions in this package implement the sequences {get_args(OEISsequenceID)}. You may want to look at https://oeis.org/.")

    dimensionsFoldingsTotalLookup: Dict[Tuple, int] = {}
    from .oeis import settingsOEISsequences
    for oeisID, settings in settingsOEISsequences.items():
        sequence = settings['valuesKnown']
        
        # Get all known dimensions and their folding counts
        for n, foldingsTotal in sequence.items():
            dimensions = settings['dimensions'](n)
            dimensions.sort()
            dimensionsFoldingsTotalLookup[tuple(dimensions)] = foldingsTotal
    
    dimensionsTuple = tuple(sorted(listPositive))
    if dimensionsTuple in dimensionsFoldingsTotalLookup:
        return dimensionsFoldingsTotalLookup[dimensionsTuple]
    else:
        raise KeyError(f"No known folding count for dimensions {listDimensions}")
