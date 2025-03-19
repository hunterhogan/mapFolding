"""
Interface for retrieving known map folding totals from OEIS (Online Encyclopedia of Integer Sequences).

This module provides utilities for accessing pre-computed map folding totals that are known
from mathematical literature and stored in the OEIS. The functions cache results for
performance and provide lookups based on map dimensions.

The main functions are:
- makeDictionaryFoldsTotalKnown: Creates a dictionary of known folding totals indexed by map dimensions
- getFoldsTotalKnown: Retrieves the folding total for a specific map shape, returning -1 if unknown
"""

from functools import cache
from mapFolding.oeis import settingsOEIS

@cache
def makeDictionaryFoldsTotalKnown() -> dict[tuple[int, ...], int]:
	"""Returns a dictionary mapping dimension tuples to their known folding totals."""
	dictionaryMapDimensionsToFoldsTotalKnown: dict[tuple[int, ...], int] = {}

	for settings in settingsOEIS.values():
		sequence = settings['valuesKnown']

		for n, foldingsTotal in sequence.items():
			mapShape = settings['getMapShape'](n)
			mapShape = tuple(sorted(mapShape))
			dictionaryMapDimensionsToFoldsTotalKnown[mapShape] = foldingsTotal
	return dictionaryMapDimensionsToFoldsTotalKnown

def getFoldsTotalKnown(mapShape: tuple[int, ...]) -> int:
	lookupFoldsTotal = makeDictionaryFoldsTotalKnown()
	return lookupFoldsTotal.get(tuple(mapShape), -1)
