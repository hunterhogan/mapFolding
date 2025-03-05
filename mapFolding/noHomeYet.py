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
			mapShape = sorted(mapShape)
			dictionaryMapDimensionsToFoldsTotalKnown[tuple(mapShape)] = foldingsTotal
	return dictionaryMapDimensionsToFoldsTotalKnown

def getFoldsTotalKnown(mapShape: tuple[int, ...]) -> int:
	lookupFoldsTotal = makeDictionaryFoldsTotalKnown()
	return lookupFoldsTotal.get(tuple(mapShape), -1)
