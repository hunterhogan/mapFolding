from mapFolding.oeis import settingsOEIS as settingsOEIS

def makeDictionaryFoldsTotalKnown() -> dict[tuple[int, ...], int]:
    """Returns a dictionary mapping dimension tuples to their known folding totals."""
def getFoldsTotalKnown(mapShape: tuple[int, ...]) -> int: ...
