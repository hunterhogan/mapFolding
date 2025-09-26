from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, walkDyckPath
from mapFolding.dataBaskets import MatrixMeandersNumPyState

def outfitDictionaryBitGroups(state: MatrixMeandersNumPyState) -> dict[tuple[int, int], int]:
    """Outfit `dictionaryBitGroups` so it may manage the computations for one iteration of the transfer matrix.

    Parameters
    ----------
    state : MatrixMeandersState
        The current state of the computation, including `dictionaryMeanders`.

    Returns
    -------
    dictionaryBitGroups : dict[tuple[int, int], int]
        A dictionary of `(bitsAlpha, bitsZulu)` to `crossings`.
    """
    state.bitWidth = max(state.dictionaryMeanders.keys()).bit_length()
    return {(arcCode & state.locatorBits, arcCode >> 1 & state.locatorBits): crossings for arcCode, crossings in state.dictionaryMeanders.items()}

def countBigInt(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
    """Count meanders with matrix transfer algorithm using Python `int` (*int*eger) contained in a Python `dict` (*dict*ionary).

    Parameters
    ----------
    state : MatrixMeandersState
        The algorithm state.

    Notes
    -----
    The matrix transfer algorithm is sophisticated, but this implementation is straightforward: compute each index one at a time,
    compute each `arcCode` one at a time, and compute each type of analysis one at a time.
    """
    dictionaryBitGroups: dict[tuple[int, int], int] = {}
    while state.kOfMatrix > 0 and areIntegersWide(state):
        state.kOfMatrix -= 1
        dictionaryBitGroups = outfitDictionaryBitGroups(state)
        state.dictionaryMeanders = {}
        for (bitsAlpha, bitsZulu), crossings in dictionaryBitGroups.items():
            bitsAlphaHasArcs: bool = bitsAlpha > 1
            bitsZuluHasArcs: bool = bitsZulu > 1
            bitsAlphaIsEven = bitsZuluIsEven = 0
            arcCodeAnalysis = (bitsAlpha | bitsZulu << 1) << 2 | 3
            if arcCodeAnalysis < state.MAXIMUMarcCode:
                state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsAlphaHasArcs:
                arcCodeAnalysis = bitsAlpha >> 2 | bitsZulu << 3 | (bitsAlphaIsEven := (1 - (bitsAlpha & 1))) << 1
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsZuluHasArcs:
                arcCodeAnalysis = bitsZulu >> 1 | bitsAlpha << 2 | (bitsZuluIsEven := (1 - (bitsZulu & 1)))
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsAlphaHasArcs and bitsZuluHasArcs and (bitsAlphaIsEven or bitsZuluIsEven):
                if bitsAlphaIsEven and (not bitsZuluIsEven):
                    bitsAlpha ^= walkDyckPath(bitsAlpha)
                elif bitsZuluIsEven and (not bitsAlphaIsEven):
                    bitsZulu ^= walkDyckPath(bitsZulu)
                arcCodeAnalysis: int = bitsZulu >> 2 << 1 | bitsAlpha >> 2
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
        dictionaryBitGroups = {}
    return state
