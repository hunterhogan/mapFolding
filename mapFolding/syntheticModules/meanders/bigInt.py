from mapFolding.algorithms.matrixMeandersBeDry import (areIntegersWide,
                                                       walkDyckPath)
from mapFolding.dataBaskets import MatrixMeandersNumPyState


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
    dictionaryArcCodeToCrossings: dict[int, int] = {}
    while state.boundary > 0 and areIntegersWide(state):
        state.boundary -= 1
        state.bitWidth = max(state.dictionaryMeanders.keys()).bit_length()
        dictionaryArcCodeToCrossings = state.dictionaryMeanders.copy()
        state.dictionaryMeanders = {}
        for arcCode, crossings in dictionaryArcCodeToCrossings.items():
            bitsAlpha: int = arcCode & state.locatorBits
            bitsZulu: int = arcCode >> 1 & state.locatorBits
            bitsAlphaHasArcs: bool = bitsAlpha > 1
            bitsZuluHasArcs: bool = bitsZulu > 1
            bitsAlphaIsEven: int = bitsAlpha & 1 ^ 1
            bitsZuluIsEven: int = bitsZulu & 1 ^ 1
            arcCodeAnalysis = (bitsAlpha | bitsZulu << 1) << 2 | 3
            if arcCodeAnalysis < state.MAXIMUMarcCode:
                state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsAlphaHasArcs:
                arcCodeAnalysis = bitsAlpha >> 2 | bitsZulu << 3 | bitsAlphaIsEven << 1
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsZuluHasArcs:
                arcCodeAnalysis = bitsZulu >> 1 | bitsAlpha << 2 | bitsZuluIsEven
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
        dictionaryArcCodeToCrossings = {}
    return state
