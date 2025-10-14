from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, walkDyckPath
from mapFolding.dataBaskets import MatrixMeandersNumPyState

def countBigInt(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
    """Count meanders with matrix transfer algorithm using Python `int` (*int*eger) contained in a Python `dict` (*dict*ionary).

    Parameters
    ----------
    state : MatrixMeandersState
        The algorithm state.

    Notes
    -----
    The matrix transfer algorithm is sophisticated, but this implementation is straightforward: compute each `boundary` one at a
    time, compute each `arcCode` one at a time, and compute each type of analysis one at a time.
    """
    dictionaryArcCodeToCrossings: dict[int, int] = {}
    while state.boundary > 0 and areIntegersWide(state):
        state.reduceBoundary()
        dictionaryArcCodeToCrossings = state.dictionaryMeanders.copy()
        state.dictionaryMeanders = {}
        for arcCode, crossings in dictionaryArcCodeToCrossings.items():
            bitsAlpha: int = arcCode & state.bitsLocator
            bitsAlphaHasArcs: bool = bitsAlpha > 1
            bitsAlphaIsEven: int = bitsAlpha & 1 ^ 1
            bitsZulu: int = arcCode >> 1 & state.bitsLocator
            bitsZuluHasArcs: bool = bitsZulu > 1
            bitsZuluIsEven: int = bitsZulu & 1 ^ 1
            arcCodeAnalysis: int = (bitsZulu << 1 | bitsAlpha) << 2 | 3
            if arcCodeAnalysis < state.MAXIMUMarcCode:
                state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsAlphaHasArcs:
                arcCodeAnalysis = bitsAlphaIsEven << 1 | bitsAlpha >> 2 | bitsZulu << 3
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsZuluHasArcs:
                arcCodeAnalysis = bitsZuluIsEven | bitsAlpha << 2 | bitsZulu >> 1
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsAlphaHasArcs and bitsZuluHasArcs and (bitsAlphaIsEven or bitsZuluIsEven):
                if bitsAlphaIsEven and (not bitsZuluIsEven):
                    bitsAlpha ^= walkDyckPath(bitsAlpha)
                elif bitsZuluIsEven and (not bitsAlphaIsEven):
                    bitsZulu ^= walkDyckPath(bitsZulu)
                arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlpha) >> 2
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
        dictionaryArcCodeToCrossings = {}
    return state
