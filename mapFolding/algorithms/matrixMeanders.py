from mapFolding.algorithms.matrixMeandersBeDry import walkDyckPath
from mapFolding.dataBaskets import MatrixMeandersState

def count(state: MatrixMeandersState) -> MatrixMeandersState:
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

    while state.boundary > 0:
        state.boundary -= 1
        state.bitWidth = max(state.dictionaryMeanders.keys()).bit_length()

        dictionaryArcCodeToCrossings = state.dictionaryMeanders.copy()
        state.dictionaryMeanders = {}

        for arcCode, crossings in dictionaryArcCodeToCrossings.items():
            bitsAlpha: int = arcCode & state.locatorBits
            bitsZulu: int = (arcCode >> 1) & state.locatorBits
            bitsAlphaHasArcs: bool = bitsAlpha > 1
            bitsZuluHasArcs: bool = bitsZulu > 1
            bitsAlphaIsEven: int = bitsAlpha & 1 ^ 1
            bitsZuluIsEven: int = bitsZulu & 1 ^ 1

            arcCodeAnalysis = ((bitsAlpha | (bitsZulu << 1)) << 2) | 3
            # simple
            if arcCodeAnalysis < state.MAXIMUMarcCode:
                state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

            if bitsAlphaHasArcs:
                arcCodeAnalysis = (bitsAlpha >> 2) | (bitsZulu << 3) | (bitsAlphaIsEven << 1)
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

            if bitsZuluHasArcs:
                arcCodeAnalysis = (bitsZulu >> 1) | (bitsAlpha << 2) | bitsZuluIsEven
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

            if bitsAlphaHasArcs and bitsZuluHasArcs and (bitsAlphaIsEven or bitsZuluIsEven):
                # aligned
                if bitsAlphaIsEven and not bitsZuluIsEven:
                    bitsAlpha ^= walkDyckPath(bitsAlpha)
                elif bitsZuluIsEven and not bitsAlphaIsEven:
                    bitsZulu ^= walkDyckPath(bitsZulu)

                arcCodeAnalysis: int = ((bitsZulu >> 2) << 1) | (bitsAlpha >> 2)
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

        dictionaryArcCodeToCrossings = {}

    return state

def doTheNeedful(state: MatrixMeandersState) -> int:
    """Compute `crossings` with a transfer matrix algorithm.

    Parameters
    ----------
    state : MatrixMeandersState
        The algorithm state.

    Returns
    -------
    crossings : int
        The computed value of `crossings`.

    Notes
    -----
    Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex

    See Also
    --------
    https://oeis.org/A000682
    https://oeis.org/A005316
    """
    state = count(state)

    return sum(state.dictionaryMeanders.values())
