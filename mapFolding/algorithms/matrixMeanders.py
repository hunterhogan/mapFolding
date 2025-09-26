from mapFolding.algorithms.matrixMeandersBeDry import walkDyckPath
from mapFolding.dataBaskets import MatrixMeandersState

def outfitDictionaryBitGroups(state: MatrixMeandersState) -> dict[tuple[int, int], int]:
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
    return {(arcCode & state.locatorBits, (arcCode >> 1) & state.locatorBits): crossings
        for arcCode, crossings in state.dictionaryMeanders.items()}

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
    dictionaryBitGroups: dict[tuple[int, int], int] = {}

    while state.kOfMatrix > 0:
        state.kOfMatrix -= 1

        dictionaryBitGroups = outfitDictionaryBitGroups(state)
        state.dictionaryMeanders = {}

        for (bitsAlpha, bitsZulu), crossings in dictionaryBitGroups.items():
            bitsAlphaHasArcs: bool = bitsAlpha > 1
            bitsZuluHasArcs: bool = bitsZulu > 1
            bitsAlphaIsEven = bitsZuluIsEven = 0

            arcCodeAnalysis = ((bitsAlpha | (bitsZulu << 1)) << 2) | 3
            # simple
            if arcCodeAnalysis < state.MAXIMUMarcCode:
                state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

            if bitsAlphaHasArcs:
                arcCodeAnalysis = (bitsAlpha >> 2) | (bitsZulu << 3) | ((bitsAlphaIsEven := 1 - (bitsAlpha & 1)) << 1)
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

            if bitsZuluHasArcs:
                arcCodeAnalysis = (bitsZulu >> 1) | (bitsAlpha << 2) | (bitsZuluIsEven := 1 - (bitsZulu & 1))
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

            if bitsAlphaHasArcs and bitsZuluHasArcs and (bitsAlphaIsEven or bitsZuluIsEven):
                # aligned
                if bitsAlphaIsEven and not bitsZuluIsEven:
                    bitsAlpha ^= walkDyckPath(bitsAlpha)  # noqa: PLW2901
                elif bitsZuluIsEven and not bitsAlphaIsEven:
                    bitsZulu ^= walkDyckPath(bitsZulu)  # noqa: PLW2901

                arcCodeAnalysis: int = ((bitsZulu >> 2) << 1) | (bitsAlpha >> 2)
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

        dictionaryBitGroups = {}

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
