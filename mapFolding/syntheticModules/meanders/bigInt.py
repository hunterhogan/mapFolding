from functools import cache
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide
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
        A dictionary of `(bitsAlpha, bitsZulu)` to `distinctCrossings`.
    """
    state.bitWidth = max(state.dictionaryMeanders.keys()).bit_length()
    return {(arcCode & state.locatorBitsAlpha, (arcCode & state.locatorBitsZulu) >> 1): distinctCrossings for arcCode, distinctCrossings in state.dictionaryMeanders.items()}

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
    """Find the bit position for flipping paired curve endpoints in meander transfer matrices.

    Parameters
    ----------
    intWithExtra_0b1 : int
        Binary representation of curve locations with an extra bit encoding parity information.

    Returns
    -------
    flipExtra_0b1_Here : int
        Bit mask indicating the position where the balance condition fails, formatted as 2^(2k).

    3L33T H@X0R
    ------------
    Binary search for first negative balance in shifted bit pairs. Returns 2^(2k) mask for
    bit position k where cumulative balance counter transitions from non-negative to negative.

    Mathematics
    -----------
    Implements the Dyck path balance verification algorithm from Jensen's transfer matrix
    enumeration. Computes the position where ∑(i=0 to k) (-1)^b_i < 0 for the first time,
    where b_i are the bits of the input at positions 2i.

    """
    findTheExtra_0b1: int = 0
    flipExtra_0b1_Here: int = 1
    while True:
        flipExtra_0b1_Here <<= 2
        if intWithExtra_0b1 & flipExtra_0b1_Here == 0:
            findTheExtra_0b1 += 1
        else:
            findTheExtra_0b1 -= 1
        if findTheExtra_0b1 < 0:
            break
    return flipExtra_0b1_Here

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
        for (bitsAlpha, bitsZulu), distinctCrossings in dictionaryBitGroups.items():
            bitsAlphaHasArcs: bool = bitsAlpha > 1
            bitsZuluHasArcs: bool = bitsZulu > 1
            bitsAlphaIsEven = bitsZuluIsEven = 0
            arcCurveAnalysis = (bitsAlpha | bitsZulu << 1) << 2 | 3
            if arcCurveAnalysis < state.MAXIMUMarcCode:
                state.dictionaryMeanders[arcCurveAnalysis] = state.dictionaryMeanders.get(arcCurveAnalysis, 0) + distinctCrossings
            if bitsAlphaHasArcs:
                arcCurveAnalysis = bitsAlpha >> 2 | bitsZulu << 3 | (bitsAlphaIsEven := (1 - (bitsAlpha & 1))) << 1
                if arcCurveAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCurveAnalysis] = state.dictionaryMeanders.get(arcCurveAnalysis, 0) + distinctCrossings
            if bitsZuluHasArcs:
                arcCurveAnalysis = bitsZulu >> 1 | bitsAlpha << 2 | (bitsZuluIsEven := (1 - (bitsZulu & 1)))
                if arcCurveAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCurveAnalysis] = state.dictionaryMeanders.get(arcCurveAnalysis, 0) + distinctCrossings
            if bitsAlphaHasArcs and bitsZuluHasArcs and (bitsAlphaIsEven or bitsZuluIsEven):
                if bitsAlphaIsEven and (not bitsZuluIsEven):
                    bitsAlpha ^= walkDyckPath(bitsAlpha)
                elif bitsZuluIsEven and (not bitsAlphaIsEven):
                    bitsZulu ^= walkDyckPath(bitsZulu)
                arcCurveAnalysis: int = bitsZulu >> 2 << 1 | bitsAlpha >> 2
                if arcCurveAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCurveAnalysis] = state.dictionaryMeanders.get(arcCurveAnalysis, 0) + distinctCrossings
        dictionaryBitGroups = {}
    return state
