from __future__ import annotations

from mapFolding.algorithms.matrixMeandersShare import integersWide吗, walkDyckPath
from mapFolding.dataBaskets import MatrixMeandersState

def countBigInt(state: MatrixMeandersState) -> MatrixMeandersState:
    """Advance one meander transfer-matrix computation until `state.boundary` reaches zero.

    You can use `count` to apply all transition rules for each boundary layer in `state` and update
    `state.dictionaryMeanders` in place [1]. The transition set includes new-arc insertion, left-right
    crossing moves, and Dyck-path-based arc joining through `walkDyckPath` [2].

    Parameters
    ----------
    state : MatrixMeandersState
        The algorithm state.

    Returns
    -------
    state : MatrixMeandersState
        The same `state` instance after all boundary layers have been processed.

    Transfer Steps
    --------------
    The function decrements `state.boundary`, consumes each previous `arcCode`, and accumulates the
    next boundary dictionary. The transition for joining arches conditionally flips one Dyck-matched
    endpoint bit before packing the next `arcCode`.

    References
    ----------
    [1] `mapFolding.dataBaskets.MatrixMeandersState`

    [2] `walkDyckPath`
    """
    while 0 < state.boundary and integersWide吗(state):
        state.reduceBoundary()
        dictionaryArcCodeToCrossings: dict[int, int] = state.dictionaryMeanders.copy()
        state.dictionaryMeanders = {}

        def analyzeArcCode(arcCode: int, crossings: int) -> None:
            bitsAlfa: int = arcCode & state.bitsLocator
            bitsAlfaHasArcs: bool = 1 < bitsAlfa
            bitsAlfaIsEven: int = bitsAlfa & 1 ^ 1
            bitsZulu: int = arcCode >> 1 & state.bitsLocator
            bitsZuluHasArcs: bool = 1 < bitsZulu
            bitsZuluIsEven: int = bitsZulu & 1 ^ 1
            arcCodeAnalysis: int = (bitsZulu << 1 | bitsAlfa) << 2 | 3
            if arcCodeAnalysis < state.MAXIMUMarcCode:
                state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsAlfaHasArcs:
                arcCodeAnalysis = bitsAlfaIsEven << 1 | bitsAlfa >> 2 | bitsZulu << 3
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsZuluHasArcs:
                arcCodeAnalysis = bitsZuluIsEven | bitsAlfa << 2 | bitsZulu >> 1
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
            if bitsAlfaHasArcs and bitsZuluHasArcs and (bitsAlfaIsEven or bitsZuluIsEven):
                if bitsAlfaIsEven and (not bitsZuluIsEven):
                    bitsAlfa ^= walkDyckPath(bitsAlfa)
                elif bitsZuluIsEven and (not bitsAlfaIsEven):
                    bitsZulu ^= walkDyckPath(bitsZulu)
                arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlfa) >> 2
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings
        tuple(map(analyzeArcCode, dictionaryArcCodeToCrossings.keys(), dictionaryArcCodeToCrossings.values()))
    return state
