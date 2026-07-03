from __future__ import annotations

from functools import cache
from mapFolding.dataBaskets import MatrixMeandersState  # noqa: TC001

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
    """Locate the first Dyck-balance failure bit in `intWithExtra_0b1`.

    You can use `walkDyckPath` to find the bit that must be toggled when an arc-joining transition in
    the meander transfer matrix closes a mismatched pair [1]. The `intWithExtra_0b1` value stores one
    side of the packed boundary state with parity bits at even positions.

    Parameters
    ----------
    intWithExtra_0b1 : int
        Packed bit pattern for one half of the current meander boundary state.

    Returns
    -------
    flipExtra_0b1_Here : int
        Bit mask `2^(2k)` at the first even-bit position where the prefix balance becomes negative.

    Bit Search
    ----------
    The scan advances by shifting `flipExtra_0b1_Here` left by `2` each step. The scan adds `1` when
    the bit is `0` and subtracts `1` when the bit is `1`. The function returns immediately at the
    first index where the running balance is negative in the Dyck-prefix sense [2].

    Mathematics
    -----------
    first negative prefix : equation
        ```text
        Let  x ≜ `intWithExtra_0b1`,  bᵢ ≜ bit(x, 2i),  sₖ ≜ ∑ᵢ₌₀ᵏ (1 if bᵢ = 0 else −1)

        k* ≜ min { k ∈ ℕ : sₖ < 0 }
        `flipExtra_0b1_Here` = 2^(2k*)
        ```

    References
    ----------
    [1] Jensen, I. (2000). A transfer matrix approach to the enumeration of plane meanders.
        Journal of Physics A: Mathematical and General, 33(34), 5953-5963.
        https://dx.doi.org/10.1088/0305-4470/33/34/301
    [2] Dyck language and balanced-parenthesis paths.
        https://en.wikipedia.org/wiki/Dyck_language
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

def count(state: MatrixMeandersState) -> MatrixMeandersState:
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
    dictionaryArcCodeToCrossings: dict[int, int] = {}

    while state.boundary > 0:
        state.reduceBoundary()

        dictionaryArcCodeToCrossings = state.dictionaryMeanders.copy()
        state.dictionaryMeanders = {}

        def analyzeArcCode(arcCode: int, crossings: int) -> None:
            bitsAlpha: int = arcCode & state.bitsLocator
            bitsAlphaHasArcs: bool = bitsAlpha > 1
            bitsAlphaIsEven: int = bitsAlpha & 1 ^ 1

            bitsZulu: int = arcCode >> 1 & state.bitsLocator
            bitsZuluHasArcs: bool = bitsZulu > 1
            bitsZuluIsEven: int = bitsZulu & 1 ^ 1

            arcCodeAnalysis: int = (bitsZulu << 1 | bitsAlpha) << 2 | 3  # Evaluate formula step-wise left to right: (parentheses) override precedence.
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
                # NOTE This analysis might modify `bitsAlpha` or `bitsZulu`, so it should be last.
                if bitsAlphaIsEven and not bitsZuluIsEven:
                    bitsAlpha ^= walkDyckPath(bitsAlpha)
                elif bitsZuluIsEven and not bitsAlphaIsEven:
                    bitsZulu ^= walkDyckPath(bitsZulu)

                arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlpha) >> 2  # Evaluate formula step-wise left to right: (parentheses) override precedence.
                if arcCodeAnalysis < state.MAXIMUMarcCode:
                    state.dictionaryMeanders[arcCodeAnalysis] = state.dictionaryMeanders.get(arcCodeAnalysis, 0) + crossings

        set(map(analyzeArcCode, dictionaryArcCodeToCrossings.keys(), dictionaryArcCodeToCrossings.values()))

        dictionaryArcCodeToCrossings = {}

    return state

def doTheNeedful(state: MatrixMeandersState) -> int:
    """Compute the total meander count encoded in `state.dictionaryMeanders`.

    You can use `doTheNeedful` as the meander transfer-matrix entry point for the `matrixMeanders`
    flow selected by `mapFolding.oeis.NOTcountingFolds` [1]. The function runs `count(state)` and
    returns the sum of final state multiplicities for the semi-meander and meandric counting context
    documented in OEIS entries A000682 and A005316 [2][3], the Jensen transfer-matrix method [4],
    and implementation lineages by Howroyd and Irvine [5][6].

    Parameters
    ----------
    state : MatrixMeandersState
        The algorithm state.

    Returns
    -------
    crossings : int
        The computed value of `crossings`.

    See Also
    --------
    `mapFolding.algorithms.matrixMeandersNumPyndas.doTheNeedful`
        Compute `crossings` with a transfer matrix algorithm implemented in NumPy.
    `mapFolding.algorithms.matrixMeandersNumPyndas.doTheNeedfulPandas`
        Compute `crossings` with a transfer matrix algorithm implemented in Pandas.

    References
    ----------
    [1] `mapFolding.oeis.NOTcountingFolds`

    [2] OEIS A000682, Semi-meanders.
        https://oeis.org/A000682
    [3] OEIS A005316, Meandric numbers.
        https://oeis.org/A005316
    [4] Jensen, I. (2000). A transfer matrix approach to the enumeration of plane meanders.
        Journal of Physics A: Mathematical and General, 33(34), 5953-5963.
        https://dx.doi.org/10.1088/0305-4470/33/34/301
    [5] Howroyd, A. (2015). C# Software for the Enumeration of Meanders.
        https://oeis.org/A005316/a005316.cs.txt
    [6] Irvine, S. A. (Java port). `A005316.java` in `archmageirvine/joeis`.
        https://github.com/archmageirvine/joeis/blob/5dc2148344bff42182e2128a6c99df78044558c5/src/irvine/oeis/a005/A005316.java
    """
    state = count(state)

    return sum(state.dictionaryMeanders.values())
