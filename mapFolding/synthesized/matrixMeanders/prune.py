from __future__ import annotations

from functools import cache
from mapFolding.dataBaskets import StateMeanders
from mapFolding.oeis.A400429oeisByFormula import boxOfDiagonals
from mapFolding.theTypes import 形ArrayArcCode
from operator import neg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, LiteralString
'Matrix Meanders Goals:\n- Extreme abstraction.\n- Find operations with latent intermediate arrays and make the intermediate array explicit.\n- Reduce or eliminate intermediate arrays and selector arrays.\n- Write formulas in prefix notation.\n- For each formula, find an equivalent prefix notation formula that never uses the same variable as input more than once: that\n\twould allow the evaluation of the expression with only a single stack, which saves memory.\n- Standardize code as much as possible to create duplicate code.\n- Convert duplicate code to procedures.\n'

def integersWide吗(state: StateMeanders) -> bool:
    """Determine whether the next transfer step exceeds the fixed-width integer capacity.

    (AI generated docstring)

    The current NumPy and pandas formulas need up to four bits beyond the widest `arcCode`.
    Each input produces at most four contributions, so four times the sum of active counts
    bounds every aggregated count in the next step. This bound can select Python integers
    earlier than an exact analysis of each output bucket would.

    `state.bitWidthLimit` counts usable nonnegative value bits. For example, unsigned 64-bit
    integers provide 64 value bits and signed 128-bit integers provide 127. The boundary
    threshold `state.arcCodeMaximum` remains a Python integer and consumes no buffer capacity.
    """
    bitWidthArcCode: int = max(state.lookupMeanders, default=0).bit_length() + 4
    bitWidthMeanders: int = (4 * sum(state.lookupMeanders.values())).bit_length()
    return state.bitWidthLimit < max(bitWidthArcCode, bitWidthMeanders)

def makeLookupMeanders(kind: Literal['closed', 'meanders', 'semi'] | LiteralString, n: int, boundary: int=0) -> dict[int, int]:
    """Create the starting `lookupMeanders` for a matrix meander count.

    (AI generated docstring)

    You can use this function to build the initial mapping from arch codes to counts. For
    `kind == 'semi'`, the number and values of the starting codes depend on `n` and `boundary`.
    For `kind == 'meanders'`, one starting code is selected using the parity of `n`.
    For `kind == 'closed'`, one fixed starting code is used.

    Parameters
    ----------
    kind : Literal['closed', 'meanders', 'semi'] | LiteralString
        Which family of meander count to initialize.
    n : int
        The number of times the meander crosses the road.
    boundary : int = 0
        The boundary from which to start counting. Zero selects `n - 1`.

    Returns
    -------
    lookupMeanders : dict[int, int]
        A dictionary whose keys are `arcCode` and whose values are `1`.

    Raises
    ------
    ValueError
        Raised when `kind` is not `'closed'`, `'semi'`, or `'meanders'`.
    """
    if not boundary:
        boundary = n - 1
    if kind == 'closed':
        lookupMeanders = {15: 1}
    elif kind == 'meanders':
        if n & 1:
            lookupMeanders = {15: 1}
        else:
            lookupMeanders = {22: 1}
    elif kind == 'semi':
        if n == 1:
            lookupMeanders: dict[int, int] = {1: 1}
        else:
            if n & 1:
                arcCode: int = 5
            else:
                arcCode = 1
            boxOfArcCodes: list[int] = [arcCode << 1 | arcCode]
            arcCodeMaximum: int = 1 << 2 * boundary + 4
            while boxOfArcCodes[-1] < arcCodeMaximum >> 8:
                arcCode = arcCode << 4 | 5
                boxOfArcCodes.append(arcCode << 1 | arcCode)
            lookupMeanders: dict[int, int] = dict.fromkeys(boxOfArcCodes, 1)
    else:
        message: str = f"I received `kind = {kind!r}` for meander computation, but I don't know that kind."
        raise ValueError(message)
    return lookupMeanders
totalBucketsMaximumBy_boundary: dict[int, int] = {1: 3, 2: 12, 3: 40, 4: 125, 5: 392, 6: 1254, 7: 4087, 8: 13623, 9: 46181, 10: 159137, 11: 555469, 12: 1961369, 13: 6991893, 14: 25134208}

def getTotalBuckets(state: StateMeanders, totalArcCodes: int=0) -> int:
    """Estimate the output allocation length for one transfer-matrix step.

    (AI generated docstring)

    You can use this function to choose an initial output-buffer length from the number of input
    records and the transfer boundary. The estimate can be smaller than the required allocation;
    callers that write additional records must allow the buffer to grow.

    Parameters
    ----------
    state : StateMeanders
        The current transfer-matrix state.
    totalArcCodes : int = 0
        Number of unique input `arcCode` at the current boundary.

    Returns
    -------
    totalBuckets : int
        The exact or estimated number of non-unique `arcCode` rows to allocate.
    """
    totalBuckets: int = (355 * totalArcCodes + 99) // 100
    return max(totalBuckets, totalBucketsMaximumBy_boundary.get(state.boundary, totalBuckets))

def makeLookupDiagonal(n: int, 次diagonal: int) -> StateMeanders:
    """Prepare a semi-meander state for counting diagonal `次diagonal` at row `n`.

    (AI generated docstring)

    You can use this function to initialize the transfer calculation for one triangle cell.
    The returned `StateMeanders` starts with one arch code and count one. Counting has not
    yet run.

    Parameters
    ----------
    n : int
        Semi-meander row number in the domain of the selected diagonal.
    次diagonal : int
        Diagonal offset from the complete right edge, starting at one.

    Returns
    -------
    state : StateMeanders
        Initial state for the selected diagonal term, ready for
        `mapFolding.basecamp.countMeanders`.

    See Also
    --------
    `mapFolding.basecamp.countMeanders`
        Run the prepared diagonal state's counting calculation.

    Examples
    --------
    `easyRun/triangleSemiMake.py` counts a diagonal term before saving the total.

        ```python
        state = countMeanders(state=makeLookupDiagonal(n, 次diagonal))
        ```

    """
    arcCode: int = (1 << 2 * (n - 2 * 次diagonal) + 2) - 1
    return StateMeanders(n, 'semi', lookupMeanders={arcCode: 1})

def prune(state: StateMeanders) -> StateMeanders:
    """Accumulate final contributions for dictionary states with known completion counts.

    (AI generated docstring)

    You can use this function between counting iterations to remove entries whose remaining work
    can be evaluated by `mapFolding.oeis.A400429oeisByFormula.boxOfDiagonals`. The function preserves
    each removed entry's final contribution in `state.countAddend` and returns the updated state.

    Parameters
    ----------
    state : StateMeanders
        Current computation state with arc codes and their subtotals in `state.lookupMeanders`.

    Returns
    -------
    state : StateMeanders
        The same state, with recognized entries removed and their final contributions accumulated.

    See Also
    --------
    `mapFolding.oeis.A400429oeisByFormula.boxOfDiagonals`
        Inspect the known diagonal formulas used to complete selected dictionary entries.
    `mapFolding.synthesized.matrixMeanders.matrixMeandersShare.pruneDataFrame`
        Prune an active table stored as a dataframe instead of a dictionary.

    Diagonal Selection
    ------------------
    The local `boundary` is `state.boundary + 1`. Successive functions in `boxOfDiagonals` match
    diagonals numbered from one through `boundary // 2`, stopping when the formula list ends.
    For diagonal `次diagonal`, the selected key is
    `(1 << (2 * (boundary - 2 * 次diagonal) + 2)) - 1`.

    The function removes the selected key, multiplies the removed subtotal by the diagonal formula
    evaluated at `boundary`, and adds the product to `state.countAddend`. A missing key contributes
    zero. If no dictionary entries remain, the function sets `state.boundary` to zero so iteration
    can stop. Other boundary-derived fields are not refreshed.

    Examples
    --------
    `mapFolding.synthesized.matrixMeanders.prune.count` prunes before iteration and after each
    transfer step.

        ```python
        state = prune(state)
        ```
    """
    boundary: int = state.boundary + 1
    次diagonal: int = 1
    while 次diagonal <= min(boundary // 2, len(boxOfDiagonals)):
        arcCode: int = (1 << 2 * (boundary - 2 * 次diagonal) + 2) - 1
        subtotalMeanders: int = state.lookupMeanders.pop(arcCode, 0)
        state.countAddend += boxOfDiagonals[次diagonal - 1](boundary) * subtotalMeanders
        次diagonal += 1
    if not state.lookupMeanders:
        state.boundary = 0
    return state

def integersWideNumPy吗(state: StateMeanders, arrayMeanders: 形ArrayArcCode) -> bool:
    """Determine whether the next transfer step exceeds the fixed-width integer capacity.

    (AI generated docstring)

    The current NumPy and pandas formulas need up to four bits beyond the widest `arcCode`.
    Each input produces at most four contributions, so four times the sum of active counts
    bounds every aggregated count in the next step. This bound can select Python integers
    earlier than an exact analysis of each output bucket would.

    `state.bitWidthLimit` counts usable nonnegative value bits. For example, unsigned 64-bit
    integers provide 64 value bits and signed 128-bit integers provide 127. The boundary
    threshold `state.arcCodeMaximum` remains a Python integer and consumes no buffer capacity.
    """
    bitWidthArcCode: int = int(arrayMeanders[..., 0].max(initial=0)).bit_length() + 4
    bitWidthMeanders: int = (4 * int(arrayMeanders[..., 1].sum(dtype=object))).bit_length()
    return state.bitWidthLimit < max(bitWidthArcCode, bitWidthMeanders)
if TYPE_CHECKING:
    from numpy import ufunc
    flipDyckEndpoint: ufunc
else:
    from numba import vectorize

    @vectorize(['int64(int64)', 'uint64(uint64)'], cache=True, nopython=True)
    def flipDyckEndpoint(intWithExtra_0b1: int) -> int:
        """Flip the matched endpoint in each unbalanced Dyck path.

        (AI generated docstring)

        Every input must contain a matching endpoint. The generated NumPy ufunc accepts
        signed and unsigned 64-bit scalars and arrays, including pandas columns.
        """
        findTheExtra_0b1: int = 0
        flipExtra_0b1_Here: int = 1
        while 0 <= findTheExtra_0b1:
            flipExtra_0b1_Here <<= 2
            if intWithExtra_0b1 & flipExtra_0b1_Here == 0:
                findTheExtra_0b1 += 1
            else:
                findTheExtra_0b1 += neg(1)
        return intWithExtra_0b1 ^ flipExtra_0b1_Here
    flipDyckEndpoint = flipDyckEndpoint.ufunc

def countBigInt(state: StateMeanders) -> StateMeanders:
    """Advance one meander transfer-matrix computation until `state.boundary` reaches zero.

    (AI generated docstring)

    You can use `count` to apply all transition rules for each boundary layer in `state` and update
    `state.lookupMeanders` in place. The transition set includes new-arc insertion, left-right
    crossing moves, and Dyck-path-based arc joining through `walkDyckPath`.

    Parameters
    ----------
    state : StateMeanders
        The algorithm state.

    Returns
    -------
    state : StateMeanders
        The same `state` instance after all boundary layers have been processed.

    Transfer Steps
    --------------
    The function decrements `state.boundary`, consumes each previous `arcCode`, and accumulates the
    next boundary dictionary. The transition for joining arches conditionally flips one Dyck-matched
    endpoint bit before packing the next `arcCode`.

    """
    state = prune(state)
    while 0 < state.boundary and integersWide吗(state):

        def analyzeArcCode(arcCode: int, meanders: int, state: StateMeanders=state) -> None:
            bitsAlfa: int = arcCode & state.bitsLocator
            bitsAlfaHasArcs: bool = 1 < bitsAlfa
            bitsAlfaIsEven: int = bitsAlfa & 1 ^ 1
            bitsZulu: int = arcCode >> 1 & state.bitsLocator
            bitsZuluHasArcs: bool = 1 < bitsZulu
            bitsZuluIsEven: int = bitsZulu & 1 ^ 1
            arcCodeAnalysis: int = (bitsZulu << 1 | bitsAlfa) << 2 | 3
            if arcCodeAnalysis < state.arcCodeMaximum:
                state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            if bitsAlfaHasArcs:
                arcCodeAnalysis = bitsAlfaIsEven << 1 | bitsAlfa >> 2 | bitsZulu << 3
                if arcCodeAnalysis < state.arcCodeMaximum:
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            if bitsZuluHasArcs:
                arcCodeAnalysis = bitsZuluIsEven | bitsAlfa << 2 | bitsZulu >> 1
                if arcCodeAnalysis < state.arcCodeMaximum:
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            if bitsAlfaHasArcs and bitsZuluHasArcs and (bitsAlfaIsEven or bitsZuluIsEven):
                if bitsAlfaIsEven and (not bitsZuluIsEven):
                    bitsAlfa ^= walkDyckPath(bitsAlfa)
                elif bitsZuluIsEven and (not bitsAlfaIsEven):
                    bitsZulu ^= walkDyckPath(bitsZulu)
                arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlfa) >> 2
                if arcCodeAnalysis < state.arcCodeMaximum:
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
        state.reduceBoundary()
        lookupArcCodeMeanders: dict[int, int] = state.lookupMeanders.copy()
        state.lookupMeanders = {}
        tuple(map(analyzeArcCode, lookupArcCodeMeanders.keys(), lookupArcCodeMeanders.values()))
        state = prune(state)
    return state
if TYPE_CHECKING:
    import pandas

def integersWidePandas吗(state: StateMeanders, dataframeAnalyzed: pandas.DataFrame) -> bool:
    """Determine whether the next transfer step exceeds the fixed-width integer capacity.

    (AI generated docstring)

    The current NumPy and pandas formulas need up to four bits beyond the widest `arcCode`.
    Each input produces at most four contributions, so four times the sum of active counts
    bounds every aggregated count in the next step. This bound can select Python integers
    earlier than an exact analysis of each output bucket would.

    `state.bitWidthLimit` counts usable nonnegative value bits. For example, unsigned 64-bit
    integers provide 64 value bits and signed 128-bit integers provide 127. The boundary
    threshold `state.arcCodeMaximum` remains a Python integer and consumes no buffer capacity.
    """
    bitWidthArcCode: int = int(dataframeAnalyzed['arcCodeAnalysis'].to_numpy().max(initial=0)).bit_length() + 4
    bitWidthMeanders: int = (4 * int(dataframeAnalyzed['meanders'].to_numpy().sum(dtype=object))).bit_length()
    return state.bitWidthLimit < max(bitWidthArcCode, bitWidthMeanders)

def pruneDataFrame(state: StateMeanders, dataframeAnalyzed: pandas.DataFrame) -> tuple[StateMeanders, pandas.DataFrame]:
    """Accumulate known diagonal contributions and retain the remaining dataframe rows.

    (AI generated docstring)

    Repeated arc codes contribute all their counts, summed as Python integers.
    The returned dataframe preserves row order and has a fresh consecutive index.
    The state retains completed contributions in `countAddend`; an empty dataframe
    sets `boundary` to zero. Retain both returned objects.
    """
    boundary: int = state.boundary + 1
    次diagonal: int = 1
    while 次diagonal <= min(boundary // 2, len(boxOfDiagonals)):
        arcCode: int = (1 << 2 * (boundary - 2 * 次diagonal) + 2) - 1
        subtotalMeanders: int = int(dataframeAnalyzed.loc[dataframeAnalyzed['arcCodeAnalysis'] == arcCode, 'meanders'].to_numpy().sum(dtype=object))
        dataframeAnalyzed = dataframeAnalyzed.loc[dataframeAnalyzed['arcCodeAnalysis'] != arcCode]
        state.countAddend += boxOfDiagonals[次diagonal - 1](boundary) * subtotalMeanders
        次diagonal += 1
    if not len(dataframeAnalyzed.index):
        state.boundary = 0
    return (state, dataframeAnalyzed.reset_index(drop=True))

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
    """Locate the first Dyck-balance failure bit in `intWithExtra_0b1`.

    (AI generated docstring)

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
    first index where the running balance is negative in the Dyck-prefix sense.

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
    """
    findTheExtra_0b1: int = 0
    flipExtra_0b1_Here: int = 1
    while 0 <= findTheExtra_0b1:
        flipExtra_0b1_Here <<= 2
        if intWithExtra_0b1 & flipExtra_0b1_Here == 0:
            findTheExtra_0b1 += 1
        else:
            findTheExtra_0b1 += neg(1)
    return flipExtra_0b1_Here

def count(state: StateMeanders) -> StateMeanders:
    """Advance one meander transfer-matrix computation until `state.boundary` reaches zero.

    (AI generated docstring)

    You can use `count` to apply all transition rules for each boundary layer in `state` and update
    `state.lookupMeanders` in place. The transition set includes new-arc insertion, left-right
    crossing moves, and Dyck-path-based arc joining through `walkDyckPath`.

    Parameters
    ----------
    state : StateMeanders
        The algorithm state.

    Returns
    -------
    state : StateMeanders
        The same `state` instance after all boundary layers have been processed.

    Transfer Steps
    --------------
    The function decrements `state.boundary`, consumes each previous `arcCode`, and accumulates the
    next boundary dictionary. The transition for joining arches conditionally flips one Dyck-matched
    endpoint bit before packing the next `arcCode`.

    """
    state = prune(state)
    while 0 < state.boundary:

        def analyzeArcCode(arcCode: int, meanders: int, state: StateMeanders=state) -> None:
            bitsAlfa: int = arcCode & state.bitsLocator
            bitsAlfaHasArcs: bool = 1 < bitsAlfa
            bitsAlfaIsEven: int = bitsAlfa & 1 ^ 1
            bitsZulu: int = arcCode >> 1 & state.bitsLocator
            bitsZuluHasArcs: bool = 1 < bitsZulu
            bitsZuluIsEven: int = bitsZulu & 1 ^ 1
            arcCodeAnalysis: int = (bitsZulu << 1 | bitsAlfa) << 2 | 3
            if arcCodeAnalysis < state.arcCodeMaximum:
                state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            if bitsAlfaHasArcs:
                arcCodeAnalysis = bitsAlfaIsEven << 1 | bitsAlfa >> 2 | bitsZulu << 3
                if arcCodeAnalysis < state.arcCodeMaximum:
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            if bitsZuluHasArcs:
                arcCodeAnalysis = bitsZuluIsEven | bitsAlfa << 2 | bitsZulu >> 1
                if arcCodeAnalysis < state.arcCodeMaximum:
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            if bitsAlfaHasArcs and bitsZuluHasArcs and (bitsAlfaIsEven or bitsZuluIsEven):
                if bitsAlfaIsEven and (not bitsZuluIsEven):
                    bitsAlfa ^= walkDyckPath(bitsAlfa)
                elif bitsZuluIsEven and (not bitsAlfaIsEven):
                    bitsZulu ^= walkDyckPath(bitsZulu)
                arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlfa) >> 2
                if arcCodeAnalysis < state.arcCodeMaximum:
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
        state.reduceBoundary()
        lookupArcCodeMeanders: dict[int, int] = state.lookupMeanders.copy()
        state.lookupMeanders = {}
        tuple(map(analyzeArcCode, lookupArcCodeMeanders.keys(), lookupArcCodeMeanders.values()))
        state = prune(state)
    return state

def doTheNeedful(state: StateMeanders) -> StateMeanders:
    """Compute the total meander count encoded in `state.lookupMeanders`.

    You can use `doTheNeedful` as the meander transfer-matrix entry point for the `matrixMeanders`
    flow selected by `mapFolding.basecamp.countMeanders`. The function runs `count(state)` and
    returns the final state for the semi-meander and meandric counting context
    documented in OEIS entries A000682 and A005316 [1][2], the Jensen transfer-matrix method [3],
    and implementation lineages by Howroyd and Irvine [4][5].

    Parameters
    ----------
    state : StateMeanders
        The algorithm state.

    Returns
    -------
    state : StateMeanders
        The completed meander transfer state.

    See Also
    --------
    `mapFolding.basecamp.countMeanders`
        Resolve the meander family and select the counting implementation.

    References
    ----------
    [1] OEIS A000682, Semi-meanders.
        https://oeis.org/A000682
    [2] OEIS A005316, Meandric numbers.
        https://oeis.org/A005316
    [3] Jensen, I. (2000). A transfer matrix approach to the enumeration of plane meanders.
        Journal of Physics A: Mathematical and General, 33(34), 5953-5963.
        https://dx.doi.org/10.1088/0305-4470/33/34/301
    [4] Howroyd, A. (2015). C# Software for the Enumeration of Meanders.
        https://oeis.org/A005316/a005316.cs.txt
    [5] Irvine, S. A. (Java port). `A005316.java` in `archmageirvine/joeis`.
        https://github.com/archmageirvine/joeis/blob/5dc2148344bf/src/irvine/oeis/a005/A005316.java
    """
    return count(state)
