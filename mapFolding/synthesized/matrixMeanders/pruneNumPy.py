from __future__ import annotations

from contextlib import suppress
from functools import cache
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding.dataBaskets import StateMeanders
from mapFolding.dataStructures import extendArray, make_memmap, ShapeArray, ShapeSlicer
from mapFolding.oeis.A400429oeisByFormula import boxOfDiagonals
from mapFolding.theTypes import 形ArcCode, 形ArrayArcCode
from numpy import (
	array, bitwise_and as Xand, bitwise_left_shift as XshiftLeft, bitwise_or as X_or, bitwise_right_shift as XshiftRight, bitwise_xor as Xxor,
	bool as numpy_bool, greater as moreThan, less_equal as lessThanEqual, memmap, multiply, subtract)
from operator import neg
from tqdm.auto import tqdm
from typing import cast, TYPE_CHECKING
import numpy
import pathlib

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
if TYPE_CHECKING:
    from mapFolding.theTypes import 形ArrayArcCode1D, 形ArrayBoolean1D, 形ArraySelector1D, 形NumPyInteger
    from numpy import dtype, ndarray
    from numpy.lib._arraysetops_impl import UniqueInverseResult
    from typing import Any
indexesAnalyzed: int = 2
次ArcCode, 次Meanders = range(indexesAnalyzed)
slicerArcCode: ShapeSlicer = ShapeSlicer(length=..., at=次ArcCode)
slicerMeanders: ShapeSlicer = ShapeSlicer(length=..., at=次Meanders)
indexesWorkbench: int = 3
次PrepArea, 次Alfa, 次Zulu = range(indexesWorkbench)
slicerPrepArea: ShapeSlicer = ShapeSlicer(length=..., at=次PrepArea)
slicerAlfa: ShapeSlicer = ShapeSlicer(length=..., at=次Alfa)
slicerZulu: ShapeSlicer = ShapeSlicer(length=..., at=次Zulu)

def pruneArray(state: StateMeanders, arrayMeanders: 形ArrayArcCode) -> tuple[StateMeanders, 形ArrayArcCode]:
    boundary: int = state.boundary + 1
    次diagonal: int = 1
    while 次diagonal <= min(boundary // 2, len(boxOfDiagonals)):
        arcCode: int = (1 << 2 * (boundary - 2 * 次diagonal) + 2) - 1
        次Known: int = int(numpy.searchsorted(arrayMeanders[slicerArcCode], arcCode))
        if 次Known < len(arrayMeanders[slicerArcCode]) and int(arrayMeanders[slicerArcCode][次Known]) == arcCode:
            state.countAddend += boxOfDiagonals[次diagonal - 1](boundary) * int(arrayMeanders[slicerMeanders][次Known])
            arrayMeanders[slicerMeanders][次Known] = 0
        次diagonal += 1
    selectorRemaining: 形ArraySelector1D = numpy.flatnonzero(arrayMeanders[slicerMeanders])
    arrayMeanders[slicerArcCode][0:len(selectorRemaining)] = arrayMeanders[slicerArcCode][selectorRemaining]
    arrayMeanders[slicerMeanders][0:len(selectorRemaining)] = arrayMeanders[slicerMeanders][selectorRemaining]
    arrayMeanders = arrayMeanders[0:len(selectorRemaining)]
    if not len(selectorRemaining):
        state.boundary = 0
    return (state, arrayMeanders)

def count(state: StateMeanders) -> StateMeanders:
    """Count meanders with transfer matrix algorithm implemented in NumPy (*Num*erical *Py*thon).

    Parameters
    ----------
    state : StateMeanders
        The algorithm state.

    Returns
    -------
    state : StateMeanders
        Updated state including `boundary` and `arrayMeanders`.

    Notes
    -----
    This version is *relatively* slow for small values of `n` (*e.g.*, 3 seconds vs. 3 milliseconds)
    due to garbage collection. On the other hand, it uses less memory for extreme values of `n`, which
    makes it faster due to less disk swapping--as compared to the pandas implementation and other
    NumPy implementations I tried.
    """
    shape = ShapeArray(length=len(state.lookupMeanders), indexes=indexesAnalyzed)
    arrayMeanders: 形ArrayArcCode = makeDataContainer(shape, 形ArcCode, 'arrayMeanders')
    del shape
    arrayMeanders[slicerArcCode] = array(list(state.lookupMeanders.keys()), dtype=形ArcCode)
    arrayMeanders[slicerMeanders] = array(list(state.lookupMeanders.values()), dtype=形ArcCode)
    state.lookupMeanders = {}
    tqdmBoundary: tqdm = tqdm(total=state.n, initial=state.n - state.boundary, postfix={'boundary': state.boundary}, disable=state.n < 28)
    while 0 < state.boundary and (not integersWideNumPy吗(state, arrayMeanders)):
        state.bitWidthArcCode = int(arrayMeanders[slicerArcCode].max(initial=0)).bit_length()
        state.setBitsLocator()
        shape = ShapeArray(length=getTotalBuckets(state, len(arrayMeanders[slicerArcCode])), indexes=indexesAnalyzed)
        arrayAnalyzed: 形ArrayArcCode = makeDataContainer(shape, 形ArcCode, 'arrayAnalyzed')
        del shape
        shape = ShapeArray(length=len(arrayMeanders[slicerArcCode]), indexes=indexesWorkbench)
        arrayWorkbench: 形ArrayArcCode = makeDataContainer(shape, 形ArcCode, 'arrayPrepArea')
        del shape
        toPrepArea: 形ArrayArcCode1D = arrayWorkbench[slicerPrepArea].view()
        bitsAlfa: 形ArrayArcCode1D = arrayWorkbench[slicerAlfa].view()
        bitsZulu: 形ArrayArcCode1D = arrayWorkbench[slicerZulu].view()
        Xand(arrayMeanders[slicerArcCode], state.bitsLocator, out=bitsAlfa)
        XshiftRight(arrayMeanders[slicerArcCode], 1, out=bitsZulu)
        Xand(bitsZulu, state.bitsLocator, out=bitsZulu)
        if isinstance(arrayMeanders, memmap) and 2 ** 32 < arrayMeanders.nbytes:
            cast('memmap', arrayWorkbench).flush()
        state.次Target = 0
        state.boundary -= 1
        tqdmBoundary.set_postfix_str(f'boundary={state.boundary}')
        state.set_arcCodeMaximum()
        moreThan(bitsAlfa, 1, out=toPrepArea)
        multiply(bitsZulu, toPrepArea, out=toPrepArea)
        selectorGreaterThan1: 形ArrayBoolean1D = numpy.empty_like(toPrepArea, dtype=numpy_bool)
        moreThan(toPrepArea, 1, out=selectorGreaterThan1)
        Xand(bitsZulu, 1, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)
        X_or(bitsAlfa, toPrepArea, out=toPrepArea)
        Xand(toPrepArea, 1, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)
        Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
        arraySelectors: 形ArraySelector1D = numpy.flatnonzero(toPrepArea)
        bitsAlfaStack: 形ArrayArcCode1D = bitsAlfa.copy()
        bitsAlfaStack[arraySelectors] = flipDyckEndpoint(bitsAlfaStack[arraySelectors])
        del arraySelectors
        Xand(bitsAlfa, 1, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)
        X_or(bitsZulu, toPrepArea, out=toPrepArea)
        Xand(toPrepArea, 1, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)
        Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
        arraySelectors: 形ArraySelector1D = numpy.flatnonzero(toPrepArea)
        Xand(bitsZulu, bitsAlfa, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)
        Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
        del selectorGreaterThan1
        Xxor(toPrepArea, 1, out=toPrepArea)
        selectorDisqualified: 形ArraySelector1D = numpy.flatnonzero(toPrepArea)
        toPrepArea[:] = bitsZulu.copy()
        toPrepArea[arraySelectors] = flipDyckEndpoint(toPrepArea[arraySelectors])
        del arraySelectors
        XshiftRight(toPrepArea, 2, out=toPrepArea)
        XshiftLeft(toPrepArea, 3, out=toPrepArea)
        X_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
        del bitsAlfaStack
        XshiftRight(toPrepArea, 2, out=toPrepArea)
        toPrepArea[selectorDisqualified] = 0
        del selectorDisqualified
        arrayAnalyzed, state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders, state.arcCodeMaximum)
        bitsAlfaStack: 形ArrayArcCode1D = numpy.empty_like(arrayMeanders[slicerArcCode])
        Xand(bitsAlfa, 1, out=bitsAlfaStack)
        subtract(1, bitsAlfaStack, out=bitsAlfaStack)
        XshiftLeft(bitsAlfaStack, 1, out=bitsAlfaStack)
        XshiftLeft(bitsZulu, 3, out=toPrepArea)
        X_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
        del bitsAlfaStack
        XshiftLeft(toPrepArea, 2, out=toPrepArea)
        X_or(bitsAlfa, toPrepArea, out=toPrepArea)
        XshiftRight(toPrepArea, 2, out=toPrepArea)
        bitsAlfaStack: 形ArrayArcCode1D = numpy.empty_like(arrayMeanders[slicerArcCode])
        lessThanEqual(bitsAlfa, 1, out=bitsAlfaStack)
        arraySelectors: 形ArraySelector1D = numpy.flatnonzero(bitsAlfaStack)
        del bitsAlfaStack
        toPrepArea[arraySelectors] = 0
        del arraySelectors
        arrayAnalyzed, state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders, state.arcCodeMaximum)
        bitsZuluStack: 形ArrayArcCode1D = numpy.empty_like(arrayMeanders[slicerArcCode])
        Xand(bitsZulu, 1, out=bitsZuluStack)
        subtract(1, bitsZuluStack, out=bitsZuluStack)
        XshiftLeft(bitsAlfa, 2, out=toPrepArea)
        X_or(bitsZuluStack, toPrepArea, out=toPrepArea)
        del bitsZuluStack
        XshiftLeft(toPrepArea, 1, out=toPrepArea)
        X_or(bitsZulu, toPrepArea, out=toPrepArea)
        XshiftRight(toPrepArea, 1, out=toPrepArea)
        bitsZuluStack: 形ArrayArcCode1D = numpy.empty_like(arrayMeanders[slicerArcCode])
        lessThanEqual(bitsZulu, 1, out=bitsZuluStack)
        arraySelectors: 形ArraySelector1D = numpy.flatnonzero(bitsZuluStack)
        del bitsZuluStack
        toPrepArea[arraySelectors] = 0
        del arraySelectors
        arrayAnalyzed, state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders, state.arcCodeMaximum)
        XshiftLeft(bitsZulu, 1, out=toPrepArea)
        X_or(bitsAlfa, toPrepArea, out=toPrepArea)
        XshiftLeft(toPrepArea, 2, out=toPrepArea)
        X_or(toPrepArea, 3, out=toPrepArea)
        arrayAnalyzed, state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders, state.arcCodeMaximum)
        del bitsAlfa, bitsZulu, toPrepArea, arrayWorkbench
        del arrayMeanders
        if 2 ** 32 < arrayAnalyzed.nbytes:
            goByeBye()
        unique: UniqueInverseResult[形ArcCode] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])
        shape = ShapeArray(length=len(unique.values), indexes=indexesAnalyzed)
        arrayMeanders = makeDataContainer(shape, 形ArcCode, 'arrayMeanders')
        del shape
        arrayMeanders[slicerArcCode] = unique.values
        arrayMeanders[slicerMeanders] = 0
        numpy.add.at(arrayMeanders[slicerMeanders], unique.inverse_indices, arrayAnalyzed[slicerMeanders])
        del unique
        del arrayAnalyzed
        state, arrayMeanders = pruneArray(state, arrayMeanders)
        tqdmBoundary.update()
    tqdmBoundary.close()
    state.lookupMeanders = dict(zip(map(int, arrayMeanders[slicerArcCode]), map(int, arrayMeanders[slicerMeanders]), strict=True))
    if isinstance(arrayMeanders, memmap):
        del arrayMeanders
        with suppress(Exception):
            pathlib.Path('arrayMeanders.mM').unlink()
        with suppress(Exception):
            pathlib.Path('arrayAnalyzed.mM').unlink()
        with suppress(Exception):
            pathlib.Path('arrayPrepArea.mM').unlink()
    return state

def makeDataContainer(shape: tuple[Any, ...], datatype: type[形NumPyInteger], name: str | None=None) -> ndarray[tuple[Any, ...], dtype[形NumPyInteger]]:
    """Create a `numpy.ndarray` of `shape` with `datatype` for matrix-meander computation.

    Parameters
    ----------
    shape : tuple[Any, ...]
        Shape of the `ndarray`.
    datatype : type[形NumPyInteger]
        Integer `dtype` used for each array element.
    name : str | None = None
        Required filename stem for the current memory-mapped allocator. The default `None`
        does not supply a usable storage name.

    Returns
    -------
    container : ndarray[tuple[Any, ...], dtype[形NumPyInteger]]
        File-backed array created by `mapFolding.dataStructures.make_memmap`. Existing
        storage with the same filename is replaced.

    """
    return make_memmap(shape, datatype, raiseIfNone(name))

def recordAnalysis(arrayAnalyzed: 形ArrayArcCode, 次start: int, arcCodeAnalysis: 形ArrayArcCode1D, arrayMeanders: 形ArrayArcCode, arcCodeMaximum: int) -> tuple[形ArrayArcCode, int]:
    """Record retained `arcCodeAnalysis` values and their counts in a possibly extended `arrayAnalyzed`.

    (AI generated docstring)

    You can use this function during a transfer step to append new arch-code records. This
    function returns both the output array and the next unused record position. The caller
    must retain both returned values because the output array can be replaced.

    Parameters
    ----------
    arrayAnalyzed : 形ArrayArcCode
        Two-axis output buffer containing arch codes and counts in the configured columns.
    次start : int
        First output record position to write.
    arcCodeAnalysis : 形ArrayArcCode1D
        Candidate arch codes aligned with the records in `arrayMeanders`. This array is
        modified in place to zero codes greater than `arcCodeMaximum`.
    arrayMeanders : 形ArrayArcCode
        Source records containing the counts associated with each candidate arch code.
    arcCodeMaximum : int
        Largest retained code. Codes equal to this value remain eligible.

    Returns
    -------
    arrayAnalyzed : 形ArrayArcCode
        Updated output buffer, extended with `mapFolding.dataStructures.extendArray`
        when the retained records exceed the current allocation.
    次Stop : int
        Position immediately after the last written output record.

    Record Selection
    ----------------
    After disallowed codes are zeroed, `numpy.flatnonzero` selects all remaining nonzero
    codes. Zero codes are omitted. Each selected code and its source count are copied to
    consecutive output records without aggregation.

    """
    selectorOverLimit: 形ArrayBoolean1D = arcCodeMaximum < arcCodeAnalysis
    arcCodeAnalysis[selectorOverLimit] = 0
    selectorAnalysis: 形ArraySelector1D = numpy.flatnonzero(arcCodeAnalysis)
    次Stop: int = 次start + len(selectorAnalysis)
    if len(arrayAnalyzed) < 次Stop:
        arrayAnalyzed = extendArray(arrayAnalyzed, 次Stop)
    sliceAnalysis: slice = slice(次start, 次Stop)
    slicerArcCodeAnalysis = ShapeSlicer(length=sliceAnalysis, at=次ArcCode)
    slicerMeandersAnalysis = ShapeSlicer(length=sliceAnalysis, at=次Meanders)
    arrayAnalyzed[slicerArcCodeAnalysis] = arcCodeAnalysis[selectorAnalysis]
    arrayAnalyzed[slicerMeandersAnalysis] = arrayMeanders[slicerMeanders][selectorAnalysis]
    return (arrayAnalyzed, 次Stop)

def doTheNeedful(state: StateMeanders) -> StateMeanders:
    """Compute `meanders` with a transfer matrix algorithm implemented in NumPy.

    Parameters
    ----------
    state : StateMeanders
        The algorithm state.

    Returns
    -------
    state : StateMeanders
        The completed meander transfer state.
    """
    state = prune(state)
    while 0 < state.boundary:
        if integersWide吗(state):
            state = countBigInt(state)
        else:
            state = count(state)
    return state
