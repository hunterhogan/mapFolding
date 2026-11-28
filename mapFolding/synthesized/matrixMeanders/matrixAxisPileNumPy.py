"""Count symmetric meander shapes by the pile cut with the NumPy transfer.

(AI generated docstring)

The module adapts `mapFolding.algorithms.matrixAxisPile`.
`mapFolding.kitAST.matrixMeanders.makeModulesAxis.makeAxisPileNumPy` generates this module.
Regenerate the module instead of editing it.

Contents
--------
Functions
    count
        Count meanders with transfer matrix algorithm implemented in NumPy (*Num*erical *Py*thon).
    doTheNeedful
        Compute `meanders` with a transfer matrix algorithm implemented in NumPy.
    getBitsLocator
        Make the mask that selects the bits of one side of the road in an `arcCode`.
    isAxisMirrored吗
        Decide whether the symmetry axis can match the strands of the two sides of an `arcCode`.
    isHalfJoined吗
        Decide whether the half that an `arcCode` describes joins with its half-turn image.
    makeDataContainer
        Create a `numpy.ndarray` of `shape` with `datatype` for matrix-meander computation.
    makeLookupPartner
        Pair each terminal on the cut with the other end of its arc, in circle order.
    pairTerminals
        Pair the terminals of one side that share an arc, and collect the crossing terminals.
    pruneArray
        Move the counts of arc codes with a known formula into `state.countAddend`.
    recordAnalysis
        Record retained `arcCodeAnalysis` values and their counts in a possibly extended `arrayAnalyzed`.
    reflectArcCode
        Reflect an `arcCode` across the road.
"""
from __future__ import annotations

from gc import collect as goByeBye
from mapFolding.algorithms.matrixAxisStrip import isJoined吗
from mapFolding.algorithms.matrixMeandersShare import getTotalBuckets, integersWide吗
from mapFolding.dataStructures import extendArray, make_memmap, ShapeArray, ShapeSlicer
from mapFolding.oeis import boxOfDiagonals
from mapFolding.synthesized.matrixMeanders.matrixMeandersShare import countBigInt, flipDyckEndpoint, integersWideNumPy吗
from mapFolding.theTypes import 形ArcCode
from numpy import (
	array, bitwise_and as Xand, bitwise_left_shift as XshiftLeft, bitwise_or as X_or, bitwise_right_shift as XshiftRight, bitwise_xor as Xxor,
	bool as numpy_bool, greater as moreThan, less_equal as lessThanEqual, memmap, multiply, subtract)
from operator import add, itemgetter, mul
from tqdm.auto import tqdm
from typing import cast, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
    from collections.abc import Iterator
    from mapFolding.dataBaskets import StateMeanders
    from mapFolding.theTypes import 形ArrayArcCode, 形ArrayArcCode1D, 形ArrayBoolean1D, 形ArraySelector1D, 形NumPyInteger
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
    """Move the counts of arc codes with a known formula into `state.countAddend`.

    (AI generated docstring)

    With b = `state.boundary + 1`, for each diagonal d ≤ b / 2 with a formula in
    `mapFolding.oeis.boxOfDiagonals`, the arc code 2^(2(b − 2d) + 2) − 1 contributes the formula at b
    times its count. The function then removes every arc code with count zero.

    Parameters
    ----------
    state : StateMeanders
        Transfer state. `state.boundary` selects the diagonals.
    arrayMeanders : 形ArrayArcCode
        Arc codes in ascending order with their counts.

    Returns
    -------
    state : StateMeanders
        The same state with the contributions in `state.countAddend`. `state.boundary` is 0 when no
        arc code remains.
    arrayMeanders : 形ArrayArcCode
        The arc codes that remain, with their counts.
    """
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
    arrayMeanders: 形ArrayArcCode = makeDataContainer(shape, 形ArcCode)
    del shape
    arrayMeanders[slicerArcCode] = array(list(state.lookupMeanders.keys()), dtype=形ArcCode)
    arrayMeanders[slicerMeanders] = array(list(state.lookupMeanders.values()), dtype=形ArcCode)
    state.lookupMeanders = {}
    tqdmBoundary: tqdm = tqdm(total=state.n, initial=state.n - state.boundary, postfix={'boundary': state.boundary}, disable=state.n < 28)
    while 0 < state.boundary and (not integersWideNumPy吗(state, arrayMeanders)):
        state.bitWidthArcCode = int(arrayMeanders[slicerArcCode].max(initial=0)).bit_length()
        state.setBitsLocator()
        shape = ShapeArray(length=getTotalBuckets(state, len(arrayMeanders[slicerArcCode])), indexes=indexesAnalyzed)
        arrayAnalyzed: 形ArrayArcCode = makeDataContainer(shape, 形ArcCode)
        del shape
        shape = ShapeArray(length=len(arrayMeanders[slicerArcCode]), indexes=indexesWorkbench)
        arrayWorkbench: 形ArrayArcCode = makeDataContainer(shape, 形ArcCode)
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
        arrayMeanders = makeDataContainer(shape, 形ArcCode)
        del shape
        arrayMeanders[slicerArcCode] = unique.values
        arrayMeanders[slicerMeanders] = 0
        numpy.add.at(arrayMeanders[slicerMeanders], unique.inverse_indices, arrayAnalyzed[slicerMeanders])
        del unique
        del arrayAnalyzed
        tqdmBoundary.update()
    tqdmBoundary.close()
    state.lookupMeanders = dict(zip(map(int, arrayMeanders[slicerArcCode]), map(int, arrayMeanders[slicerMeanders]), strict=True))
    return state

def makeDataContainer(shape: tuple[Any, ...], datatype: type[形NumPyInteger]) -> ndarray[tuple[Any, ...], dtype[形NumPyInteger]]:
    """Create a `numpy.ndarray` of `shape` with `datatype` for matrix-meander computation.

    Parameters
    ----------
    shape : tuple[Any, ...]
        Shape of the `ndarray`.
    datatype : type[形NumPyInteger]
        Integer `dtype` used for each array element.

    Returns
    -------
    container : ndarray[tuple[Any, ...], dtype[形NumPyInteger]]
        File-backed array created by `mapFolding.dataStructures.make_memmap`. The operating
        system deletes the storage after the array and every view of it are released.

    """
    return make_memmap(shape, datatype)

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
    while 0 < state.boundary:
        if integersWide吗(state):
            state = countBigInt(state)
        else:
            state = count(state)
    state.lookupMeanders = dict(filter(itemgetter(1), state.lookupMeanders.items()))
    state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
    state.lookupMeanders = {}
    return state

def getBitsLocator(arcCode: int) -> int:
    """Make the mask that selects the bits of one side of the road in an `arcCode`.

    (AI generated docstring)

    Parameters
    ----------
    arcCode : int
        State of the pile-cut transfer.

    Returns
    -------
    bitsLocator : int
        Mask `0b…010101` with a set bit at each even position up to the length of `arcCode`. `arcCode
        & bitsLocator` is `bitsAlfa`, and `arcCode >> 1 & bitsLocator` is `bitsZulu`.
    """
    return (1 << (arcCode.bit_length() | 1) + 1) // 3

def reflectArcCode(arcCode: int) -> int:
    """Reflect an `arcCode` across the road.

    (AI generated docstring)

    The reflection σ exchanges `bitsAlfa` and `bitsZulu`.

    Parameters
    ----------
    arcCode : int
        State of the pile-cut transfer.

    Returns
    -------
    arcCodeReflected : int
        State of the reflected half.
    """
    bitsLocator: int = getBitsLocator(arcCode)
    return (arcCode & bitsLocator) << 1 | arcCode >> 1 & bitsLocator

def pairTerminals(bitsSide: int, boxOfTerminals: Iterator[int], lookupPartner: list[int]) -> list[int]:
    """Pair the terminals of one side that share an arc, and collect the crossing terminals.

    (AI generated docstring)

    Each terminal of the side takes the next circle position from `boxOfTerminals`. A `0` bit is the
    inner end of an arc on the same side. A `1` bit is the outer end of that arc when an inner end is
    unmatched, and otherwise the end of an arc that crosses the road.

    Input Mutation
    --------------
    This function records each pair of the side in `lookupPartner`.

    Parameters
    ----------
    bitsSide : int
        `bitsAlfa` or `bitsZulu` of an `arcCode`, read from the road outward in steps of two bits.
    boxOfTerminals : Iterator[int]
        Circle position of each terminal of the side, in the order of `bitsSide`.
    lookupPartner : list[int]
        Partner of each circle position.

    Returns
    -------
    boxOfCross : list[int]
        Circle positions of the terminals of the side whose arcs cross the road, in the order of
        `bitsSide`.
    """
    boxOfOpen: list[int] = []
    boxOfCross: list[int] = []
    while 1 < bitsSide:
        terminal: int = next(boxOfTerminals)
        if bitsSide & 1 == 0:
            boxOfOpen.append(terminal)
        elif boxOfOpen:
            lookupPartner[terminal] = boxOfOpen.pop()
            lookupPartner[lookupPartner[terminal]] = terminal
        else:
            boxOfCross.append(terminal)
        bitsSide >>= 2
    return boxOfCross

def makeLookupPartner(arcCode: int) -> list[int]:
    """Pair each terminal on the cut with the other end of its arc, in circle order.

    (AI generated docstring)

    The circle order lists the terminals of `bitsAlfa` from the road outward and then the terminals of
    `bitsZulu` from the outside to the road. The k-th crossing terminal of `bitsAlfa` pairs with the
    k-th crossing terminal of `bitsZulu`.

    Parameters
    ----------
    arcCode : int
        State of the pile-cut transfer at the symmetry axis.

    Returns
    -------
    lookupPartner : list[int]
        List of N circle positions. `lookupPartner[terminal]` is the position of the terminal that
        shares an arc with `terminal`.
    """
    bitsLocator: int = getBitsLocator(arcCode)
    bitsAlfa: int = arcCode & bitsLocator
    bitsZulu: int = arcCode >> 1 & bitsLocator
    lookupPartner: list[int] = list(range((bitsAlfa.bit_length() + bitsZulu.bit_length()) // 2 - 1))
    boxOfCrossAlfa: list[int] = pairTerminals(bitsAlfa, iter(range(len(lookupPartner))), lookupPartner)
    boxOfCrossZulu: list[int] = pairTerminals(bitsZulu, iter(range(len(lookupPartner) - 1, -1, -1)), lookupPartner)
    tuple(map(lookupPartner.__setitem__, boxOfCrossAlfa, boxOfCrossZulu, strict=True))
    tuple(map(lookupPartner.__setitem__, boxOfCrossZulu, boxOfCrossAlfa, strict=True))
    return lookupPartner

def isAxisMirrored吗(arcCode: int) -> bool:
    """Decide whether the symmetry axis can match the strands of the two sides of an `arcCode`.

    (AI generated docstring)

    With L_alfa strands on the side of `bitsAlfa` and L_zulu strands on the side of `bitsZulu`, the
    symmetry axis can match the sides when L_alfa − L_zulu ∈ {0, 2}.

    Parameters
    ----------
    arcCode : int
        State of the pile-cut transfer at the symmetry axis.

    Returns
    -------
    isAxisMirrored : bool
        `True` when L_alfa − L_zulu ∈ {0, 2}.
    """
    bitsLocator: int = getBitsLocator(arcCode)
    return (arcCode >> 1 & bitsLocator).bit_length() <= (arcCode & bitsLocator).bit_length() <= (arcCode >> 1 & bitsLocator).bit_length() + 4

def isHalfJoined吗(arcCode: int) -> bool:
    """Decide whether the half that an `arcCode` describes joins with its half-turn image.

    (AI generated docstring)

    The test takes the center terminal on the side of `bitsAlfa`. The half joins when
    `isAxisMirrored吗` accepts the numbers of strands and
    `mapFolding.algorithms.matrixAxisStrip.isJoined吗` accepts the circle order.

    Parameters
    ----------
    arcCode : int
        State of the pile-cut transfer at the symmetry axis.

    Returns
    -------
    isHalfJoined : bool
        `True` when the half and its half-turn image form one symmetric meander shape.
    """
    return isAxisMirrored吗(arcCode) and isJoined吗(makeLookupPartner(arcCode))
