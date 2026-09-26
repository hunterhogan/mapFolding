from __future__ import annotations

from contextlib import suppress
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding.algorithms.matrixMeandersShare import flipTheExtra_0b1, getTotalBuckets, integersWide吗
from mapFolding.dataBaskets import ShapeArray, ShapeSlicer, StateMeanders
from mapFolding.dataStructures import make_memmap
from mapFolding.synthesized.matrixMeanders.bigInt import countBigInt
from mapFolding.theTypes import 形ArcCode
from numba import jit
from numpy import (
	array, bitwise_and as Xand, bitwise_left_shift as XshiftLeft, bitwise_or as X_or, bitwise_right_shift as XshiftRight, bitwise_xor as Xxor,
	bool as numpy_bool, greater as moreThan, less_equal as lessThanEqual, memmap, multiply, subtract)
from tqdm.auto import tqdm
from typing import cast, TYPE_CHECKING
import numpy
import pathlib

if TYPE_CHECKING:
	from mapFolding.theTypes import 形ArrayArcCode, 形ArrayArcCode1D, 形ArrayBoolean1D, 形ArraySelector1D, 形NumPyInteger
	from numpy import dtype, ndarray
	from numpy.lib._arraysetops_impl import UniqueInverseResult
	from typing import Any

indexesAnalyzed: int = 2
次ArcCode, 次Meanders = range(indexesAnalyzed)
slicerArcCode: ShapeSlicer = ShapeSlicer(length=..., axis=次ArcCode)
slicerMeanders: ShapeSlicer = ShapeSlicer(length=..., axis=次Meanders)

indexesWorkbench: int = 3
次PrepArea, 次Alfa, 次Zulu = range(indexesWorkbench)
slicerPrepArea: ShapeSlicer = ShapeSlicer(length=..., axis=次PrepArea)
slicerAlfa: ShapeSlicer = ShapeSlicer(length=..., axis=次Alfa)
slicerZulu: ShapeSlicer = ShapeSlicer(length=..., axis=次Zulu)

def makeDataContainer(shape: tuple[Any, ...], datatype: type[形NumPyInteger], name: str | None = None) -> ndarray[tuple[Any, ...], dtype[形NumPyInteger]]:
    """Create a `numpy.ndarray` of `shape` with `datatype` for matrix-meander computation.

    Parameters
    ----------
    shape : tuple[Any, ...]
        Shape of the `ndarray`.
    datatype : type[形NumPyInteger]
        Integer `dtype` used for each array element.
    name : str | None = None
        If applicable, filename stem `f"{name}.mM"` for a file based `ndarray`.

    Returns
    -------
    container : ndarray[tuple[Any, ...], dtype[形NumPyInteger]]
        `numpy.ndarray` of `shape` with `datatype`.
    """
    return make_memmap(shape, datatype, raiseIfNone(name))

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

    tqdmBoundary: tqdm = tqdm(total=state.n, initial=state.n - state.boundary, postfix={'boundary': state.boundary}, disable=False)
    while 0 < state.boundary and not integersWide吗(state, arrayMeanders=arrayMeanders):
        def recordAnalysis(arrayAnalyzed: 形ArrayArcCode, 次Target: int, arcCode: 形ArrayArcCode1D, arrayMeanders: 形ArrayArcCode) -> int:
            """Record valid `arcCode` and corresponding `meanders` in `arrayAnalyzed`."""
            selectorOverLimit: 形ArrayBoolean1D = state.arcCodeMAXIMUM < arcCode
            arcCode[selectorOverLimit] = 0
            del selectorOverLimit

            selectorAnalysis: 形ArraySelector1D = numpy.flatnonzero(arcCode)

            次Stop: int = 次Target + len(selectorAnalysis)
            sliceAnalysis: slice = slice(次Target, 次Stop)

            slicerArcCodeAnalysis = ShapeSlicer(length=sliceAnalysis, axis=次ArcCode)
            slicerMeandersAnalysis = ShapeSlicer(length=sliceAnalysis, axis=次Meanders)
            del sliceAnalysis

            arrayAnalyzed[slicerArcCodeAnalysis] = arcCode[selectorAnalysis]
            del slicerArcCodeAnalysis

            arrayAnalyzed[slicerMeandersAnalysis] = arrayMeanders[slicerMeanders][selectorAnalysis]
            del slicerMeandersAnalysis, selectorAnalysis

            return 次Stop

        state.setBitWidthNumPy(arrayMeanders)
        # TODO Reconfirm and document my decision to recreate `bitsLocator` in the NumPy version.
		# Reminder: unlike the baseline version that uses Python's dynamically-sized `int`, this uses
		# a fixed-size integer. I distinctly remember that when I created most of this module (today:
		# 2026 Sep 14; most of the work completed nine months ago), I reflected on the difference, and
		# I performed empirical tests measuring processing speed. But I don't remember any insights or
		# test results. I have a very vague memory that I couldn't find a performance difference, so I
		# decided to use the same system in all versions so I could keep the code simple and uniform.
        state.setBitsLocator()

        shape = ShapeArray(length=getTotalBuckets(state, len(arrayMeanders[slicerArcCode])), indexes=indexesAnalyzed)
        arrayAnalyzed: 形ArrayArcCode = makeDataContainer(shape, 形ArcCode, 'arrayAnalyzed')
        del shape

        shape = ShapeArray(length=len(arrayMeanders[slicerArcCode]), indexes=indexesWorkbench)
        arrayWorkbench: 形ArrayArcCode = makeDataContainer(shape, 形ArcCode, 'arrayPrepArea')
        del shape

        #=EndNotes##arrayWorkbench=
        toPrepArea: 形ArrayArcCode1D = arrayWorkbench[slicerPrepArea].view()
        bitsAlfa: 形ArrayArcCode1D = arrayWorkbench[slicerAlfa].view()
        bitsZulu: 形ArrayArcCode1D = arrayWorkbench[slicerZulu].view()

        Xand(arrayMeanders[slicerArcCode], state.bitsLocator, out=bitsAlfa)
        XshiftRight(arrayMeanders[slicerArcCode], 1, out=bitsZulu)
        Xand(bitsZulu, state.bitsLocator, out=bitsZulu)

        if isinstance(arrayMeanders, memmap):
            cast('memmap', arrayWorkbench).flush()

        state.次Target = 0

        state.boundary -= 1
        tqdmBoundary.set_postfix(boundary=state.boundary)  # pyright: ignore[reportUnknownMemberType]
        state.set_arcCodeMAXIMUM()

#================ analyze aligned ===== if 1 < bitsAlfa and 1 < bitsZulu =============================================
        #=EndNotes##analyzeArcCodesAligned=
#-------- < * < 1 bitsAlfa < 1 bitsZulu --------------------
        moreThan(bitsAlfa, 1, out=toPrepArea)

        multiply(bitsZulu, toPrepArea, out=toPrepArea)
        selectorGreaterThan1: 形ArrayBoolean1D = numpy.empty_like(toPrepArea, dtype=numpy_bool)
        moreThan(toPrepArea, 1, out=selectorGreaterThan1)

#-------- if bitsAlfaAtEven and not bitsZuluAtEven ------ #-------- ^ & | ^ & bitsZulu 1 1 bitsAlfa 1 1 ------------
        Xand(bitsZulu, 1, out=toPrepArea)

        Xxor(toPrepArea, 1, out=toPrepArea)
        X_or(bitsAlfa, toPrepArea, out=toPrepArea)
        Xand(toPrepArea, 1, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)

        Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
        arraySelectors: 形ArraySelector1D = numpy.flatnonzero(toPrepArea)

        bitsAlfaStack: 形ArrayArcCode1D = bitsAlfa.copy()
        bitsAlfaStack[arraySelectors] = flipTheExtra_0b1(bitsAlfaStack[arraySelectors])
        del arraySelectors

#-------- if bitsZuluAtEven and not bitsAlfaAtEven ------ #-------- ^ & | ^ & bitsAlfa 1 1 bitsZulu 1 1 ------------
        Xand(bitsAlfa, 1, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)
        X_or(bitsZulu, toPrepArea, out=toPrepArea)
        Xand(toPrepArea, 1, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)
        Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
        arraySelectors: 形ArraySelector1D = numpy.flatnonzero(toPrepArea)

#-------- bitsAlfaAtEven or bitsZuluAtEven -------------- #-------- ^ & & bitsAlfa 1 bitsZulu 1 --------------------
        Xand(bitsZulu, bitsAlfa, out=toPrepArea)
        Xxor(toPrepArea, 1, out=toPrepArea)

        Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
        del selectorGreaterThan1
        Xxor(toPrepArea, 1, out=toPrepArea)
        selectorDisqualified: 形ArraySelector1D = numpy.flatnonzero(toPrepArea)

        toPrepArea[:] = bitsZulu.copy()
        toPrepArea[arraySelectors] = flipTheExtra_0b1(toPrepArea[arraySelectors])
        del arraySelectors
        XshiftRight(toPrepArea, 2, out=toPrepArea)

#-------- (bitsZulu >> 2 << 3 | bitsAlfa) >> 2 ---------- #-------- >> | << >> bitsZulu 2 3 bitsAlfa 2 ------------

        XshiftLeft(toPrepArea, 3, out=toPrepArea)
        X_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
        del bitsAlfaStack
        XshiftRight(toPrepArea, 2, out=toPrepArea)

        toPrepArea[selectorDisqualified] = 0
        del selectorDisqualified

        state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)

#================== analyze bitsAlfa ====== (1 - (bitsAlfa & 1)) << 1 | bitsAlfa >> 2 | bitsZulu << 3 ========
        bitsAlfaStack: 形ArrayArcCode1D = numpy.empty_like(arrayMeanders[slicerArcCode])
#-------- >> | << | (<< - 1 & bitsAlfa 1 1) << bitsZulu 3 2 bitsAlfa 2 ----------
        Xand(bitsAlfa, 1, out=bitsAlfaStack)
        subtract(1, bitsAlfaStack, out=bitsAlfaStack)
        XshiftLeft(bitsAlfaStack, 1, out=bitsAlfaStack)

        XshiftLeft(bitsZulu, 3, out=toPrepArea)

        X_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
        del bitsAlfaStack
        XshiftLeft(toPrepArea, 2, out=toPrepArea)
        X_or(bitsAlfa, toPrepArea, out=toPrepArea)
        XshiftRight(toPrepArea, 2, out=toPrepArea)

#-------- if 1 < bitsAlfa ------------ < 1 bitsAlfa -----
        bitsAlfaStack: 形ArrayArcCode1D = numpy.empty_like(arrayMeanders[slicerArcCode])
        lessThanEqual(bitsAlfa, 1, out=bitsAlfaStack)
        arraySelectors: 形ArraySelector1D = numpy.flatnonzero(bitsAlfaStack)
        del bitsAlfaStack
        toPrepArea[arraySelectors] = 0
        del arraySelectors

        state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)

#================== analyze bitsZulu ========== (1 - (bitsZulu & 1)) | bitsAlfa << 2 | bitsZulu >> 1 ============
        bitsZuluStack: 形ArrayArcCode1D = numpy.empty_like(arrayMeanders[slicerArcCode])
#-------- >> | << | (- 1 & bitsZulu 1) << bitsAlfa 2 1 bitsZulu 1 ----------
        Xand(bitsZulu, 1, out=bitsZuluStack)
        subtract(1, bitsZuluStack, out=bitsZuluStack)

        XshiftLeft(bitsAlfa, 2, out=toPrepArea)

        X_or(bitsZuluStack, toPrepArea, out=toPrepArea)
        del bitsZuluStack
        XshiftLeft(toPrepArea, 1, out=toPrepArea)

        X_or(bitsZulu, toPrepArea, out=toPrepArea)
        XshiftRight(toPrepArea, 1, out=toPrepArea)

#-------- if 1 < bitsZulu ------------- < 1 bitsZulu ------
        bitsZuluStack: 形ArrayArcCode1D = numpy.empty_like(arrayMeanders[slicerArcCode])
        lessThanEqual(bitsZulu, 1, out=bitsZuluStack)
        arraySelectors: 形ArraySelector1D = numpy.flatnonzero(bitsZuluStack)
        del bitsZuluStack
        toPrepArea[arraySelectors] = 0
        del arraySelectors

        state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)

#================== analyze simple ======================= (bitsZulu << 1 | bitsAlfa) << 2 | 3 =======================
#-------- | << | bitsAlfa << bitsZulu 1 2 3 --------------
        XshiftLeft(bitsZulu, 1, out=toPrepArea)
        X_or(bitsAlfa, toPrepArea, out=toPrepArea)
        XshiftLeft(toPrepArea, 2, out=toPrepArea)
        X_or(toPrepArea, 3, out=toPrepArea)

        state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)

        del bitsAlfa, bitsZulu, toPrepArea, arrayWorkbench
#================================================ aggregation ========================================================-

        del arrayMeanders
        goByeBye()

        unique: UniqueInverseResult[形ArcCode] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

        shape = ShapeArray(length=len(unique.values), indexes=indexesAnalyzed)
        arrayMeanders = makeDataContainer(shape, 形ArcCode, 'arrayMeanders')
        del shape

        arrayMeanders[slicerArcCode] = unique.values
        arrayMeanders[slicerMeanders] = 0
        numpy.add.at(arrayMeanders[slicerMeanders], unique.inverse_indices, arrayAnalyzed[slicerMeanders])
        del unique

		# ruff: ignore[commented-out-code]
        # arrayAnalyzed, state.次Target = consolidateAnalyzed(arrayAnalyzed, state.次Target)
        # shape = ShapeArray(length=state.次Target, indexes=indexesAnalyzed)
        # arrayMeanders = makeDataContainer(shape, 形ArcCode, 'arrayMeanders')
        # del shape
        # arrayMeanders[:] = arrayAnalyzed[0:state.次Target]

        del arrayAnalyzed

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

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True, locals={})
def consolidateAnalyzed(arrayAnalyzed: 形ArrayArcCode, 次Stop: int) -> tuple[形ArrayArcCode, int]:
    # PAINFULLY slow.
	indexAnalyzed: int = 0
	indexConsolidated: int = 0
	#Custom compaction avoids the full-size index arrays required by NumPy's `unique`.
	while indexAnalyzed < 次Stop:
		arcCode: int = arrayAnalyzed[indexAnalyzed, 次ArcCode]
		meanders: int = 0
		while indexAnalyzed < 次Stop and arrayAnalyzed[indexAnalyzed, 次ArcCode] == arcCode:
			meanders += arrayAnalyzed[indexAnalyzed, 次Meanders]
			indexAnalyzed += 1
		arrayAnalyzed[indexConsolidated, 次ArcCode] = arcCode
		arrayAnalyzed[indexConsolidated, 次Meanders] = meanders
		indexConsolidated += 1
	return arrayAnalyzed, indexConsolidated

def doTheNeedful(state: StateMeanders) -> int:
    """Compute `meanders` with a transfer matrix algorithm implemented in NumPy.

    Parameters
    ----------
    state : StateMeanders
        The algorithm state.

    Returns
    -------
    meanders : int
        The computed value of `meanders`.
    """
    while 0 < state.boundary:
        if integersWide吗(state):
            state = countBigInt(state)
        else:
            state = count(state)
    return sum(state.lookupMeanders.values())
