"""NOTES.

`bitsAlpha`, `bitsZulu`, `viewStackAnalysis`, and `arrayPrepArea` are now completely abstract, so I can replace the underlying
data structures with anything. For example, `bitsAlpha` could be computed on the fly as needed instead of being stored in memory.
"""
from gc import collect
from mapFolding import axisOfLength, ShapeArray, ShapeSlicer
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from numpy import (
	bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, greater, logical_and, logical_not,
	multiply, subtract)
from numpy.typing import NDArray
from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from numpy.lib._arraysetops_impl import UniqueInverseResult

indicesPrepArea: int = 3
indexAnalysis, indexAlpha, indexZulu = range(indicesPrepArea)
slicerAnalysis: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalysis)
slicerAlpha: ShapeSlicer = ShapeSlicer(length=..., indices=indexAlpha)
slicerZulu: ShapeSlicer = ShapeSlicer(length=..., indices=indexZulu)

indicesAnalyzed: int = 2
indexAnalyzedArcCode, indexAnalyzedDistinctCrossings = range(indicesAnalyzed)
slicerAnalyzedArcCode: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalyzedArcCode)
slicerAnalyzedDistinctCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalyzedDistinctCrossings)

def countNumPy(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Count crossings with transfer matrix algorithm implemented in NumPy (*Num*erical *Py*thon).

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	state : MatrixMeandersState
		Updated state including `kOfMatrix` and `arrayMeanders`.
	"""
	while state.kOfMatrix > 0 and not areIntegersWide(state):
		def aggregateAnalyzed(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
			"""Create new `arrayMeanders` by deduplicating `arcCode` and summing `distinctCrossings`."""
			unique: UniqueInverseResult[numpy.uint64] = numpy.unique_inverse(arrayAnalyzed[slicerAnalyzedArcCode])

			shape = ShapeArray(length=len(unique.values), indices=state.indicesMeanders)
			state.arrayMeanders = numpy.zeros(shape, dtype=state.datatypeArcCode)
			del shape

			state.arrayMeanders[state.slicerArcCode] = unique.values
			numpy.add.at(state.arrayMeanders[state.slicerDistinctCrossings], unique.inverse_indices, arrayAnalyzed[slicerAnalyzedDistinctCrossings])
			del unique

			return state

		def makePrepArea(state: MatrixMeandersNumPyState) -> NDArray[numpy.uint64]:
			"""Create `arrayPrepArea`."""
			shape = ShapeArray(length=len(state.arrayMeanders[state.slicerArcCode]), indices=indicesPrepArea)
			arrayPrepArea: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
			del shape

			bitwise_and(state.arrayMeanders[state.slicerArcCode], state.locatorBits, out=arrayPrepArea[slicerAlpha])
			bitwise_right_shift(state.arrayMeanders[state.slicerArcCode], 1, out=arrayPrepArea[slicerZulu])
			bitwise_and(arrayPrepArea[slicerZulu], state.locatorBits, out=arrayPrepArea[slicerZulu])

			return arrayPrepArea

		def makeViews(state: MatrixMeandersNumPyState) -> tuple[NDArray[numpy.uint64], NDArray[numpy.uint64], NDArray[numpy.uint64]]:
			"""Create views of prep area."""
			arrayTarget: NDArray[numpy.uint64] = makePrepArea(state)
			viewStackAnalysis: NDArray[numpy.uint64] = arrayTarget[slicerAnalysis].view()
			viewAlpha: NDArray[numpy.uint64] = arrayTarget[slicerAlpha].view()
			viewZulu: NDArray[numpy.uint64] = arrayTarget[slicerZulu].view()
			return viewStackAnalysis, viewAlpha, viewZulu # match the order of `indicesPrepArea`

		def recordAnalysis(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState, arcCode: NDArray[numpy.uint64]) -> MatrixMeandersNumPyState:
			"""Record valid `arcCode` and corresponding `distinctCrossings` in `arrayAnalyzed`."""
			selectorOverLimit = arcCode > state.MAXIMUMarcCode
			multiply(arcCode, 0, out=arcCode, where=selectorOverLimit)
			del selectorOverLimit

			selectorNonzero: NDArray[numpy.intp] = numpy.flatnonzero(arcCode)

			indexStop: int = state.indexTarget + len(selectorNonzero)
			sliceNonzero: slice = slice(state.indexTarget, indexStop)
			state.indexTarget = indexStop
			del indexStop

			slicerArcCodeNonzero = ShapeSlicer(length=sliceNonzero, indices=indexAnalyzedArcCode)
			slicerDistinctCrossingsNonzero = ShapeSlicer(length=sliceNonzero, indices=indexAnalyzedDistinctCrossings)
			del sliceNonzero

			arrayAnalyzed[slicerArcCodeNonzero] = arcCode[selectorNonzero]
			del slicerArcCodeNonzero

			arrayAnalyzed[slicerDistinctCrossingsNonzero] = state.arrayMeanders[state.slicerDistinctCrossings][selectorNonzero]
			del slicerDistinctCrossingsNonzero, selectorNonzero

			return state

# TODO bitwidth should be automatic.
		state.bitWidth = int(state.arrayMeanders[state.slicerArcCode].max()).bit_length()

		lengthArrayAnalyzed: int = getBucketsTotal(state, 1.2)
		shape = ShapeArray(length=lengthArrayAnalyzed, indices=indicesAnalyzed)
		del lengthArrayAnalyzed
		arrayAnalyzed: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
		del shape

		# Just one explicit, long-form, annotated unpacking for demonstration and for the type checker.
		views: tuple[NDArray[numpy.uint64], NDArray[numpy.uint64], NDArray[numpy.uint64]] = makeViews(state)
		viewStackAnalysis: NDArray[numpy.uint64] = views[indexAnalysis]
		bitsAlpha: NDArray[numpy.uint64] = views[indexAlpha]
		bitsZulu: NDArray[numpy.uint64] = views[indexZulu]

		state.indexTarget = 0

		state.kOfMatrix -= 1

# ----------------- analyze simple ------------------------------------------------------------------------------------
# ------- ((bitsAlpha | (bitsZulu << 1)) << 2) | 3 --------
# ------- | << | bitsAlpha << bitsZulu 1 2 3 --------------
		bitwise_left_shift(bitsZulu, 1, out=viewStackAnalysis)
		bitwise_or(bitsAlpha, viewStackAnalysis, out=viewStackAnalysis)
		bitwise_left_shift(viewStackAnalysis, 2, out=viewStackAnalysis)
		bitwise_or(viewStackAnalysis, 3, out=viewStackAnalysis)

		state = recordAnalysis(arrayAnalyzed, state, viewStackAnalysis)

# ----------------- analyze bitsAlpha ---------------------------------------------------------------------------------
		stackInMemory: NDArray[numpy.uint64] = numpy.zeros_like(viewStackAnalysis)

# ------- (((((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3)) << 2) | bitsAlpha) >> 2 ---
# ------- >> | << | (<< - 1 & bitsAlpha 1 1) << bitsZulu 3 2 bitsAlpha 2 --------------
		bitwise_and(bitsAlpha, 1, out=stackInMemory)
		subtract(1, stackInMemory, out=stackInMemory)
		bitwise_left_shift(stackInMemory, 1, out=stackInMemory)
		bitwise_left_shift(bitsZulu, 3, out=viewStackAnalysis)
		bitwise_or(stackInMemory, viewStackAnalysis, out=viewStackAnalysis)
		del stackInMemory
		bitwise_left_shift(viewStackAnalysis, 2, out=viewStackAnalysis)
		bitwise_or(bitsAlpha, viewStackAnalysis, out=viewStackAnalysis)
		bitwise_right_shift(viewStackAnalysis, 2, out=viewStackAnalysis)

# ------- if bitsAlpha > 1 --------------------------------
		selectorUnderLimit: NDArray[numpy.bool_] = numpy.less_equal(bitsAlpha, 1, dtype=numpy.bool_)
		multiply(viewStackAnalysis, 0, out=viewStackAnalysis, where=selectorUnderLimit)
		del selectorUnderLimit

		state = recordAnalysis(arrayAnalyzed, state, viewStackAnalysis)

# ----------------- analyze bitsZulu ----------------------------------------------------------------------------------
		stackInMemory: NDArray[numpy.uint64] = numpy.zeros_like(viewStackAnalysis)

# ------- ((((1 - (bitsZulu & 1)) | (bitsAlpha << 2)) << 1) | bitsZulu) >> 1 ----
# ------- >> | << | (- 1 & bitsZulu 1) << bitsAlpha 2 1 bitsZulu 1 --------------
		bitwise_and(bitsZulu, 1, out=stackInMemory)
		subtract(1, stackInMemory, out=stackInMemory)
		bitwise_left_shift(bitsAlpha, 2, out=viewStackAnalysis)
		bitwise_or(stackInMemory, viewStackAnalysis, out=viewStackAnalysis)
		del stackInMemory
		bitwise_left_shift(viewStackAnalysis, 1, out=viewStackAnalysis)
		bitwise_or(bitsZulu, viewStackAnalysis, out=viewStackAnalysis)
		bitwise_right_shift(viewStackAnalysis, 1, out=viewStackAnalysis)

# ------- if bitsZulu > 1 ---------------------------------
		selectorUnderLimit: NDArray[numpy.bool_] = numpy.less_equal(bitsZulu, 1, dtype=numpy.bool_)
		multiply(viewStackAnalysis, 0, out=viewStackAnalysis, where=selectorUnderLimit)
		del selectorUnderLimit

		state = recordAnalysis(arrayAnalyzed, state, viewStackAnalysis)

# ================= analyze aligned ===================================================================================
# ======= if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven) =====
# ======= if bitsAlpha > 1 and bitsZulu > 1 ===============
		greater(bitsAlpha, 1, out=viewStackAnalysis)
		multiply(bitsZulu, viewStackAnalysis, out=viewStackAnalysis)
		greater(viewStackAnalysis, 1, out=viewStackAnalysis)

# ======= arrayMeanders resize(remove disqualified values) ====
# TODO shrink the array in place
		selectorKeepThese: NDArray[numpy.intp] = numpy.flatnonzero(viewStackAnalysis)
		state.arrayMeanders = numpy.take(state.arrayMeanders, selectorKeepThese, axis=axisOfLength)
		del selectorKeepThese

		del viewStackAnalysis, bitsAlpha, bitsZulu
		viewStackAnalysis, bitsAlpha, bitsZulu = makeViews(state)

# ======= if ... and (bitsAlphaIsEven or bitsZuluIsEven) ============
# ======= ^ & & bitsAlpha 1 bitsZulu 1 ============
		bitwise_and(bitsAlpha, 1, out=viewStackAnalysis)
		bitwise_and(bitsZulu, viewStackAnalysis, out=viewStackAnalysis)
		bitwise_xor(viewStackAnalysis, 1, out=viewStackAnalysis)

# ======= arrayMeanders resize(qualified values) ====
		selectorKeepThese = numpy.flatnonzero(viewStackAnalysis)
		state.arrayMeanders = numpy.take(state.arrayMeanders, selectorKeepThese, axis=axisOfLength)
		del selectorKeepThese

		del viewStackAnalysis, bitsAlpha, bitsZulu
		viewStackAnalysis, bitsAlpha, bitsZulu = makeViews(state)
		goByeBye()

# ======= align bitsAlpha and bitsZulu ========================================
# ======= if bitsAlphaAtEven and not bitsZuluAtEven =======
# ======= (1 - (bitsAlpha & 1)) & (bitsZulu & 1) ==========
		bitwise_and(bitsAlpha, 1, out=viewStackAnalysis)
		selectorAlphaAtOdd: NDArray[numpy.bool_] = viewStackAnalysis.astype(dtype=numpy.bool_)
		bitwise_and(bitsZulu, 1, out=viewStackAnalysis)
		selectorZuluAtOdd: NDArray[numpy.bool_] = viewStackAnalysis.astype(dtype=numpy.bool_)

		arrayBitsAlpha: NDArray[numpy.uint64] = bitsAlpha.copy()

		flipTheExtra_0b1AsUfunc(bitsAlpha, out=arrayBitsAlpha
			, where=logical_and(logical_not(selectorAlphaAtOdd), selectorZuluAtOdd)
			, casting='unsafe')

# ======= if bitsZuluAtEven and not bitsAlphaAtEven =======
# ======= (1 - (bitsZulu & 1)) & (bitsAlpha & 1) ==========
		arrayBitsZulu: NDArray[numpy.uint64] = bitsZulu.copy()

		flipTheExtra_0b1AsUfunc(bitsZulu, out=arrayBitsZulu
			, where=logical_and(selectorAlphaAtOdd, logical_not(selectorZuluAtOdd))
			, casting='unsafe')
		del selectorAlphaAtOdd
		del selectorZuluAtOdd

# ======= (((bitsZulu >> 2) << 3) | bitsAlpha) >> 2 =======
# ======= >> | << >> bitsZulu 2 3 bitsAlpha 2 =============
		bitwise_right_shift(arrayBitsZulu, 2, out=viewStackAnalysis)
		del arrayBitsZulu
		bitwise_left_shift(viewStackAnalysis, 3, out=viewStackAnalysis)
		bitwise_or(arrayBitsAlpha, viewStackAnalysis, out=viewStackAnalysis)
		del arrayBitsAlpha
		bitwise_right_shift(viewStackAnalysis, 2, out=viewStackAnalysis)

		state = recordAnalysis(arrayAnalyzed, state, viewStackAnalysis)

		del viewStackAnalysis, bitsAlpha, bitsZulu
		arrayAnalyzed.resize((state.indexTarget, indicesAnalyzed))

# ----------------------------------------------- aggregation ---------------------------------------------------------
		state = aggregateAnalyzed(arrayAnalyzed, state)

		del arrayAnalyzed
		goByeBye()

	return state

def doTheNeedful(state: MatrixMeandersNumPyState) -> int:
	"""Compute `distinctCrossings` with a transfer matrix algorithm implemented in NumPy.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	distinctCrossings : int
		The computed value of `distinctCrossings`.

	Notes
	-----
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex

	See Also
	--------
	https://oeis.org/A000682
	https://oeis.org/A005316
	"""
	while state.kOfMatrix > 0:
		if areIntegersWide(state):
			state = countBigInt(state)
		else:
			state.makeArray()
			goByeBye()
			state = countNumPy(state)
			state.makeDictionary()
			goByeBye()
	return sum(state.dictionaryMeanders.values())

# TODO FIXME
def goByeBye() -> None:
	if True:
		collect()

