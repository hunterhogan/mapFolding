from gc import collect as goByeBye
from mapFolding import ShapeArray, ShapeSlicer
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from numpy import (
	bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, greater, less_equal, multiply, subtract)
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from numpy.lib._arraysetops_impl import UniqueInverseResult

indicesPrepArea: int = 1
indexAnalysis = 0
slicerAnalysis: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalysis)

indicesAnalyzed: int = 2
indexArcCode, indexCrossings = range(indicesAnalyzed)
slicerArcCode: ShapeSlicer = ShapeSlicer(length=..., indices=indexArcCode)
slicerCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexCrossings)

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
			"""Create new `arrayMeanders` by deduplicating `arcCode` and summing `crossings`."""
			unique: UniqueInverseResult[numpy.uint64] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

			shape = ShapeArray(length=len(unique.values), indices=state.indicesMeanders)
			state.arrayMeanders = numpy.zeros(shape, dtype=state.datatypeArcCode)
			del shape

			state.arrayMeanders[state.slicerArcCode] = unique.values
			numpy.add.at(state.arrayMeanders[state.slicerCrossings], unique.inverse_indices, arrayAnalyzed[slicerCrossings])
			del unique

			return state

		def bitsAlpha(state: MatrixMeandersNumPyState) -> NDArray[numpy.uint64]:
			return bitwise_and(state.arrayMeanders[state.slicerArcCode], state.locatorBits)

		def bitsZulu(state: MatrixMeandersNumPyState) -> NDArray[numpy.uint64]:
			IsThisMemoryEfficient: NDArray[numpy.uint64] = bitwise_right_shift(state.arrayMeanders[state.slicerArcCode], 1)
			return bitwise_and(IsThisMemoryEfficient, state.locatorBits)

		def makeStorage[个: numpy.integer[Any]](dataTarget: NDArray[个], state: MatrixMeandersNumPyState, storageTarget: NDArray[numpy.uint64], indexAssignment: int = indexArcCode) -> NDArray[个]:
			"""Store `dataTarget` in `storageTarget` on `indexAssignment` if there is enough space, otherwise allocate a new array."""
			lengthStorageTarget: int = len(storageTarget)
			storageAvailable: int = lengthStorageTarget - state.indexTarget
			lengthDataTarget: int = len(dataTarget)

			if storageAvailable >= lengthDataTarget:
				indexStart: int = lengthStorageTarget - lengthDataTarget
				sliceStorage: slice = slice(indexStart, lengthStorageTarget)
				del indexStart
				slicerStorageAtIndex: ShapeSlicer = ShapeSlicer(length=sliceStorage, indices=indexAssignment)
				del sliceStorage
				storageTarget[slicerStorageAtIndex] = dataTarget.copy()
				arrayStorage = storageTarget[slicerStorageAtIndex].view() # pyright: ignore[reportAssignmentType]
				del slicerStorageAtIndex
			else:
				arrayStorage: NDArray[个] = dataTarget.copy()

			del storageAvailable, lengthDataTarget, lengthStorageTarget

			return arrayStorage

		def recordAnalysis(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState, arcCode: NDArray[numpy.uint64]) -> MatrixMeandersNumPyState:
			"""Record valid `arcCode` and corresponding `crossings` in `arrayAnalyzed`."""
			selectorOverLimit = arcCode > state.MAXIMUMarcCode
			arcCode[selectorOverLimit] = 0
			del selectorOverLimit

			selectorAnalysis: NDArray[numpy.intp] = numpy.flatnonzero(arcCode)

			indexStop: int = state.indexTarget + len(selectorAnalysis)
			sliceNonzero: slice = slice(state.indexTarget, indexStop)
			state.indexTarget = indexStop
			del indexStop

			slicerArcCodeNonzero = ShapeSlicer(length=sliceNonzero, indices=indexArcCode)
			slicerCrossingsNonzero = ShapeSlicer(length=sliceNonzero, indices=indexCrossings)
			del sliceNonzero

			arrayAnalyzed[slicerArcCodeNonzero] = arcCode[selectorAnalysis]
			del slicerArcCodeNonzero

			arrayAnalyzed[slicerCrossingsNonzero] = state.arrayMeanders[state.slicerCrossings][selectorAnalysis]
			del slicerCrossingsNonzero, selectorAnalysis

			return state

# TODO bitwidth should be automatic.
		state.bitWidth = int(state.arrayMeanders[state.slicerArcCode].max()).bit_length()

		lengthArrayAnalyzed: int = getBucketsTotal(state, 1.2)
		shape = ShapeArray(length=lengthArrayAnalyzed, indices=indicesAnalyzed)
		del lengthArrayAnalyzed
		goByeBye()

		arrayAnalyzed: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
		del shape

		shape = ShapeArray(length=len(state.arrayMeanders[state.slicerArcCode]), indices=indicesPrepArea)
		arrayPrepArea: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
		del shape

		prepArea: NDArray[numpy.uint64] = arrayPrepArea[slicerAnalysis].view()

		state.indexTarget = 0

		state.kOfMatrix -= 1

# =============== analyze aligned ===== if bitsAlpha > 1 and bitsZulu > 1 =============================================
# ======= > * > bitsAlpha 1 bitsZulu 1 ====================
		greater(bitsAlpha(state), 1, out=prepArea)
		multiply(bitsZulu(state), prepArea, out=prepArea)
		greater(prepArea, 1, out=prepArea)
		selectorGreaterThan1: NDArray[numpy.uint64] = makeStorage(prepArea, state, arrayAnalyzed, indexArcCode)

# ======= if bitsAlphaAtEven and not bitsZuluAtEven =======
# ======= ^ & | ^ & bitsZulu 1 1 bitsAlpha 1 1 ============
		bitwise_and(bitsZulu(state), 1, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)
		bitwise_or(bitsAlpha(state), prepArea, out=prepArea)
		bitwise_and(prepArea, 1, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea)
		selectorAlignAlpha: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexCrossings)

		arrayBitsAlpha: NDArray[numpy.uint64] = bitsAlpha(state)

		arrayBitsAlpha[selectorAlignAlpha] = flipTheExtra_0b1AsUfunc(arrayBitsAlpha[selectorAlignAlpha])
		del selectorAlignAlpha 													# NOTE FREE indexCrossings

# ======= if bitsZuluAtEven and not bitsAlphaAtEven =======
# ======= ^ & | ^ & bitsAlpha 1 1 bitsZulu 1 1 ============
		bitwise_and(bitsAlpha(state), 1, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)
		bitwise_or(bitsZulu(state), prepArea, out=prepArea)
		bitwise_and(prepArea, 1, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea)
		selectorAlignZulu: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexCrossings)

# ======= bitsAlphaAtEven or bitsZuluAtEven ===============
# ======= ^ & & bitsAlpha 1 bitsZulu 1 ====================
		bitwise_and(bitsAlpha(state), 1, out=prepArea)
		bitwise_and(bitsZulu(state), prepArea, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea) # selectorBitsAtEven
		del selectorGreaterThan1 												# NOTE FREE indexArcCode
		bitwise_xor(prepArea, 1, out=prepArea)
		selectorDisqualified: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexArcCode)

		prepArea = bitsZulu(state)

		prepArea[selectorAlignZulu] = flipTheExtra_0b1AsUfunc(prepArea[selectorAlignZulu])
		del selectorAlignZulu 													# NOTE FREE indexCrossings

		arrayBitsZulu: NDArray[numpy.uint64] = makeStorage(prepArea, state, arrayAnalyzed, indexCrossings)

# ======= (((bitsZulu >> 2) << 3) | bitsAlpha) >> 2 =======
# ======= >> | << >> bitsZulu 2 3 bitsAlpha 2 =============
		bitwise_right_shift(arrayBitsZulu, 2, out=prepArea)
		del arrayBitsZulu 														# NOTE FREE indexCrossings
		bitwise_left_shift(prepArea, 3, out=prepArea)
		bitwise_or(arrayBitsAlpha, prepArea, out=prepArea)
		del arrayBitsAlpha
		bitwise_right_shift(prepArea, 2, out=prepArea)

		prepArea[selectorDisqualified] = 0
		del selectorDisqualified 												# NOTE FREE indexArcCode

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze bitsAlpha ------- (bitsAlpha >> 2) | (bitsZulu << 3) | ((1 - (bitsAlpha & 1)) << 1) ---------
# ------- >> | << | (<< - 1 & bitsAlpha 1 1) << bitsZulu 3 2 bitsAlpha 2 ----------
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(bitsAlpha(state), state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlphaStack, 1, out=bitsAlphaStack)
		subtract(1, bitsAlphaStack, out=bitsAlphaStack)
		bitwise_left_shift(bitsAlphaStack, 1, out=bitsAlphaStack)
		bitwise_left_shift(bitsZulu(state), 3, out=prepArea)
		bitwise_or(bitsAlphaStack, prepArea, out=prepArea)
		del bitsAlphaStack 														# NOTE FREE indexCrossings
		bitwise_left_shift(prepArea, 2, out=prepArea)
		bitwise_or(bitsAlpha(state), prepArea, out=prepArea)
		bitwise_right_shift(prepArea, 2, out=prepArea)

# ------- if bitsAlpha > 1 ------------ > bitsAlpha 1 -----
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(bitsAlpha(state), state, arrayAnalyzed, indexCrossings)
		less_equal(bitsAlphaStack, 1, out=bitsAlphaStack)
		selectorUnderLimit: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(bitsAlphaStack), state, arrayAnalyzed, indexArcCode)
		prepArea[selectorUnderLimit] = 0
		del selectorUnderLimit 													# NOTE FREE indexArcCode

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze bitsZulu ---------- (bitsZulu >> 1) | (bitsAlpha << 2) | (1 - (bitsZulu & 1)) -------------
# ------- >> | << | (- 1 & bitsZulu 1) << bitsAlpha 2 1 bitsZulu 1 ----------
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(bitsZulu(state), state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsZuluStack, 1, out=bitsZuluStack)
		subtract(1, bitsZuluStack, out=bitsZuluStack)
		bitwise_left_shift(bitsAlpha(state), 2, out=prepArea)
		bitwise_or(bitsZuluStack, prepArea, out=prepArea)
		del bitsZuluStack 														# NOTE FREE indexCrossings
		bitwise_left_shift(prepArea, 1, out=prepArea)
		bitwise_or(bitsZulu(state), prepArea, out=prepArea)
		bitwise_right_shift(prepArea, 1, out=prepArea)

# ------- if bitsZulu > 1 ------------- > bitsZulu 1 ------
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(bitsZulu(state), state, arrayAnalyzed, indexCrossings)
		less_equal(bitsZuluStack, 1, out=bitsZuluStack)
		selectorUnderLimit = makeStorage(numpy.flatnonzero(bitsZuluStack), state, arrayAnalyzed, indexArcCode)
		del bitsZuluStack 														# NOTE FREE indexCrossings
		prepArea[selectorUnderLimit] = 0
		del selectorUnderLimit 													# NOTE FREE indexArcCode

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze simple ------------------------ ((bitsAlpha | (bitsZulu << 1)) << 2) | 3 ------------------
# ------- | << | bitsAlpha << bitsZulu 1 2 3 --------------
		bitwise_left_shift(bitsZulu(state), 1, out=prepArea)
		bitwise_or(bitsAlpha(state), prepArea, out=prepArea)
		bitwise_left_shift(prepArea, 2, out=prepArea)
		bitwise_or(prepArea, 3, out=prepArea)

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

		del prepArea, arrayPrepArea
# ----------------------------------------------- aggregation ---------------------------------------------------------
		arrayAnalyzed.resize((state.indexTarget, indicesAnalyzed))

		goByeBye()
		state = aggregateAnalyzed(arrayAnalyzed, state)

		del arrayAnalyzed

	return state

def doTheNeedful(state: MatrixMeandersNumPyState) -> int:
	"""Compute `crossings` with a transfer matrix algorithm implemented in NumPy.

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
	while state.kOfMatrix > 0:
		if areIntegersWide(state):
			state = countBigInt(state)
		else:
			state.makeArray()
			state = countNumPy(state)
			state.makeDictionary()
	return sum(state.dictionaryMeanders.values())
