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

"""More ideas for memory efficiency:
- In `recordAnalysis`, use `numpy.unique_inverse` on the newly added `arcCode`.
- In `recordAnalysis`, use `numpy.unique_inverse` on `arrayAnalyzed` after adding new `arcCode`.
- By deduplicating more often, I can decrease the allocated size of `arrayAnalyzed`. However, that may reduce the usefulness of `makeStorage`.
- I suspect that `makeStorage(numpy.flatnonzero(prepArea)...` contains a latent array.
- In every case (or almost every case) I use a selector, I _feel_ like there is more memory efficient way I don't know about.
- It's not clear to me whether or not I am instructing numpy to release _all_ memory I no longer need.
- Hypothetically, `arcCode` might be compressible, but IDK no nothing.
- `numpy.unique_inverse` uses a ton of memory, but I've failed to find a better way. BUT I might be able to improve
	`aggregateAnalyzed` now that `arrayArcCodes` is a single axis.
- For analyzeBitsAlpha and analyzeBitsZulu, find an equivalent formula that does not require a secondary stack.
- analyzeAligned requires a ton of memory. By analyzing it before the other three analyses (therefore `arrayAnalyzed` is empty)
	and using `makeStorage`, I've offset much of the usage, but I don't have confidence it's a good flow.

To mitigate memory problems:
- Put `arrayAnalyzed` in a `numpy.memmap`.
"""

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
		Updated state including `boundary` and `arrayMeanders`.
	"""
	while state.boundary > 0 and not areIntegersWide(state):
		def aggregateAnalyzed(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
			"""Create new `arrayMeanders` by deduplicating `arcCode` and summing `crossings`."""
			unique: UniqueInverseResult[numpy.uint64] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

			state.arrayArcCodes = unique.values
			state.arrayCrossings = numpy.zeros_like(state.arrayArcCodes, dtype=state.datatypeCrossings)
			numpy.add.at(state.arrayCrossings, unique.inverse_indices, arrayAnalyzed[slicerCrossings])
			del unique

			return state

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
			sliceAnalysis: slice = slice(state.indexTarget, indexStop)
			state.indexTarget = indexStop
			del indexStop

			slicerArcCodeAnalysis = ShapeSlicer(length=sliceAnalysis, indices=indexArcCode)
			slicerCrossingsAnalysis = ShapeSlicer(length=sliceAnalysis, indices=indexCrossings)
			del sliceAnalysis

			arrayAnalyzed[slicerArcCodeAnalysis] = arcCode[selectorAnalysis]
			del slicerArcCodeAnalysis

			arrayAnalyzed[slicerCrossingsAnalysis] = state.arrayCrossings[selectorAnalysis]
			del slicerCrossingsAnalysis, selectorAnalysis
			goByeBye()
			return state

		state.bitWidth = int(state.arrayArcCodes.max()).bit_length()

		lengthArrayAnalyzed: int = getBucketsTotal(state, 1.2)
		shape = ShapeArray(length=lengthArrayAnalyzed, indices=indicesAnalyzed)
		del lengthArrayAnalyzed
		goByeBye()

		arrayAnalyzed: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
		del shape

		shape = ShapeArray(length=len(state.arrayArcCodes), indices=indicesPrepArea)
		arrayPrepArea: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
		del shape

		prepArea: NDArray[numpy.uint64] = arrayPrepArea[slicerAnalysis].view()

		state.indexTarget = 0

		state.boundary -= 1

# =============== analyze aligned ===== if bitsAlpha > 1 and bitsZulu > 1 =============================================
		arrayBitsAlpha: NDArray[numpy.uint64] = bitwise_and(state.arrayArcCodes, state.locatorBits)	# NOTE extra array
# ======= > * > bitsAlpha 1 bitsZulu 1 ====================
		greater(arrayBitsAlpha, 1, out=prepArea)
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.locatorBits, out=bitsZuluStack)
		multiply(bitsZuluStack, prepArea, out=prepArea)
		greater(prepArea, 1, out=prepArea)
		selectorGreaterThan1: NDArray[numpy.uint64] = makeStorage(prepArea, state, arrayAnalyzed, indexArcCode)
																					# X indexArcCode X indexCrossings
# ======= if bitsAlphaAtEven and not bitsZuluAtEven =======
# ======= ^ & | ^ & bitsZulu 1 1 bitsAlpha 1 1 ============
		bitwise_and(bitsZuluStack, 1, out=prepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_xor(prepArea, 1, out=prepArea)
		bitwise_or(arrayBitsAlpha, prepArea, out=prepArea)
		bitwise_and(prepArea, 1, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea)
		selectorAlignAlpha: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexCrossings)
																					# X indexArcCode X indexCrossings
		arrayBitsAlpha[selectorAlignAlpha] = flipTheExtra_0b1AsUfunc(arrayBitsAlpha[selectorAlignAlpha])
		del selectorAlignAlpha 														# X indexArcCode O indexCrossings

# ======= if bitsZuluAtEven and not bitsAlphaAtEven =======
# ======= ^ & | ^ & bitsAlpha 1 1 bitsZulu 1 1 ============
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlphaStack, state.locatorBits, out=bitsAlphaStack)
		bitwise_and(bitsAlphaStack, 1, out=prepArea)
		del bitsAlphaStack 															# X indexArcCode O indexCrossings
		bitwise_xor(prepArea, 1, out=prepArea)
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)
		bitwise_and(bitsZuluStack, state.locatorBits, out=bitsZuluStack)
		bitwise_or(bitsZuluStack, prepArea, out=prepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_and(prepArea, 1, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea)
		selectorAlignZulu: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexCrossings)
																					# X indexArcCode X indexCrossings
# ======= bitsAlphaAtEven or bitsZuluAtEven ===============
		bitwise_and(state.arrayArcCodes, state.locatorBits, out=prepArea)
# ======= ^ & & bitsAlpha 1 bitsZulu 1 ====================
		bitwise_and(prepArea, 1, out=prepArea)
		sherpaBitsZulu: NDArray[numpy.uint64] = bitwise_right_shift(state.arrayArcCodes, 1) # NOTE 2° extra array
		bitwise_and(sherpaBitsZulu, state.locatorBits, out=sherpaBitsZulu)
		bitwise_and(sherpaBitsZulu, prepArea, out=prepArea)
		del sherpaBitsZulu															# NOTE del 2° extra array
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea) # selectorBitsAtEven
		del selectorGreaterThan1 													# O indexArcCode X indexCrossings
		bitwise_xor(prepArea, 1, out=prepArea)
		selectorDisqualified: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexArcCode)
																					# X indexArcCode X indexCrossings
		bitwise_right_shift(state.arrayArcCodes, 1, out=prepArea)
		bitwise_and(prepArea, state.locatorBits, out=prepArea)

		prepArea[selectorAlignZulu] = flipTheExtra_0b1AsUfunc(prepArea[selectorAlignZulu])
		del selectorAlignZulu 														# X indexArcCode O indexCrossings

		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(prepArea, state, arrayAnalyzed, indexCrossings)

# ======= (((bitsZulu >> 2) << 3) | bitsAlpha) >> 2 =======
# ======= >> | << >> bitsZulu 2 3 bitsAlpha 2 =============
		bitwise_right_shift(bitsZuluStack, 2, out=prepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_left_shift(prepArea, 3, out=prepArea)
		bitwise_or(arrayBitsAlpha, prepArea, out=prepArea)
		del arrayBitsAlpha															# NOTE del extra array
		bitwise_right_shift(prepArea, 2, out=prepArea)

		prepArea[selectorDisqualified] = 0
		del selectorDisqualified 													# O indexArcCode O indexCrossings

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze bitsAlpha ------- (bitsAlpha >> 2) | (bitsZulu << 3) | ((1 - (bitsAlpha & 1)) << 1) ---------
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.locatorBits, out=bitsAlphaStack)			# X indexArcCode O indexCrossings
# ------- >> | << | (<< - 1 & bitsAlpha 1 1) << bitsZulu 3 2 bitsAlpha 2 ----------
		bitwise_and(bitsAlphaStack, 1, out=bitsAlphaStack)
		subtract(1, bitsAlphaStack, out=bitsAlphaStack)
		bitwise_left_shift(bitsAlphaStack, 1, out=bitsAlphaStack)
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)
		bitwise_and(bitsZuluStack, state.locatorBits, out=bitsZuluStack)
		bitwise_left_shift(bitsZuluStack, 3, out=prepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_or(bitsAlphaStack, prepArea, out=prepArea)
		del bitsAlphaStack 															# O indexArcCode O indexCrossings
		bitwise_left_shift(prepArea, 2, out=prepArea)
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlphaStack, state.locatorBits, out=bitsAlphaStack)			# O indexArcCode X indexCrossings
		bitwise_or(bitsAlphaStack, prepArea, out=prepArea)
		bitwise_right_shift(prepArea, 2, out=prepArea)

# ------- if bitsAlpha > 1 ------------ > bitsAlpha 1 -----
		less_equal(bitsAlphaStack, 1, out=bitsAlphaStack)
		selectorUnderLimit: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(bitsAlphaStack), state, arrayAnalyzed, indexArcCode)
		del bitsAlphaStack 															# X indexArcCode O indexCrossings
		prepArea[selectorUnderLimit] = 0
		del selectorUnderLimit 														# O indexArcCode O indexCrossings

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze bitsZulu ---------- (bitsZulu >> 1) | (bitsAlpha << 2) | (1 - (bitsZulu & 1)) -------------
		arrayBitsZulu: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		arrayBitsZulu = bitwise_right_shift(arrayBitsZulu, 1)						# O indexArcCode X indexCrossings
		arrayBitsZulu = bitwise_and(arrayBitsZulu, state.locatorBits)
# ------- >> | << | (- 1 & bitsZulu 1) << bitsAlpha 2 1 bitsZulu 1 ----------
		bitwise_and(arrayBitsZulu, 1, out=arrayBitsZulu)
		subtract(1, arrayBitsZulu, out=arrayBitsZulu)
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.locatorBits, out=bitsAlphaStack)			# X indexArcCode X indexCrossings
		bitwise_left_shift(bitsAlphaStack, 2, out=prepArea)
		del bitsAlphaStack 															# O indexArcCode X indexCrossings
		bitwise_or(arrayBitsZulu, prepArea, out=prepArea)
		del arrayBitsZulu 															# O indexArcCode O indexCrossings
		bitwise_left_shift(prepArea, 1, out=prepArea)
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.locatorBits, out=bitsZuluStack)
		bitwise_or(bitsZuluStack, prepArea, out=prepArea)
		bitwise_right_shift(prepArea, 1, out=prepArea)

# ------- if bitsZulu > 1 ------------- > bitsZulu 1 ------
		less_equal(bitsZuluStack, 1, out=bitsZuluStack)
		selectorUnderLimit = makeStorage(numpy.flatnonzero(bitsZuluStack), state, arrayAnalyzed, indexArcCode)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		prepArea[selectorUnderLimit] = 0
		del selectorUnderLimit 														# O indexArcCode O indexCrossings

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze simple ------------------------ ((bitsAlpha | (bitsZulu << 1)) << 2) | 3 ------------------
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.locatorBits, out=bitsZuluStack)
# ------- | << | bitsAlpha << bitsZulu 1 2 3 --------------
		bitwise_left_shift(bitsZuluStack, 1, out=prepArea)
		del bitsZuluStack 															# O indexArcCode O indexCrossings
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.locatorBits, out=bitsAlphaStack)			# X indexArcCode O indexCrossings
		bitwise_or(bitsAlphaStack, prepArea, out=prepArea)
		del bitsAlphaStack 															# O indexArcCode O indexCrossings
		bitwise_left_shift(prepArea, 2, out=prepArea)
		bitwise_or(prepArea, 3, out=prepArea)

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

		del prepArea, arrayPrepArea
# ----------------------------------------------- aggregation ---------------------------------------------------------
		state.arrayArcCodes = numpy.zeros((0,), dtype=state.datatypeArcCode)
		arrayAnalyzed.resize((state.indexTarget, indicesAnalyzed))

		goByeBye()
		state = aggregateAnalyzed(arrayAnalyzed, state)

		del arrayAnalyzed

		if state.n >= 45:
		# oeisID,n,boundary,buckets,arcCodes,arcCodeBitWidth,crossingsBitWidth
			print(state.oeisID, state.n, state.boundary+1, state.indexTarget, len(state.arrayArcCodes), int(state.arrayArcCodes.max()).bit_length(), int(state.arrayCrossings.max()).bit_length(), sep=',')  # noqa: T201
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
	while state.boundary > 0:
		if areIntegersWide(state):
			state = countBigInt(state)
		else:
			state.makeArray()
			state = countNumPy(state)
			state.makeDictionary()
	return sum(state.dictionaryMeanders.values())
