from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from numpy import (
	bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, greater, logical_and, logical_not,
	multiply, subtract)
from numpy.typing import NDArray
from types import EllipsisType
from typing import Final, NamedTuple, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from numpy.lib._arraysetops_impl import UniqueInverseResult

"""Goals:
- Extreme abstraction.
- Find operations with latent intermediate arrays, and make the intermediate array explicit or eliminate it.
- Reduce or eliminate selector arrays.
- Write formulas in prefix notation.
- Find prefix notation formulas that never use the same variable as input more than once: that would allow the evaluation of the
	expression with only a single stack, which saves memory.
- Standardize code as much as possible to create duplicate code.
- Convert duplicate code to procedures.
"""

# TODO Figure out how to have a SSOT for the axis order.
axisOfLength: Final[int] = 0

class ShapeArray(NamedTuple):
	"""Always use this to construct arrays, so you can reorder the axes merely by reordering this class."""

	length: int
	indices: int

class ShapeSlicer(NamedTuple):
	"""Always use this to construct slicers, so you can reorder the axes merely by reordering this class."""

	length: EllipsisType | slice
	indices: int

indicesMeanders: int = 2
indexMeandersArcCode, indexMeandersDistinctCrossings = range(indicesMeanders)
slicerMeandersArcCode: ShapeSlicer = ShapeSlicer(length=..., indices=indexMeandersArcCode)
slicerMeandersDistinctCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexMeandersDistinctCrossings)

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
	"""Count meanders with matrix transfer algorithm using NumPy.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing.

	Returns
	-------
	state : MatrixMeandersState
		Updated state with new `kOfMatrix` and `arrayMeanders`.
	"""
	while state.kOfMatrix > 0 and not areIntegersWide(state):
		def aggregateAnalyzed(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
			"""Deduplicate `arcCode` by summing `distinctCrossings`; create curve groups."""
			unique: UniqueInverseResult[numpy.uint64] = numpy.unique_inverse(arrayAnalyzed[slicerAnalyzedArcCode])

			shape = ShapeArray(length=len(unique.values), indices=indicesMeanders)
			state.arrayMeanders = numpy.zeros(shape, dtype=state.datatypeArcCode)
			del shape

			state.arrayMeanders[slicerMeandersArcCode] = unique.values
			numpy.add.at(state.arrayMeanders[slicerMeandersDistinctCrossings], unique.inverse_indices, arrayAnalyzed[slicerAnalyzedDistinctCrossings])

			del unique

			return state

		def recordAnalysis(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState, arcCode: NDArray[numpy.uint64]) -> MatrixMeandersNumPyState:
			"""Deduplicate `arcCode` by summing `distinctCrossings`."""
			selectorOverLimit = arcCode > state.MAXIMUMarcCode
			multiply(arcCode, 0, out=arcCode, where=selectorOverLimit)
			del selectorOverLimit

			selectorNonzero: NDArray[numpy.bool_] = arcCode.astype(dtype=numpy.bool_)

			unique: UniqueInverseResult[numpy.uint64] = numpy.unique_inverse(arcCode[selectorNonzero])

			indexStop: int = state.indexTarget + len(unique.values)
			sliceUnique: slice = slice(state.indexTarget, indexStop)
			state.indexTarget = indexStop
			del indexStop

			slicerUniqueValues = ShapeSlicer(length=sliceUnique, indices=indexAnalyzedArcCode)
			slicerAddAt = ShapeSlicer(length=sliceUnique, indices=indexAnalyzedDistinctCrossings)
			del sliceUnique

			arrayAnalyzed[slicerUniqueValues] = unique.values
			del slicerUniqueValues

			numpy.add.at(arrayAnalyzed[slicerAddAt], unique.inverse_indices, state.arrayMeanders[slicerMeandersDistinctCrossings][selectorNonzero])
			del slicerAddAt, unique, selectorNonzero

			return state

		def makePrepArea(state: MatrixMeandersNumPyState) -> NDArray[numpy.uint64]:
			"""Create `arrayPrepArea`."""
			shape = ShapeArray(length=len(state.arrayMeanders[slicerMeandersArcCode]), indices=indicesPrepArea)
			arrayPrepArea: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
			del shape

			bitwise_and(state.arrayMeanders[slicerMeandersArcCode], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
			bitwise_and(state.arrayMeanders[slicerMeandersArcCode], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
			bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

			return arrayPrepArea

		def makeViews(arrayPrepArea: NDArray[numpy.uint64]) -> tuple[NDArray[numpy.uint64], NDArray[numpy.uint64], NDArray[numpy.uint64]]:
			"""Create views of `arrayPrepArea`."""
			viewStackAnalysis: NDArray[numpy.uint64] = arrayPrepArea[slicerAnalysis].view()
			viewAlpha: NDArray[numpy.uint64] = arrayPrepArea[slicerAlpha].view()
			viewZulu: NDArray[numpy.uint64] = arrayPrepArea[slicerZulu].view()
			return viewStackAnalysis, viewAlpha, viewZulu # match values of `indicesPrepArea`

		state.bitWidth = int(state.arrayMeanders[slicerMeandersArcCode].max()).bit_length()

		lengthArrayAnalyzed: int = getBucketsTotal(state, 1.2)
		shape = ShapeArray(length=lengthArrayAnalyzed, indices=indicesAnalyzed)
		del lengthArrayAnalyzed
		arrayAnalyzed: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
		del shape

		arrayPrepArea: NDArray[numpy.uint64] = makePrepArea(state)

		# Just one explicit, long-form, annotated unpacking for demonstration and for the type checker.
		views: tuple[NDArray[numpy.uint64], NDArray[numpy.uint64], NDArray[numpy.uint64]] = makeViews(arrayPrepArea)
		viewStackAnalysis: NDArray[numpy.uint64] = views[indexAnalysis]
		viewAlpha: NDArray[numpy.uint64] = views[indexAlpha]
		viewZulu: NDArray[numpy.uint64] = views[indexZulu]

		state.indexTarget = 0

		state.kOfMatrix -= 1

# ----------------- analyze simple ------------------------------------------------------------------------------------
# ------- ((bitsAlpha | (bitsZulu << 1)) << 2) | 3 --------
# ------- | << | bitsAlpha << bitsZulu 1 2 3 --------------
		bitwise_left_shift(viewZulu, 1, out=viewStackAnalysis)
		bitwise_or(viewAlpha, viewStackAnalysis, out=viewStackAnalysis)
		bitwise_left_shift(viewStackAnalysis, 2, out=viewStackAnalysis)
		bitwise_or(viewStackAnalysis, 3, out=viewStackAnalysis)

		state = recordAnalysis(arrayAnalyzed, state, viewStackAnalysis)

# ----------------- analyze bitsAlpha ---------------------------------------------------------------------------------
		stackInMemory: NDArray[numpy.uint64] = numpy.zeros_like(arrayPrepArea[slicerAnalysis])

# ------- (((((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3)) << 2) | bitsAlpha) >> 2 ---
# ------- >> | << | (<< - 1 & bitsAlpha 1 1) << bitsZulu 3 2 bitsAlpha 2 --------------
		bitwise_and(viewAlpha, 1, out=stackInMemory)
		subtract(1, stackInMemory, out=stackInMemory)
		bitwise_left_shift(stackInMemory, 1, out=stackInMemory)
		bitwise_left_shift(viewZulu, 3, out=viewStackAnalysis)
		bitwise_or(stackInMemory, viewStackAnalysis, out=viewStackAnalysis)
		del stackInMemory
		bitwise_left_shift(viewStackAnalysis, 2, out=viewStackAnalysis)
		bitwise_or(viewAlpha, viewStackAnalysis, out=viewStackAnalysis)
		bitwise_right_shift(viewStackAnalysis, 2, out=viewStackAnalysis)

# ------- if bitsAlpha > 1 --------------------------------
		selectorUnderLimit: NDArray[numpy.bool_] = numpy.less_equal(viewAlpha, 1, dtype=numpy.bool_)
		multiply(viewStackAnalysis, 0, out=viewStackAnalysis, where=selectorUnderLimit)
		del selectorUnderLimit

		state = recordAnalysis(arrayAnalyzed, state, viewStackAnalysis)

# ----------------- analyze bitsZulu ----------------------------------------------------------------------------------
		stackInMemory: NDArray[numpy.uint64] = numpy.zeros_like(arrayPrepArea[slicerAnalysis])

# ------- ((((1 - (bitsZulu & 1)) | (bitsAlpha << 2)) << 1) | bitsZulu) >> 1 ----
# ------- >> | << | (- 1 & bitsZulu 1) << bitsAlpha 2 1 bitsZulu 1 --------------
		bitwise_and(viewZulu, 1, out=stackInMemory)
		subtract(1, stackInMemory, out=stackInMemory)
		bitwise_left_shift(viewAlpha, 2, out=viewStackAnalysis)
		bitwise_or(stackInMemory, viewStackAnalysis, out=viewStackAnalysis)
		del stackInMemory
		bitwise_left_shift(viewStackAnalysis, 1, out=viewStackAnalysis)
		bitwise_or(viewZulu, viewStackAnalysis, out=viewStackAnalysis)
		bitwise_right_shift(viewStackAnalysis, 1, out=viewStackAnalysis)

# ------- if bitsZulu > 1 ---------------------------------
		selectorUnderLimit: NDArray[numpy.bool_] = numpy.less_equal(viewZulu, 1, dtype=numpy.bool_)
		multiply(viewStackAnalysis, 0, out=viewStackAnalysis, where=selectorUnderLimit)
		del selectorUnderLimit

		state = recordAnalysis(arrayAnalyzed, state, viewStackAnalysis)

# ================= analyze aligned ===================================================================================
# ======= if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven) =====
		greater(viewAlpha, 1, out=viewStackAnalysis)

# ======= arrayMeanders resize(remove disqualified values) ====
		selectorKeepThese: NDArray[numpy.intp] = numpy.flatnonzero(viewStackAnalysis)
		state.arrayMeanders = numpy.take(state.arrayMeanders, selectorKeepThese, axis=axisOfLength)
		del selectorKeepThese

		del viewStackAnalysis, viewAlpha, viewZulu
		arrayPrepArea = makePrepArea(state)
		viewStackAnalysis, viewAlpha, viewZulu = makeViews(arrayPrepArea)

		greater(viewZulu, 1, out=viewStackAnalysis)

# ======= arrayMeanders resize(remove disqualified values) ====
		selectorKeepThese = numpy.flatnonzero(viewStackAnalysis)
		state.arrayMeanders = numpy.take(state.arrayMeanders, selectorKeepThese, axis=axisOfLength)
		del selectorKeepThese

		del viewStackAnalysis, viewAlpha, viewZulu
		arrayPrepArea = makePrepArea(state)
		viewStackAnalysis, viewAlpha, viewZulu = makeViews(arrayPrepArea)

# ======= if ... and (bitsAlphaIsEven or bitsZuluIsEven) ============
# ======= ^ & & bitsAlpha 1 bitsZulu 1 ============
		bitwise_and(viewAlpha, 1, out=viewStackAnalysis)
		bitwise_and(viewZulu, viewStackAnalysis, out=viewStackAnalysis)
		bitwise_xor(viewStackAnalysis, 1, out=viewStackAnalysis)

# ======= arrayMeanders resize(qualified values) ====
		selectorKeepThese = numpy.flatnonzero(viewStackAnalysis)
		state.arrayMeanders = numpy.take(state.arrayMeanders, selectorKeepThese, axis=axisOfLength)
		del selectorKeepThese

		del viewStackAnalysis, viewAlpha, viewZulu
		arrayPrepArea = makePrepArea(state)
		viewStackAnalysis, viewAlpha, viewZulu = makeViews(arrayPrepArea)

# ======= align bitsAlpha and bitsZulu ========================================
# ======= if bitsAlphaAtEven and not bitsZuluAtEven =======
# ======= (1 - (bitsAlpha & 1)) & (bitsZulu & 1) =======
		bitwise_and(viewAlpha, 1, out=viewStackAnalysis)
		selectorAlphaAtOdd: NDArray[numpy.bool_] = viewStackAnalysis.astype(dtype=numpy.bool_)
		bitwise_and(viewZulu, 1, out=viewStackAnalysis)
		selectorZuluAtOdd: NDArray[numpy.bool_] = viewStackAnalysis.astype(dtype=numpy.bool_)

		flipTheExtra_0b1AsUfunc(viewAlpha, out=arrayPrepArea[slicerAlpha]
			, where=logical_and(logical_not(selectorAlphaAtOdd), selectorZuluAtOdd)
			, casting='unsafe')

# ======= if bitsZuluAtEven and not bitsAlphaAtEven =======
# ======= (1 - (bitsZulu & 1)) & (bitsAlpha & 1) =======
		flipTheExtra_0b1AsUfunc(viewZulu, out=arrayPrepArea[slicerZulu]
			, where=logical_and(selectorAlphaAtOdd, logical_not(selectorZuluAtOdd))
			, casting='unsafe')
		del selectorAlphaAtOdd
		del selectorZuluAtOdd
# ======= (((bitsZulu >> 2) << 3) | bitsAlpha) >> 2 =======
# ======= >> | << >> bitsZulu 2 3 bitsAlpha 2 =============
		bitwise_right_shift(viewZulu, 2, out=viewStackAnalysis)
		bitwise_left_shift(viewStackAnalysis, 3, out=viewStackAnalysis)
		bitwise_or(viewStackAnalysis, viewAlpha, out=viewStackAnalysis)
		bitwise_right_shift(viewStackAnalysis, 2, out=viewStackAnalysis)

		state = recordAnalysis(arrayAnalyzed, state, viewStackAnalysis)

		del viewStackAnalysis, viewAlpha, viewZulu
		del arrayPrepArea

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
