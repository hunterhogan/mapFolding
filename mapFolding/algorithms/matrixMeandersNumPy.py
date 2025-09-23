from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from numpy import (
	bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, greater, logical_and, logical_or,
	multiply)
from numpy.typing import NDArray
from types import EllipsisType
from typing import NamedTuple
import numpy

class ShapeArray(NamedTuple):
	"""Always use this to construct arrays, so you can reorder the axes merely by reordering this class."""

	length: int
	indices: int

class ShapeSlicer(NamedTuple):
	"""Always use this to construct slicers, so you can reorder the axes merely by reordering this class."""

# Figure out how to have a SSOT for the axis order.
	length: EllipsisType
	indices: int

indicesCurveLocations = indicesTotal = 2
indexΩ: int = (indicesTotal - indicesTotal) - 1
indexZ0Z_CurveLocations = indexΩ = indexΩ + 1
indexZ0Z_DistinctCrossings = indexΩ = indexΩ + 1
if indexΩ != indicesTotal - 1:
	message = f"Please inspect the code above this `if` check. '{indicesTotal = }', therefore '{indexΩ = }' must be '{indicesTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del indicesTotal, indexΩ

slicerZ0Z_CurveLocations: ShapeSlicer = ShapeSlicer(length=..., indices=indexZ0Z_CurveLocations)
slicerZ0Z_DistinctCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexZ0Z_DistinctCrossings)

indicesAnalyzed = indicesTotal = 2
indexΩ: int = (indicesTotal - indicesTotal) - 1
indexCurveLocations = indexΩ = indexΩ + 1
indexDistinctCrossings = indexΩ = indexΩ + 1
if indexΩ != indicesTotal - 1:
	message = f"Please inspect the code above this `if` check. '{indicesTotal = }', therefore '{indexΩ = }' must be '{indicesTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del indicesTotal, indexΩ

slicerCurveLocations: ShapeSlicer = ShapeSlicer(length=..., indices=indexCurveLocations)
slicerDistinctCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexDistinctCrossings)

indicesPrepArea = indicesTotal = 3
indexΩ: int = (indicesTotal - indicesTotal) - 1
indexAnalysis = indexΩ = indexΩ + 1
indexAlpha = indexΩ = indexΩ + 1
indexZulu = indexΩ = indexΩ + 1
if indexΩ != indicesTotal - 1:
	message = f"Please inspect the code above this `if` check. '{indicesTotal = }', therefore '{indexΩ = }' must be '{indicesTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del indicesTotal, indexΩ

slicerAnalysis: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalysis)
slicerAlpha: ShapeSlicer = ShapeSlicer(length=..., indices=indexAlpha)
slicerZulu: ShapeSlicer = ShapeSlicer(length=..., indices=indexZulu)

def countNumPy(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Count meanders with matrix transfer algorithm using NumPy.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing.

	Returns
	-------
	state : MatrixMeandersState
		Updated state with new `kOfMatrix` and `dictionaryCurveLocations`.
	"""
	while state.kOfMatrix > 0 and not areIntegersWide(state):
		def recordAnalysis(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState, curveLocations: NDArray[numpy.uint64]) -> MatrixMeandersNumPyState:
			"""Deduplicate `curveLocations` by summing `distinctCrossings`."""
			unique = numpy.unique_inverse(curveLocations[numpy.flatnonzero(curveLocations)])

			indexStop: int = state.indexTarget + int(unique.values.size)
			arrayAnalyzed[state.indexTarget:indexStop, indexCurveLocations] = unique.values

			numpy.add.at(arrayAnalyzed[state.indexTarget:indexStop, indexDistinctCrossings], unique.inverse_indices, state.arrayCurveLocations[slicerZ0Z_DistinctCrossings][numpy.flatnonzero(curveLocations)])
			del unique
			state.indexTarget = indexStop
			del indexStop
			return state

		def aggregateAnalyzed(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState) -> None:
			"""Deduplicate `curveLocations` by summing `distinctCrossings`; create curve groups."""
			unique = numpy.unique_inverse(arrayAnalyzed[..., indexCurveLocations])

			shape = ShapeArray(length=len(unique.values), indices=indicesCurveLocations)
			state.arrayCurveLocations = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
			del shape

			state.arrayCurveLocations[slicerZ0Z_CurveLocations] = unique.values
			numpy.add.at(state.arrayCurveLocations[slicerZ0Z_DistinctCrossings], unique.inverse_indices, arrayAnalyzed[slicerDistinctCrossings])

			del unique

		shape = ShapeArray(length=len(state.arrayCurveLocations[slicerZ0Z_CurveLocations]), indices=indicesPrepArea)
		arrayPrepArea = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
		del shape

		state.kOfMatrix -= 1
		state.indexTarget = 0
		state.bitWidth = int(state.arrayCurveLocations[slicerZ0Z_CurveLocations].max()).bit_length()

		lengthArrayAnalyzed: int = max(getBucketsTotal(state, 1.2), 1961369)
		arrayAnalyzed: NDArray[numpy.uint64] = numpy.zeros((lengthArrayAnalyzed, indicesAnalyzed), dtype=state.datatypeCurveLocations)
		del lengthArrayAnalyzed

# ----------------- analyze simple ------------------------------------------------------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- ((bitsAlpha | (bitsZulu << 1)) << 2) | 3 --------
		bitwise_left_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalysis])
		bitwise_left_shift(arrayPrepArea[slicerAnalysis], 2, out=arrayPrepArea[slicerAnalysis])
		bitwise_or(arrayPrepArea[slicerAnalysis], 3, out=arrayPrepArea[slicerAnalysis])

# ------- remove overlimit --------------------------------
		multiply(arrayPrepArea[slicerAnalysis], 0, out=arrayPrepArea[slicerAnalysis], where=arrayPrepArea[slicerAnalysis] > state.MAXIMUMcurveLocations)

		state = recordAnalysis(arrayAnalyzed, state, arrayPrepArea[slicerAnalysis])

# ----------------- analyze bitsAlpha ---------------------------------------------------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- ((((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) << 2) | bitsAlpha) >> 2 ---
		bitwise_and(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_xor(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_left_shift(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_left_shift(arrayPrepArea[slicerZulu], 3, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalysis])
		bitwise_left_shift(arrayPrepArea[slicerAnalysis], 2, out=arrayPrepArea[slicerAnalysis])

		# ------- assign bitsAlpha ------------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])

		bitwise_or(arrayPrepArea[slicerAnalysis], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalysis])
		bitwise_right_shift(arrayPrepArea[slicerAnalysis], 2, out=arrayPrepArea[slicerAnalysis])

# ------- if bitsAlpha > 1 --------------------------------
		multiply(arrayPrepArea[slicerAnalysis], 0, out=arrayPrepArea[slicerAnalysis], where=arrayPrepArea[slicerAlpha] <= 1)
# ------- remove overlimit --------------------------------
		multiply(arrayPrepArea[slicerAnalysis], 0, out=arrayPrepArea[slicerAnalysis], where=arrayPrepArea[slicerAnalysis] > state.MAXIMUMcurveLocations)

		state = recordAnalysis(arrayAnalyzed, state, arrayPrepArea[slicerAnalysis])

# ----------------- analyze bitsZulu ----------------------------------------------------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- (((1 - (bitsZulu & 1)) | (bitsAlpha << 2) << 1) | bitsZulu) >> 1 ----
		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_xor(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_left_shift(arrayPrepArea[slicerAlpha], 2, out=arrayPrepArea[slicerAlpha])
		bitwise_or(arrayPrepArea[slicerZulu], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalysis])
		bitwise_left_shift(arrayPrepArea[slicerAnalysis], 1, out=arrayPrepArea[slicerAnalysis])

		# ------- assign bitsZulu -------------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

		bitwise_or(arrayPrepArea[slicerAnalysis], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalysis])
		bitwise_right_shift(arrayPrepArea[slicerAnalysis], 1, out=arrayPrepArea[slicerAnalysis])

# ------- if bitsZulu > 1 ---------------------------------
		multiply(arrayPrepArea[slicerAnalysis], 0, out=arrayPrepArea[slicerAnalysis], where=arrayPrepArea[slicerZulu] <= 1)
# ------- remove overlimit --------------------------------
		multiply(arrayPrepArea[slicerAnalysis], 0, out=arrayPrepArea[slicerAnalysis], where=arrayPrepArea[slicerAnalysis] > state.MAXIMUMcurveLocations)

		state = recordAnalysis(arrayAnalyzed, state, arrayPrepArea[slicerAnalysis])

# ----------------- analyze aligned -----------------------------------------------------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven) -----
# ------- if bitsAlpha > 1 and bitsZulu > 1 ---------------
		greater(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		greater(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		logical_and(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalysis])

# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- ... and (bitsAlphaIsEven or bitsZuluIsEven) -----
		bitwise_and(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_xor(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_xor(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		logical_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAlpha], casting='unsafe') # NOTE atypical pattern for `out=`
		logical_and(arrayPrepArea[slicerAnalysis], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalysis])

# ------- arrayCurveLocations.resize(qualified values) ----
		selectorKeepThese = arrayPrepArea[slicerAnalysis].astype(dtype=numpy.bool_)
		state.arrayCurveLocations = state.arrayCurveLocations[selectorKeepThese]
		del selectorKeepThese

# ------- recreate arrayPrepArea --------------------------
		shape = ShapeArray(length=len(state.arrayCurveLocations[slicerZ0Z_CurveLocations]), indices=indicesPrepArea)
		arrayPrepArea = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
		del shape

# ------- align bitsAlpha and bitsZulu ----------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- if bitsAlphaAtEven and not bitsZuluAtEven -------
		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		selectorModifyHere = arrayPrepArea[slicerZulu].astype(dtype=numpy.bool_)
		flipTheExtra_0b1AsUfunc(arrayPrepArea[slicerAlpha], where=selectorModifyHere, out=arrayPrepArea[slicerAlpha], casting='unsafe')
		del selectorModifyHere

		# ------- assign bitsZulu -------------------------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- if bitsZuluAtEven and not bitsAlphaAtEven -------
		bitwise_and(state.arrayCurveLocations[slicerZ0Z_CurveLocations], 1, out=state.arrayCurveLocations[slicerZ0Z_CurveLocations]) # NOTE atypical pattern for `out=`
		selectorModifyHere = state.arrayCurveLocations[slicerZ0Z_CurveLocations].astype(dtype=numpy.bool_)
		flipTheExtra_0b1AsUfunc(arrayPrepArea[slicerZulu], where=selectorModifyHere, out=arrayPrepArea[slicerZulu], casting='unsafe')
		del selectorModifyHere

# ------- (bitsAlpha >> 2) | ((bitsZulu >> 2) << 1) -------
		bitwise_right_shift(arrayPrepArea[slicerAlpha], 2, out=arrayPrepArea[slicerAlpha])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 2, out=arrayPrepArea[slicerZulu])
		bitwise_left_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalysis])

# ------- remove overlimit --------------------------------
		multiply(arrayPrepArea[slicerAnalysis], 0, out=arrayPrepArea[slicerAnalysis], where=arrayPrepArea[slicerAnalysis] > state.MAXIMUMcurveLocations)

		state = recordAnalysis(arrayAnalyzed, state, arrayPrepArea[slicerAnalysis])

		del arrayPrepArea

		arrayAnalyzed.resize((state.indexTarget, indicesAnalyzed))

# ----------------------------------------------- aggregation ---------------------------------------------------------
		aggregateAnalyzed(arrayAnalyzed, state)

		del arrayAnalyzed

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
	return sum(state.dictionaryCurveLocations.values())
