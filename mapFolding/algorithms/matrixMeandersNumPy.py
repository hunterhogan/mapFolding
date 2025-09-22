from copy import copy
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from numpy import (
	bitwise_and, bitwise_invert, bitwise_left_shift, bitwise_not, bitwise_or, bitwise_right_shift, bitwise_xor, greater,
	logical_and, logical_not, logical_or, multiply)
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

# TODO Figure out how to have a SSOT for the axis order.
	length: EllipsisType
	indices: int

indicesArrayAnalyzed = indicesTotal = 2
indexΩ: int = (indicesTotal - indicesTotal) - 1
indexAnalyzed = indexΩ = indexΩ + 1
indexDistinctCrossings = indexΩ = indexΩ + 1
if indexΩ != indicesTotal - 1:
	message = f"Please inspect the code above this `if` check. '{indicesTotal = }', therefore '{indexΩ = }' must be '{indicesTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del indicesTotal, indexΩ

indicesArrayCurveLocations = indicesTotal = 2
indexΩ: int = (indicesTotal - indicesTotal) - 1
indexCurveLocations = indexΩ = indexΩ + 1
indexDistinctCrossings = indexΩ = indexΩ + 1
if indexΩ != indicesTotal - 1:
	message = f"Please inspect the code above this `if` check. '{indicesTotal = }', therefore '{indexΩ = }' must be '{indicesTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del indicesTotal, indexΩ

indicesArrayPrepArea = indicesTotal = 3
indexΩ: int = (indicesTotal - indicesTotal) - 1
indexAnalyzed = indexΩ = indexΩ + 1
indexAlpha = indexΩ = indexΩ + 1
indexZulu = indexΩ = indexΩ + 1
if indexΩ != indicesTotal - 1:
	message = f"Please inspect the code above this `if` check. '{indicesTotal = }', therefore '{indexΩ = }' must be '{indicesTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del indicesTotal, indexΩ

slicerAnalyzed: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalyzed)
slicerCurveLocations: ShapeSlicer = ShapeSlicer(length=..., indices=indexCurveLocations)
slicerDistinctCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexDistinctCrossings)
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

	Notes
	-----
	Original: `curveLocations = ((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) | (bitsAlpha >> 2)`
	Substitution: `curveLocations = ((((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) << 2) | bitsAlpha) >> 2`

	Original: `curveLocations = (1 - (bitsZulu & 1)) | (bitsAlpha << 2) | (bitsZulu >> 1)`
	Substitution: `curveLocations = (((1 - (bitsZulu & 1)) | (bitsAlpha << 2) << 1) | bitsZulu) >> 1`
	"""
	while (state.kOfMatrix > 0 and not areIntegersWide(state, array=state.arrayCurveLocations)):
		length: int = getBucketsTotal(state)
		shape = ShapeArray(length=length, indices=indicesArrayAnalyzed)
		arrayAnalyzed = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
		shape = ShapeArray(length=len(state.arrayCurveLocations[slicerCurveLocations]), indices=indicesArrayPrepArea)
		arrayPrepArea = numpy.zeros(shape, dtype=state.datatypeCurveLocations)

		state.bitWidth = int(state.arrayCurveLocations[slicerCurveLocations].max()).bit_length()
		state.indexTarget = 0
		state.kOfMatrix -= 1

# NOTE ((bitsAlpha | (bitsZulu << 1)) << 2) | 3
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])

		bitwise_left_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])
		bitwise_left_shift(arrayPrepArea[slicerAnalyzed], 2, out=arrayPrepArea[slicerAnalyzed])
		bitwise_or(arrayPrepArea[slicerAnalyzed], 3, out=arrayPrepArea[slicerAnalyzed])

		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAnalyzed] > state.MAXIMUMcurveLocations)

		viewKeepThese = arrayPrepArea[slicerAnalyzed].view(dtype=numpy.bool_)
		indexStop = int(viewKeepThese.sum())
		sliceKeepThese: slice = slice(copy(state.indexTarget), copy(state.indexTarget) + indexStop)
		state.indexTarget += indexStop

		viewArrayAnalyzedSlicerAnalyzed = arrayAnalyzed[slicerAnalyzed].view()
		viewArrayAnalyzedSlicerDistinctCrossings = arrayAnalyzed[slicerDistinctCrossings].view()

		viewArrayAnalyzedSlicerAnalyzed[sliceKeepThese] = arrayPrepArea[slicerAnalyzed][viewKeepThese]
		viewArrayAnalyzedSlicerDistinctCrossings[sliceKeepThese] = state.arrayCurveLocations[slicerDistinctCrossings][viewKeepThese]

# NOTE if bitsAlpha > 1:
# NOTE 		curveLocations = ((((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) << 2) | bitsAlpha) >> 2
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])

		bitwise_and(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_invert(arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAlpha])
		bitwise_left_shift(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_left_shift(arrayPrepArea[slicerZulu], 3, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])
		bitwise_left_shift(arrayPrepArea[slicerAnalyzed], 2, out=arrayPrepArea[slicerAnalyzed])

		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])

		bitwise_or(arrayPrepArea[slicerAnalyzed], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalyzed])
		bitwise_right_shift(arrayPrepArea[slicerAnalyzed], 2, out=arrayPrepArea[slicerAnalyzed])

		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAlpha] <= 1)
		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAnalyzed] > state.MAXIMUMcurveLocations)

		viewKeepThese = arrayPrepArea[slicerAnalyzed].view(dtype=numpy.bool_)
		indexStop = int(viewKeepThese.sum())
		sliceKeepThese = slice(copy(state.indexTarget), copy(state.indexTarget) + indexStop)
		state.indexTarget += indexStop

		viewArrayAnalyzedSlicerAnalyzed = arrayAnalyzed[slicerAnalyzed].view()
		viewArrayAnalyzedSlicerDistinctCrossings = arrayAnalyzed[slicerDistinctCrossings].view()

		viewArrayAnalyzedSlicerAnalyzed[sliceKeepThese] = arrayPrepArea[slicerAnalyzed][viewKeepThese]
		viewArrayAnalyzedSlicerDistinctCrossings[sliceKeepThese] = state.arrayCurveLocations[slicerDistinctCrossings][viewKeepThese]

# NOTE if bitsZulu > 1:
# NOTE 		curveLocations = (((1 - (bitsZulu & 1)) | (bitsAlpha << 2) << 1) | bitsZulu) >> 1
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])

		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_invert(arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerZulu])
		bitwise_left_shift(arrayPrepArea[slicerAlpha], 2, out=arrayPrepArea[slicerAlpha])
		bitwise_or(arrayPrepArea[slicerZulu], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalyzed])
		bitwise_left_shift(arrayPrepArea[slicerAnalyzed], 1, out=arrayPrepArea[slicerAnalyzed])

		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])

		bitwise_or(arrayPrepArea[slicerAnalyzed], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])
		bitwise_right_shift(arrayPrepArea[slicerAnalyzed], 1, out=arrayPrepArea[slicerAnalyzed])

		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerZulu] <= 1)
		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAnalyzed] > state.MAXIMUMcurveLocations)

		viewKeepThese = arrayPrepArea[slicerAnalyzed].view(dtype=numpy.bool_)
		indexStop = int(viewKeepThese.sum())
		sliceKeepThese = slice(copy(state.indexTarget), copy(state.indexTarget) + indexStop)
		state.indexTarget += indexStop

		viewArrayAnalyzedSlicerAnalyzed = arrayAnalyzed[slicerAnalyzed].view()
		viewArrayAnalyzedSlicerDistinctCrossings = arrayAnalyzed[slicerDistinctCrossings].view()

		viewArrayAnalyzedSlicerAnalyzed[sliceKeepThese] = arrayPrepArea[slicerAnalyzed][viewKeepThese]
		viewArrayAnalyzedSlicerDistinctCrossings[sliceKeepThese] = state.arrayCurveLocations[slicerDistinctCrossings][viewKeepThese]

# NOTE if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven):
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])

		greater(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		greater(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		logical_and(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])

		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])

		bitwise_and(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_invert(arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAlpha])
		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_invert(arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerZulu])
		logical_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAlpha], casting='unsafe') # NOTE atypical pattern for `out=`
		logical_and(arrayPrepArea[slicerAnalyzed], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalyzed])

		logical_not(arrayPrepArea[slicerAnalyzed], out=arrayPrepArea[slicerAnalyzed])

		viewZeroOut = arrayPrepArea[slicerAnalyzed].view(dtype=numpy.bool_)

		multiply(state.arrayCurveLocations[viewZeroOut], 0, out=state.arrayCurveLocations[viewZeroOut])

		# sort, in-place, `state.arrayCurveLocations` properly, meaning it will work for any ordering of `ShapeArray`
		# figure out how to "address" the non-zero values
		# resize, in-place, `state.arrayCurveLocations` to keep only the non-zero values
		# recreate `arrayPrepArea` so it is the same length as `state.arrayCurveLocations`
		# modify some bitsAlpha and some bitsZulu
		# finish the analysis

# NOTE if bitsAlphaAtEven and not bitsZuluAtEven:
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])

		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

		viewModifyHere = arrayPrepArea[slicerZulu].view(dtype=numpy.bool_)




# NOTE curveLocations = (bitsAlpha >> 2) | ((bitsZulu >> 2) << 1)
		bitwise_right_shift(arrayPrepArea[slicerAlpha], 2, out=arrayPrepArea[slicerAlpha])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 2, out=arrayPrepArea[slicerZulu])
		bitwise_left_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])




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
			state = Z0Z_makeArray(state)
			goByeBye()
			state = countNumPy(state)
			# Make dict?

	return sum(state.dictionaryCurveLocations.values())

def Z0Z_makeArray(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Convert `state` to use NumPy arrays.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	state : MatrixMeandersState
		The updated algorithm state.
	"""
	shape = ShapeArray(length=len(state.dictionaryCurveLocations), indices=indicesArrayCurveLocations)
	arrayWorkbench = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
	arrayWorkbench[slicerCurveLocations] = list(state.dictionaryCurveLocations.keys())
	arrayWorkbench[slicerDistinctCrossings] = list(state.dictionaryCurveLocations.values())
	state.arrayCurveLocations = arrayWorkbench.copy()
	state.dictionaryCurveLocations = {}
	del arrayWorkbench
	return state
