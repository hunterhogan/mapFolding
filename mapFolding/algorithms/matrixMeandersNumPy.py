from copy import copy
from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from numpy import (
	bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, greater, logical_and, logical_not,
	logical_or, multiply)
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

slicerCurveLocations: ShapeSlicer = ShapeSlicer(length=..., indices=indexCurveLocations)
slicerDistinctCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexDistinctCrossings)
slicerAnalyzed: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalyzed)
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
		del length
		arrayAnalyzed = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
		del shape
		shape = ShapeArray(length=len(state.arrayCurveLocations[slicerCurveLocations]), indices=indicesArrayPrepArea)
		arrayPrepArea = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
		del shape

		state.bitWidth = int(state.arrayCurveLocations[slicerCurveLocations].max()).bit_length()
		state.indexTarget = 0
		state.kOfMatrix -= 1

# ----------------- analyze simple ------------------------------------------------------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- ((bitsAlpha | (bitsZulu << 1)) << 2) | 3 --------
		bitwise_left_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])
		bitwise_left_shift(arrayPrepArea[slicerAnalyzed], 2, out=arrayPrepArea[slicerAnalyzed])
		bitwise_or(arrayPrepArea[slicerAnalyzed], 3, out=arrayPrepArea[slicerAnalyzed])

# ------- remove overlimit --------------------------------
		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAnalyzed] > state.MAXIMUMcurveLocations)

# ------- record analysis ---------------------------------
		viewKeepThese = arrayPrepArea[slicerAnalyzed].astype(dtype=numpy.bool_)
		indexStop = numpy.count_nonzero(viewKeepThese)
		sliceKeepThese: slice = slice(copy(state.indexTarget), state.indexTarget + indexStop)
		state.indexTarget += indexStop
		del indexStop

		viewArrayAnalyzedSlicerAnalyzed = arrayAnalyzed[slicerAnalyzed].view()
		viewArrayAnalyzedSlicerDistinctCrossings = arrayAnalyzed[slicerDistinctCrossings].view()
		viewArrayAnalyzedSlicerAnalyzed[sliceKeepThese] = arrayPrepArea[slicerAnalyzed][viewKeepThese]
		del viewArrayAnalyzedSlicerAnalyzed
		viewArrayAnalyzedSlicerDistinctCrossings[sliceKeepThese] = state.arrayCurveLocations[slicerDistinctCrossings][viewKeepThese]
		del sliceKeepThese, viewArrayAnalyzedSlicerDistinctCrossings, viewKeepThese

# ----------------- analyze bitsAlpha ---------------------------------------------------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- ((((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) << 2) | bitsAlpha) >> 2 ---
		bitwise_and(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_xor(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_left_shift(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_left_shift(arrayPrepArea[slicerZulu], 3, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])
		bitwise_left_shift(arrayPrepArea[slicerAnalyzed], 2, out=arrayPrepArea[slicerAnalyzed])

		# ------- assign bitsAlpha ------------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])

		bitwise_or(arrayPrepArea[slicerAnalyzed], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalyzed])
		bitwise_right_shift(arrayPrepArea[slicerAnalyzed], 2, out=arrayPrepArea[slicerAnalyzed])

# ------- if bitsAlpha > 1 --------------------------------
		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAlpha] <= 1)
# ------- remove overlimit --------------------------------
		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAnalyzed] > state.MAXIMUMcurveLocations)

# ------- record analysis ---------------------------------
		viewKeepThese = arrayPrepArea[slicerAnalyzed].astype(dtype=numpy.bool_)
		indexStop = int(viewKeepThese.sum())
		sliceKeepThese = slice(copy(state.indexTarget), copy(state.indexTarget) + indexStop)
		state.indexTarget += indexStop
		del indexStop

		viewArrayAnalyzedSlicerAnalyzed = arrayAnalyzed[slicerAnalyzed].view()
		viewArrayAnalyzedSlicerDistinctCrossings = arrayAnalyzed[slicerDistinctCrossings].view()

		viewArrayAnalyzedSlicerAnalyzed[sliceKeepThese] = arrayPrepArea[slicerAnalyzed][viewKeepThese]
		del viewArrayAnalyzedSlicerAnalyzed
		viewArrayAnalyzedSlicerDistinctCrossings[sliceKeepThese] = state.arrayCurveLocations[slicerDistinctCrossings][viewKeepThese]
		del sliceKeepThese, viewArrayAnalyzedSlicerDistinctCrossings, viewKeepThese

# ----------------- analyze bitsZulu ----------------------------------------------------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- (((1 - (bitsZulu & 1)) | (bitsAlpha << 2) << 1) | bitsZulu) >> 1 ----
		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_xor(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_left_shift(arrayPrepArea[slicerAlpha], 2, out=arrayPrepArea[slicerAlpha])
		bitwise_or(arrayPrepArea[slicerZulu], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalyzed])
		bitwise_left_shift(arrayPrepArea[slicerAnalyzed], 1, out=arrayPrepArea[slicerAnalyzed])

		# ------- assign bitsZulu -------------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

		bitwise_or(arrayPrepArea[slicerAnalyzed], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])
		bitwise_right_shift(arrayPrepArea[slicerAnalyzed], 1, out=arrayPrepArea[slicerAnalyzed])

# ------- if bitsZulu > 1 ---------------------------------
		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerZulu] <= 1)
# ------- remove overlimit --------------------------------
		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAnalyzed] > state.MAXIMUMcurveLocations)

# ------- record analysis ---------------------------------
		viewKeepThese = arrayPrepArea[slicerAnalyzed].astype(dtype=numpy.bool_)
		indexStop = int(viewKeepThese.sum())
		sliceKeepThese = slice(copy(state.indexTarget), copy(state.indexTarget) + indexStop)
		state.indexTarget += indexStop
		del indexStop

		viewArrayAnalyzedSlicerAnalyzed = arrayAnalyzed[slicerAnalyzed].view()
		viewArrayAnalyzedSlicerDistinctCrossings = arrayAnalyzed[slicerDistinctCrossings].view()

		viewArrayAnalyzedSlicerAnalyzed[sliceKeepThese] = arrayPrepArea[slicerAnalyzed][viewKeepThese]
		del viewArrayAnalyzedSlicerAnalyzed
		viewArrayAnalyzedSlicerDistinctCrossings[sliceKeepThese] = state.arrayCurveLocations[slicerDistinctCrossings][viewKeepThese]
		del sliceKeepThese, viewArrayAnalyzedSlicerDistinctCrossings, viewKeepThese

# ----------------- analyze aligned -----------------------------------------------------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven) -----
# ------- if bitsAlpha > 1 and bitsZulu > 1 ---------------
		greater(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		greater(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		logical_and(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])

# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- ... and (bitsAlphaIsEven or bitsZuluIsEven) -----
		bitwise_and(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_xor(arrayPrepArea[slicerAlpha], 1, out=arrayPrepArea[slicerAlpha])
		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_xor(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		logical_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAlpha], casting='unsafe') # NOTE atypical pattern for `out=`
		logical_and(arrayPrepArea[slicerAnalyzed], arrayPrepArea[slicerAlpha], out=arrayPrepArea[slicerAnalyzed])

# ------- arrayCurveLocations.resize(qualified values) ----
		viewKeepThese = arrayPrepArea[slicerAnalyzed].astype(dtype=numpy.bool_)
		state.arrayCurveLocations = state.arrayCurveLocations[viewKeepThese]
		del viewKeepThese

# ------- recreate arrayPrepArea --------------------------
		shape = ShapeArray(length=len(state.arrayCurveLocations[slicerCurveLocations]), indices=indicesArrayPrepArea)
		arrayPrepArea = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
		del shape

# ------- align bitsAlpha and bitsZulu ----------------------------------------
# ------- assign bitsAlpha and bitsZulu -------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha, out=arrayPrepArea[slicerAlpha])
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- if bitsAlphaAtEven and not bitsZuluAtEven -------
		bitwise_and(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		viewModifyHere = arrayPrepArea[slicerZulu].astype(dtype=numpy.bool_)
		flipTheExtra_0b1AsUfunc(arrayPrepArea[slicerAlpha], where=viewModifyHere, out=arrayPrepArea[slicerAlpha], casting='unsafe')
		del viewModifyHere

		# ------- assign bitsZulu -------------------------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu, out=arrayPrepArea[slicerZulu])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])

# ------- if bitsZuluAtEven and not bitsAlphaAtEven -------
		bitwise_and(state.arrayCurveLocations[slicerCurveLocations], 1, out=state.arrayCurveLocations[slicerCurveLocations]) # NOTE atypical pattern for `out=`
		viewModifyHere = state.arrayCurveLocations[slicerCurveLocations].astype(dtype=numpy.bool_)
		flipTheExtra_0b1AsUfunc(arrayPrepArea[slicerZulu], where=viewModifyHere, out=arrayPrepArea[slicerZulu], casting='unsafe')
		del viewModifyHere

# ------- (bitsAlpha >> 2) | ((bitsZulu >> 2) << 1) -------
		bitwise_right_shift(arrayPrepArea[slicerAlpha], 2, out=arrayPrepArea[slicerAlpha])
		bitwise_right_shift(arrayPrepArea[slicerZulu], 2, out=arrayPrepArea[slicerZulu])
		bitwise_left_shift(arrayPrepArea[slicerZulu], 1, out=arrayPrepArea[slicerZulu])
		bitwise_or(arrayPrepArea[slicerAlpha], arrayPrepArea[slicerZulu], out=arrayPrepArea[slicerAnalyzed])

# ------- remove overlimit --------------------------------
		multiply(arrayPrepArea[slicerAnalyzed], 0, out=arrayPrepArea[slicerAnalyzed], where=arrayPrepArea[slicerAnalyzed] > state.MAXIMUMcurveLocations)

# ------- record analysis ---------------------------------
		viewKeepThese = arrayPrepArea[slicerAnalyzed].astype(dtype=numpy.bool_)
		indexStop = int(viewKeepThese.sum())
		sliceKeepThese = slice(copy(state.indexTarget), copy(state.indexTarget) + indexStop)
		state.indexTarget += indexStop
		del indexStop

		viewArrayAnalyzedSlicerAnalyzed = arrayAnalyzed[slicerAnalyzed].view()
		viewArrayAnalyzedSlicerDistinctCrossings = arrayAnalyzed[slicerDistinctCrossings].view()

		viewArrayAnalyzedSlicerAnalyzed[sliceKeepThese] = arrayPrepArea[slicerAnalyzed][viewKeepThese]
		del arrayPrepArea, viewArrayAnalyzedSlicerAnalyzed
		viewArrayAnalyzedSlicerDistinctCrossings[sliceKeepThese] = state.arrayCurveLocations[slicerDistinctCrossings][viewKeepThese]
		del sliceKeepThese, viewArrayAnalyzedSlicerDistinctCrossings, viewKeepThese
# Should I use .resize or .empty?
		state.arrayCurveLocations = numpy.empty((0,), dtype=state.datatypeCurveLocations)

# ------- aggregate all analyses --------------------------------------------------------------------------------------
		unique = numpy.unique_inverse(arrayAnalyzed[slicerAnalyzed])

		shape = ShapeArray(length=len(unique.values), indices=indicesArrayCurveLocations)
		state.arrayCurveLocations = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
		del shape
		state.arrayCurveLocations[slicerCurveLocations] = unique.values
		numpy.add.at(state.arrayCurveLocations[slicerDistinctCrossings], unique.inverse_indices, arrayAnalyzed[slicerDistinctCrossings])
		del arrayAnalyzed, unique
# I have always had a very strong feeling the above process could be more memory efficient. If I recall correctly, I estimate
# A000682(46) will peak at 1.6 billion 64-bit elements in `unique.values`. I don't want to COPY that: I want to put the values
# into their "forever home."

	return state


def Z0Z_makeArrayCurveLocations(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Convert `state` to use NumPy arrays.

	`state.dictionaryCurveLocations` is converted to `state.arrayCurveLocations`.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	state : MatrixMeandersState
		The updated algorithm state.

	Notes
	-----
	Contradictory thoughts and feelings:
	1. This should probably be a method on `MatrixMeandersState`.
	2. I didn't make it a method because I'm overly concerned about memory allocation and deallocation.
	3. I didn't try very hard to control memory use because I believed `dictionaryCurveLocations` would "only" be a few hundred thousand items.
	4. Quick test: A000682(46) at k = 27, the normal transition point from bigInt to NumPy, has 20095980 items in `dictionaryCurveLocations`.
	5. 20 million * 2 int * ~60 bits/int = 2.4 GB for just the dictionary.

	Conclusion: I need to put more effort into memory management of this conversion process.
	"""
	shape = ShapeArray(length=len(state.dictionaryCurveLocations), indices=indicesArrayCurveLocations)
	arrayWorkbench = numpy.zeros(shape, dtype=state.datatypeCurveLocations)
	arrayWorkbench[slicerCurveLocations] = list(state.dictionaryCurveLocations.keys())
	arrayWorkbench[slicerDistinctCrossings] = list(state.dictionaryCurveLocations.values())
	state.arrayCurveLocations = arrayWorkbench.copy()
	state.dictionaryCurveLocations = {}
	del arrayWorkbench
	return state


def Z0Z_makeDictionaryCurveLocations(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Convert `state` to use a dictionary.

	`state.arrayCurveLocations` is converted to `state.dictionaryCurveLocations`.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	state : MatrixMeandersState
		The updated algorithm state.
	"""
	state.dictionaryCurveLocations = {int(key): int(value) for key, value in zip(
		state.arrayCurveLocations[slicerCurveLocations], state.arrayCurveLocations[slicerDistinctCrossings]
		, strict=True)
	}
	state.arrayCurveLocations = numpy.empty((0,), dtype=state.datatypeCurveLocations)
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
			state = Z0Z_makeArrayCurveLocations(state)
			goByeBye()
			state = countNumPy(state)
			state = Z0Z_makeDictionaryCurveLocations(state)
			goByeBye()

	return sum(state.dictionaryCurveLocations.values())

