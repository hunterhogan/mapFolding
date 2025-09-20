from functools import cache
from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersBeDry import (
	areIntegersWide, getBucketsTotal, indexAnalyzed, indexCurveLocations, indexDistinctCrossings)
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt, walkDyckPath
from typing import Literal, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from types import EllipsisType

@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

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
	arrayCurveLocations= numpy.stack((
			numpy.zeros(len(state.dictionaryCurveLocations), dtype=state.datatypeCurveLocations)
			, numpy.atleast_1d(numpy.fromiter(state.dictionaryCurveLocations.values(), dtype=state.datatypeCurveLocations, count=len(state.dictionaryCurveLocations)))
			, numpy.atleast_1d(numpy.fromiter(state.dictionaryCurveLocations.keys(), dtype=state.datatypeCurveLocations, count=len(state.dictionaryCurveLocations)))
			)
		, axis=-1, dtype=state.datatypeCurveLocations)

	slicerAnalyzed: tuple[EllipsisType, int] = (..., indexAnalyzed)
	slicerCurveLocations: tuple[EllipsisType, int] = (..., indexCurveLocations)

	state.dictionaryCurveLocations.clear()

	while (state.kOfMatrix > 0 and not areIntegersWide(state, array=arrayCurveLocations)):
# NOTE I've read B023 dozens of times and I still don't understand.
# ruff: noqa: B023
		def aggregateCurveLocations() -> None:
			nonlocal arrayAnalyzed, arrayCurveLocations
			unique = numpy.unique_inverse(arrayAnalyzed[0:state.indexTarget, indexAnalyzed])

			arrayCurveLocations= numpy.stack((
					numpy.zeros(len(unique.values), dtype=state.datatypeCurveLocations)
					, numpy.zeros(len(unique.values), dtype=state.datatypeCurveLocations)
					, unique.values
					)
				, axis=-1, dtype=state.datatypeCurveLocations)

			numpy.add.at(arrayCurveLocations[..., indexDistinctCrossings], unique.inverse_indices, arrayAnalyzed[0:state.indexTarget, indexDistinctCrossings])

			del unique

		def analyzeCurveLocationsAligned() -> None:
			"""Compute `curveLocations` from `bitsAlpha` and `bitsZulu` if both are greater than 1 and at least one is an even number.

			Before computing `curveLocations`, some values of `bitsAlpha` and `bitsZulu` are modified.

			Formula
			-------
			```python
			if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven):
				curveLocations = (bitsAlpha >> 2) | ((bitsZulu >> 2) << 1)
			```
			"""
			# NOTE Before using `slicerPrepArea`, make sure there is enough room.
# NOTE At some point, you can overwrite `arrayCurveLocations[slicerCurveLocations`.
			nonlocal arrayAnalyzed, arrayCurveLocations

			# NOTE This is the easy version of the function so I can make sure the other functions work.

			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu
		, out=arrayCurveLocations[slicerAnalyzed]) # bitsZulu step 1
			numpy.right_shift(arrayCurveLocations[slicerAnalyzed], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # bitsZulu step 2

			# if bitsAlpha > 1 and bitsZulu > 1 and not bitsAlphaIsEven and bitsZuluIsEven
			selectorAligned = numpy.logical_and(numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha) > 1, arrayCurveLocations[slicerAnalyzed] > 1) # if bitsAlpha > 1 and bitsZulu > 1
			selectorBitsAlphaAtEven = numpy.bitwise_xor(numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], 1), 1)
			selectorBitsZuluAtEven = numpy.bitwise_xor(numpy.bitwise_and(arrayCurveLocations[slicerAnalyzed], 1), 1)
			selectorAligned = numpy.logical_and(selectorAligned, numpy.logical_or(selectorBitsAlphaAtEven, selectorBitsZuluAtEven)) # if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven)

			flipTheExtra_0b1AsUfunc(arrayCurveLocations[slicerAnalyzed], where=numpy.logical_and(selectorAligned, numpy.logical_not(selectorBitsAlphaAtEven)), casting='unsafe', dtype=state.datatypeCurveLocations
		, out=arrayCurveLocations[slicerAnalyzed])

			arrayCurveLocations[slicerAnalyzed] >>= 2 # (bitsZulu >> 2)
			arrayCurveLocations[slicerAnalyzed] <<= 1 # (bitsZulu ...) << 1

			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha
		, out=arrayCurveLocations[slicerCurveLocations]) # bitsAlpha

			flipTheExtra_0b1AsUfunc(arrayCurveLocations[slicerCurveLocations], where=numpy.logical_and(selectorAligned, numpy.logical_not(selectorBitsZuluAtEven)), casting='unsafe', dtype=state.datatypeCurveLocations
		, out=arrayCurveLocations[slicerCurveLocations])

			arrayCurveLocations[slicerCurveLocations] >>= 2 # (bitsAlpha >> 2)

			arrayCurveLocations[slicerAnalyzed] |= arrayCurveLocations[slicerCurveLocations] # (bitsAlpha ...) | ...

			arrayCurveLocations[~selectorAligned, indexAnalyzed] = state.MAXIMUMcurveLocations + 1 # if not (bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven))

			del selectorAligned, selectorBitsAlphaAtEven, selectorBitsZuluAtEven

		def analyzeCurveLocationsAlpha() -> None:
			"""Compute `curveLocations` from `bitsAlpha`.

			Formula
			-------
			```python
			if bitsAlpha > 1:
				curveLocations = ((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) | (bitsAlpha >> 2)
			# `(1 - (bitsAlpha & 1)` is an evenness test, so `(curveLocations & 1) ^ 1` is identical.
			```
			"""
			nonlocal arrayAnalyzed, arrayCurveLocations
			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # bitsAlpha
			numpy.bitwise_xor(arrayCurveLocations[slicerAnalyzed], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # (1 - (bitsAlpha ...))
			numpy.left_shift(arrayCurveLocations[slicerAnalyzed], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # ((bitsAlpha ...) << 1)

			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu
		, out=arrayAnalyzed[slicerPrepArea]) # bitsZulu step 1
			numpy.right_shift(arrayAnalyzed[slicerPrepArea], 1
		, out=arrayAnalyzed[slicerPrepArea]) # bitsZulu step 2

			numpy.left_shift(arrayAnalyzed[slicerPrepArea], 3
		, out=arrayAnalyzed[slicerPrepArea]) # (bitsZulu << 3)

			numpy.bitwise_or(arrayCurveLocations[slicerAnalyzed], arrayAnalyzed[slicerPrepArea]
		, out=arrayCurveLocations[slicerAnalyzed]) # ... | (bitsZulu ...)

			"""NOTE In this code block, I rearranged the "formula" to use `bitsTarget` for two goals. 1. `(bitsAlpha >> 2)`.
			2. `if bitsAlpha > 1`. The trick is in the equivalence of v1 and v2.
				v1: BITScow | (BITSwalk >> 2)
				v2: ((BITScow << 2) | BITSwalk) >> 2

			The "formula" calls for v1, but by using v2, `bitsTarget` is not changed. Therefore, because `bitsTarget` is
			`bitsAlpha`, I can use `bitsTarget` for goal 2, `if bitsAlpha > 1`.
			"""
			numpy.left_shift(arrayCurveLocations[slicerAnalyzed], 2
		, out=arrayCurveLocations[slicerAnalyzed]) # v2: (BITScow << 2)

			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha
		, out=arrayAnalyzed[slicerPrepArea]) # bitsAlpha

			numpy.bitwise_or(arrayCurveLocations[slicerAnalyzed], arrayAnalyzed[slicerPrepArea]
		, out=arrayCurveLocations[slicerAnalyzed]) # v2: (... | BITSwalk)
			numpy.right_shift(arrayCurveLocations[slicerAnalyzed], 2
		, out=arrayCurveLocations[slicerAnalyzed]) # v2: ... >> 2. Net effect: ... | (bitsAlpha >> 2)


# NOTE If I did this correctly, arrayCurveLocations[indexAnalyzed] will be over the limit, so `recordCurveLocations` will filter
# it out. And, I didn't create a monster masking array.
			numpy.add(arrayCurveLocations[slicerAnalyzed], state.MAXIMUMcurveLocations + 1, where=(arrayAnalyzed[slicerPrepArea] <= 1)
		, out=arrayCurveLocations[slicerAnalyzed]) # if bitsAlpha > 1

		def analyzeCurveLocationsSimple() -> None:
			"""Compute `curveLocations` with the 'simple' formula.

			Formula
			-------
			```python
			curveLocations = ((bitsAlpha | (bitsZulu << 1)) << 2) | 3
			```
			"""
			nonlocal arrayAnalyzed, arrayCurveLocations

			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha
		, out=arrayCurveLocations[slicerAnalyzed]) # bitsAlpha

			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu
		, out=arrayAnalyzed[slicerPrepArea]) # bitsZulu step 1
			numpy.right_shift(arrayAnalyzed[slicerPrepArea], 1
		, out=arrayAnalyzed[slicerPrepArea]) # bitsZulu step 2

			numpy.left_shift(arrayAnalyzed[slicerPrepArea], 1
		, out=arrayAnalyzed[slicerPrepArea]) # (bitsZulu << 1)

			numpy.bitwise_or(arrayCurveLocations[slicerAnalyzed], arrayAnalyzed[slicerPrepArea]
		, out=arrayCurveLocations[slicerAnalyzed]) # (bitsAlpha | (bitsZulu ...))
			numpy.left_shift(arrayCurveLocations[slicerAnalyzed], 2
		, out=arrayCurveLocations[slicerAnalyzed]) # (... << 2)
			numpy.bitwise_or(arrayCurveLocations[slicerAnalyzed], 3
		, out=arrayCurveLocations[slicerAnalyzed]) # ... | 3

		def analyzeCurveLocationsZulu() -> None:
			"""Compute `curveLocations` from `bitsZulu`.

			Formula
			-------
			```python
			if bitsZulu > 1:
				curveLocations = (1 - (bitsZulu & 1)) | (bitsAlpha << 2) | (bitsZulu >> 1)
			```
			"""
			nonlocal arrayAnalyzed, arrayCurveLocations
			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], 0b10 # This time, we only need 2 bits: state.locatorBitsZulu
		, out=arrayCurveLocations[slicerAnalyzed]) # bitsZulu step 1
			numpy.right_shift(arrayCurveLocations[slicerAnalyzed], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # bitsZulu step 2

			numpy.bitwise_and(arrayCurveLocations[slicerAnalyzed], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # (bitsZulu & 1)
			numpy.bitwise_xor(arrayCurveLocations[slicerAnalyzed], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # (1 - (bitsZulu ...))

			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsAlpha
		, out=arrayAnalyzed[slicerPrepArea]) # bitsAlpha

			numpy.left_shift(arrayAnalyzed[slicerPrepArea], 2
		, out=arrayAnalyzed[slicerPrepArea]) # (bitsAlpha << 2)

			numpy.bitwise_or(arrayCurveLocations[slicerAnalyzed], arrayAnalyzed[slicerPrepArea]
		, out=arrayCurveLocations[slicerAnalyzed]) # ... | (bitsAlpha ...)

			# NOTE Same trick as in `analyzeCurveLocationsAlpha`.
			numpy.left_shift(arrayCurveLocations[slicerAnalyzed], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # v2: (BITScow << 1)

			numpy.bitwise_and(arrayCurveLocations[slicerCurveLocations], state.locatorBitsZulu
		, out=arrayAnalyzed[slicerPrepArea]) # bitsZulu step 1
			numpy.right_shift(arrayAnalyzed[slicerPrepArea], 1
		, out=arrayAnalyzed[slicerPrepArea]) # bitsZulu step 2

			numpy.bitwise_or(arrayCurveLocations[slicerAnalyzed], arrayAnalyzed[slicerPrepArea]
		, out=arrayCurveLocations[slicerAnalyzed]) # v2: (... | BITSwalk)
			numpy.right_shift(arrayCurveLocations[slicerAnalyzed], 1
		, out=arrayCurveLocations[slicerAnalyzed]) # v2: ... >> 1. Net effect: ... | (bitsZulu >> 1)

			numpy.add(arrayCurveLocations[slicerAnalyzed], state.MAXIMUMcurveLocations + 1, where=(arrayAnalyzed[slicerPrepArea] <= 1)
		, out=arrayCurveLocations[slicerAnalyzed]) # if bitsZulu > 1

		def recordCurveLocations() -> None:
			nonlocal arrayAnalyzed, arrayCurveLocations

			selectorMAXIMUMcurveLocations = (arrayCurveLocations[slicerAnalyzed] <= state.MAXIMUMcurveLocations)
			indexStop = numpy.count_nonzero(selectorMAXIMUMcurveLocations)

			arrayAnalyzed[state.indexTarget:state.indexTarget + indexStop, ...] = arrayCurveLocations[selectorMAXIMUMcurveLocations, 0:2].copy()

			state.indexTarget += indexStop

		state.bitWidth = int(arrayCurveLocations[slicerCurveLocations].max()).bit_length()
		shapeArray: tuple[int, Literal[2]] = (getBucketsTotal(state), 2)
		arrayAnalyzed = numpy.zeros(shapeArray, state.datatypeCurveLocations)

		slicerPrepArea: tuple[slice, int] = (slice(0 - (len(arrayCurveLocations)), None), indexAnalyzed)
		state.indexTarget = 0

		state.kOfMatrix -= 1

		analyzeCurveLocationsSimple()
		recordCurveLocations()

		analyzeCurveLocationsAlpha()
		recordCurveLocations()

		analyzeCurveLocationsZulu()
		recordCurveLocations()

		analyzeCurveLocationsAligned()
		recordCurveLocations()

		aggregateCurveLocations()

	state.dictionaryCurveLocations = {
			int(key): int(value) for key, value in zip(
				arrayCurveLocations[slicerCurveLocations]
				, arrayCurveLocations[..., indexDistinctCrossings]
			, strict=True)}

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
			state = countNumPy(state)

		goByeBye()

	return sum(state.dictionaryCurveLocations.values())
