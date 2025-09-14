"""Count meanders with matrix transfer algorithm.

Notes
-----
- Odd/even of `groupAlpha` == the odd/even of `curveLocations`. Proof: `groupAlphaIsEven = curveLocations & 1 & 1 ^ 1`.
- Odd/even of `groupZulu` == `curveLocations` second-least significant bit. So `groupZuluIsEven = bool(curveLocations & 2 ^ 2)`.
"""
from functools import cache
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding import MatrixMeandersState
from mapFolding.algorithms.getBucketsTotal import getBucketsTotal
from warnings import warn
import numpy
import pyarrow
import pyarrow.compute as pyarrowCompute

datatypeCurveLocations = pyarrow.uint64()
datatypeDistinctCrossings = pyarrow.uint64()
bitWidthCurveLocationsMaximum: int = 60
bitWidthDistinctCrossingsMaximum: int = 60

@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def outfitDictionaryCurveGroups(state: MatrixMeandersState, dictionaryCurveLocations: dict[int, int]) -> tuple[MatrixMeandersState, dict[tuple[int, int], int]]:
	"""Outfit `dictionaryCurveGroups` so it may manage the computations for one iteration of the transfer matrix.

	Parameters
	----------
	state : MatrixMeandersState
		The current state of the computation, including `dictionaryCurveLocations`.

	Returns
	-------
	dictionaryCurveGroups : dict[tuple[int, int], int]
		A dictionary of `(groupAlpha, groupZulu)` to `distinctCrossings`.
	"""
	state.bitWidth = max(dictionaryCurveLocations.keys()).bit_length()
	dictionaryCurveGroups: dict[tuple[int, int], int] = {(curveLocations & state.locatorGroupAlpha, (curveLocations & state.locatorGroupZulu) >> 1): distinctCrossings
		for curveLocations, distinctCrossings in dictionaryCurveLocations.items()}
	return state, dictionaryCurveGroups

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
	"""Find the bit position for flipping paired curve endpoints in meander transfer matrices.

	Parameters
	----------
	intWithExtra_0b1 : int
		Binary representation of curve locations with an extra bit encoding parity information.

	Returns
	-------
	flipExtra_0b1_Here : int
		Bit mask indicating the position where the balance condition fails, formatted as 2^(2k).

	3L33T H@X0R
	------------
	Binary search for first negative balance in shifted bit pairs. Returns 2^(2k) mask for
	bit position k where cumulative balance counter transitions from non-negative to negative.

	Mathematics
	-----------
	Implements the Dyck path balance verification algorithm from Jensen's transfer matrix
	enumeration. Computes the position where ∑(i=0 to k) (-1)^b_i < 0 for the first time,
	where b_i are the bits of the input at positions 2i.

	"""
	findTheExtra_0b1: int = 0
	flipExtra_0b1_Here: int = 1
	while True:
		flipExtra_0b1_Here <<= 2
		if (intWithExtra_0b1 & flipExtra_0b1_Here) == 0:
			findTheExtra_0b1 += 1
		else:
			findTheExtra_0b1 -= 1
		if findTheExtra_0b1 < 0:
			break
	return flipExtra_0b1_Here

def countBigInt(state: MatrixMeandersState) -> MatrixMeandersState:
	"""Count meanders with matrix transfer algorithm using Python primitive `int` contained in a Python primitive `dict`.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing current `kOfMatrix`, `dictionaryCurveLocations`, and thresholds.

	Notes
	-----
	The algorithm is sophisticated, but this implementation is straightforward. Compute each index one at a time, compute each
	`curveLocations` one at a time, and compute each type of analysis one at a time.
	"""
	dictionaryCurveGroups: dict[tuple[int, int], int] = {}
	dictionaryCurveLocations: dict[int, int] = state.dictionaryCurveLocations.copy()
	state.dictionaryCurveLocations.clear()
	k: int = state.kOfMatrix
	MAXIMUMcurveLocations = 1 << (2 * k + 4)
	while (k > 0
			and (((max(dictionaryCurveLocations.keys()).bit_length()) > raiseIfNone(state.bitWidthCurveLocationsMaximum))
				or (MAXIMUMcurveLocations.bit_length() > raiseIfNone(state.bitWidthCurveLocationsMaximum))
				or ((max(dictionaryCurveLocations.values()).bit_length()) > raiseIfNone(state.bitWidthDistinctCrossingsMaximum))
		)):

		k -= 1
		MAXIMUMcurveLocations = 1 << (2 * k + 4)

		state, dictionaryCurveGroups = outfitDictionaryCurveGroups(state, dictionaryCurveLocations)
		dictionaryCurveLocations.clear()

		for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
			groupAlphaCurves: bool = groupAlpha > 1
			groupZuluHasCurves: bool = groupZulu > 1
			groupAlphaIsEven = groupZuluIsEven = 0

			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			# simple
			if curveLocationAnalysis < MAXIMUMcurveLocations:
				dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 1)) << 1)
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupZuluHasCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupAlphaCurves and groupZuluHasCurves and (groupAlphaIsEven or groupZuluIsEven):
				# aligned
				if groupAlphaIsEven and not groupZuluIsEven:
					groupAlpha ^= walkDyckPath(groupAlpha)
				elif groupZuluIsEven and not groupAlphaIsEven:
					groupZulu ^= walkDyckPath(groupZulu)

				curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

	state.kOfMatrix = k
	state.dictionaryCurveLocations = dictionaryCurveLocations
	return state

def countArrow(state: MatrixMeandersState) -> MatrixMeandersState:
	"""Count meanders with matrix transfer algorithm using pyarrow.

	Parameters
	----------
	state : MatrixMeandersState
		The computation state and settings.

	Returns
	-------
	state : MatrixMeandersState
		The computation state.
	"""
	arrowAnalyzed = pyarrow.table([
		pyarrow.array(list(state.dictionaryCurveLocations.keys()), type=state.datatypeCurveLocations)
		, pyarrow.array(list(state.dictionaryCurveLocations.values()), type=state.datatypeDistinctCrossings)
	], names=['analyzed', 'distinctCrossings'])
	state.dictionaryCurveLocations.clear()

	while (state.kOfMatrix > 0
		and (int(pyarrowCompute.max(arrowAnalyzed['analyzed']).as_py()).bit_length() <= raiseIfNone(state.bitWidthCurveLocationsMaximum))
		and (state.MAXIMUMcurveLocations.bit_length() <= raiseIfNone(state.bitWidthCurveLocationsMaximum))
		and (int(pyarrowCompute.max(arrowAnalyzed['distinctCrossings']).as_py()).bit_length() <= raiseIfNone(state.bitWidthDistinctCrossingsMaximum))):

		def aggregateCurveLocations() -> None:
			nonlocal arrowAnalyzed
			slicedTable = arrowAnalyzed.slice(0, state.indexStartAnalyzed)
			groupedResult = slicedTable.group_by('analyzed').aggregate([('distinctCrossings', 'sum')])
			# The result will have columns 'analyzed' and 'distinctCrossings_sum', so we need to rename
			arrowAnalyzed = groupedResult.rename_columns(['analyzed', 'distinctCrossings'])

		def analyzeCurveLocationsAligned() -> None:
			"""Compute `curveLocations` from `groupAlpha` and `groupZulu` if at least one is an even number.

			Before computing `curveLocations`, some values of `groupAlpha` and `groupZulu` are modified.

			Warning
			-------
			This function deletes rows from `arrowCurveLocations`. Always run this analysis last.

			Formula
			-------
			```python
			if groupAlpha > 1 and groupZulu > 1 and (groupAlphaIsEven or groupZuluIsEven):
				curveLocations = (groupAlpha >> 2) | ((groupZulu >> 2) << 1)
			```
			"""
			nonlocal arrowCurveLocations

			# NOTE Step 1 drop unqualified rows

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupAlpha)) # Ima `groupAlpha`.

			arrowCurveLocations = arrowCurveLocations.filter(pyarrowCompute.greater(ImaGroupZulpha, pyarrow.scalar(1))) # if groupAlphaHasCurves

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupZulu)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)

			arrowCurveLocations = arrowCurveLocations.filter(pyarrowCompute.greater(ImaGroupZulpha, pyarrow.scalar(1))) # if groupZuluHasCurves

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(0b10)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_and(ImaGroupZulpha, pyarrow.scalar(1)) # (groupZulu & 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_xor(ImaGroupZulpha, pyarrow.scalar(1)) # (1 - (groupZulu ...))
			arrowCurveLocations = arrowCurveLocations.set_column(1, 'analyzed', ImaGroupZulpha) # selectorGroupZuluAtEven

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(1)) # (groupAlpha & 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_xor(ImaGroupZulpha, pyarrow.scalar(1)) # (1 - (groupAlpha ...))
			selectorGroupAlphaAtODD = pyarrowCompute.cast(ImaGroupZulpha, pyarrow.bool_()) # selectorGroupAlphaAtODD

			filterCondition = pyarrowCompute.or_(selectorGroupAlphaAtODD, pyarrowCompute.cast(arrowCurveLocations['analyzed'], pyarrow.bool_())) # if (groupAlphaIsEven or groupZuluIsEven)
			arrowCurveLocations = arrowCurveLocations.filter(filterCondition)

			# NOTE Step 2 modify rows

			# Make a selector for groupZuluAtEven, so you can modify groupAlpha
			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(0b10)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_and(ImaGroupZulpha, pyarrow.scalar(1)) # (groupZulu & 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_xor(ImaGroupZulpha, pyarrow.scalar(1)) # (1 - (groupZulu ...))
			selectorGroupZuluAtEven = pyarrowCompute.cast(ImaGroupZulpha, pyarrow.bool_()) # selectorGroupZuluAtEven

			analyzedColumn = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupAlpha)) # (groupAlpha)

			# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
			notGroupZuluAtEven = pyarrowCompute.invert(selectorGroupZuluAtEven)
			# Handle the conditional modification more carefully for array length matching
			modifiedAnalyzedColumn = []
			analyzedColumnList = analyzedColumn.to_pylist()
			notGroupZuluAtEvenList = notGroupZuluAtEven.to_pylist()

			for analyzed_val, should_modify in zip(analyzedColumnList, notGroupZuluAtEvenList, strict=True):
				if should_modify:
					modifiedAnalyzedColumn.append(int(analyzed_val) ^ walkDyckPath(int(analyzed_val)))
				else:
					modifiedAnalyzedColumn.append(analyzed_val)

			analyzedColumn = pyarrow.array(modifiedAnalyzedColumn, type=state.datatypeCurveLocations)

			# if groupZuluIsEven and not groupAlphaIsEven, modifyGroupZuluPairedToOdd
			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupZulu)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)

			groupAlphaIsOdd = pyarrowCompute.cast(pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(1)), pyarrow.bool_())

			# Handle the conditional modification more carefully for array length matching
			modifiedZuluColumn = []
			ImaGroupZulphaList = ImaGroupZulpha.to_pylist()
			groupAlphaIsOddList = groupAlphaIsOdd.to_pylist()

			for zulu_val, should_modify in zip(ImaGroupZulphaList, groupAlphaIsOddList, strict=True):
				if should_modify:
					modifiedZuluColumn.append(int(zulu_val) ^ walkDyckPath(int(zulu_val)))
				else:
					modifiedZuluColumn.append(zulu_val)

			ImaGroupZulpha = pyarrow.array(modifiedZuluColumn, type=state.datatypeCurveLocations)

			# NOTE Step 3 compute curveLocations
			analyzedColumn = pyarrowCompute.shift_right(analyzedColumn, pyarrow.scalar(2)) # (groupAlpha >> 2)

			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(2)) # (groupZulu >> 2)
			ImaGroupZulpha = pyarrowCompute.shift_left(ImaGroupZulpha, pyarrow.scalar(1)) # ((groupZulu ...) << 1)

			analyzedColumn = pyarrowCompute.bit_wise_or(analyzedColumn, ImaGroupZulpha) # ... | (groupZulu ...)

			analyzedColumn = pyarrowCompute.if_else(
				pyarrowCompute.greater_equal(analyzedColumn, pyarrow.scalar(state.MAXIMUMcurveLocations)),
				pyarrow.scalar(0, type=state.datatypeCurveLocations),
				analyzedColumn
			)
			arrowCurveLocations = arrowCurveLocations.set_column(1, 'analyzed', analyzedColumn)

		def analyzeCurveLocationsAlpha() -> None:
			"""Compute `curveLocations` from `groupAlpha`.

			Formula
			-------
			```python
			if groupAlpha > 1:
				curveLocations = ((1 - (groupAlpha & 1)) << 1) | (groupZulu << 3) | (groupAlpha >> 2)
			# `(1 - (groupAlpha & 1)` is an evenness test.
			```
			"""
			nonlocal arrowCurveLocations
			analyzedColumn = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(1)) # (groupAlpha & 1)
			analyzedColumn = pyarrowCompute.bit_wise_xor(analyzedColumn, pyarrow.scalar(1)) # (1 - (groupAlpha ...))

			analyzedColumn = pyarrowCompute.shift_left(analyzedColumn, pyarrow.scalar(1)) # ((groupAlpha ...) << 1)

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupZulu)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)

			ImaGroupZulpha = pyarrowCompute.shift_left(ImaGroupZulpha, pyarrow.scalar(3)) # (groupZulu << 3)
			analyzedColumn = pyarrowCompute.bit_wise_or(analyzedColumn, ImaGroupZulpha) # ... | (groupZulu ...)

			analyzedColumn = pyarrowCompute.shift_left(analyzedColumn, pyarrow.scalar(2)) # ... | (groupAlpha >> 2)

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupAlpha)) # Ima `groupAlpha`.

			analyzedColumn = pyarrowCompute.bit_wise_or(analyzedColumn, ImaGroupZulpha) # ... | (groupAlpha)
			analyzedColumn = pyarrowCompute.shift_right(analyzedColumn, pyarrow.scalar(2)) # (... >> 2)

			analyzedColumn = pyarrowCompute.if_else(
				pyarrowCompute.less_equal(ImaGroupZulpha, pyarrow.scalar(1)), # if groupAlpha > 1
				pyarrow.scalar(0, type=state.datatypeCurveLocations),
				analyzedColumn
			)

			analyzedColumn = pyarrowCompute.if_else(
				pyarrowCompute.greater_equal(analyzedColumn, pyarrow.scalar(state.MAXIMUMcurveLocations)),
				pyarrow.scalar(0, type=state.datatypeCurveLocations),
				analyzedColumn
			)
			arrowCurveLocations = arrowCurveLocations.set_column(1, 'analyzed', analyzedColumn)

		def analyzeCurveLocationsSimple() -> None:
			"""Compute curveLocations with the 'simple' bridges formula.

			Formula
			-------
			```python
			curveLocations = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			```

			Notes
			-----
			Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the last bits are '0b00'. '0 + 3'
			is '0b11', and '0b00 | 0b11' is also '0b11'.

			"""
			nonlocal arrowCurveLocations
			analyzedColumn = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupAlpha))

			groupZulu = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupZulu))
			groupZulu = pyarrowCompute.shift_right(groupZulu, pyarrow.scalar(1)) # (groupZulu >> 1)
			groupZulu = pyarrowCompute.shift_left(groupZulu, pyarrow.scalar(1)) # (groupZulu << 1)

			analyzedColumn = pyarrowCompute.bit_wise_or(analyzedColumn, groupZulu) # ((groupAlpha | (groupZulu ...))

			analyzedColumn = pyarrowCompute.shift_left(analyzedColumn, pyarrow.scalar(2)) # (... << 2)
			analyzedColumn = pyarrowCompute.add(analyzedColumn, pyarrow.scalar(3)) # (...) | 3

			analyzedColumn = pyarrowCompute.if_else(
				pyarrowCompute.greater_equal(analyzedColumn, pyarrow.scalar(state.MAXIMUMcurveLocations)),
				pyarrow.scalar(0, type=state.datatypeCurveLocations),
				analyzedColumn
			)
			arrowCurveLocations = arrowCurveLocations.set_column(1, 'analyzed', analyzedColumn)

		def analyzeCurveLocationsZulu() -> None:
			"""Compute `curveLocations` from `groupZulu`.

			Formula
			-------
			```python
			if groupZulu > 1:
				curveLocations = (1 - (groupZulu & 1)) | (groupAlpha << 2) | (groupZulu >> 1)
			```
			"""
			nonlocal arrowCurveLocations
			analyzedColumn = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(0b10)) # Ima `groupZulu`.
			analyzedColumn = pyarrowCompute.shift_right(analyzedColumn, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)
			analyzedColumn = pyarrowCompute.bit_wise_and(analyzedColumn, pyarrow.scalar(1)) # (groupZulu & 1)
			analyzedColumn = pyarrowCompute.bit_wise_xor(analyzedColumn, pyarrow.scalar(1)) # (1 - (groupZulu ...))

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupAlpha)) # Ima `groupAlpha`.

			ImaGroupZulpha = pyarrowCompute.shift_left(ImaGroupZulpha, pyarrow.scalar(2)) # (groupAlpha << 2)
			analyzedColumn = pyarrowCompute.bit_wise_or(analyzedColumn, ImaGroupZulpha) # ... | (groupAlpha ...)

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupZulu)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)

			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # (groupZulu >> 1)

			analyzedColumn = pyarrowCompute.bit_wise_or(analyzedColumn, ImaGroupZulpha) # ... | (groupZulu ...)

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupZulu)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)

			analyzedColumn = pyarrowCompute.if_else(
				pyarrowCompute.less_equal(ImaGroupZulpha, pyarrow.scalar(1)), # if groupZulu > 1
				pyarrow.scalar(0, type=state.datatypeCurveLocations),
				analyzedColumn
			)

			analyzedColumn = pyarrowCompute.if_else(
				pyarrowCompute.greater_equal(analyzedColumn, pyarrow.scalar(state.MAXIMUMcurveLocations)),
				pyarrow.scalar(0, type=state.datatypeCurveLocations),
				analyzedColumn
			)
			arrowCurveLocations = arrowCurveLocations.set_column(1, 'analyzed', analyzedColumn)

		def recordCurveLocations() -> None:
			nonlocal arrowAnalyzed

			nonZeroMask = pyarrowCompute.greater(arrowCurveLocations['analyzed'], pyarrow.scalar(0))
			nonZeroCount = int(pyarrowCompute.sum(pyarrowCompute.cast(nonZeroMask, pyarrow.int64())).as_py() or 0)
			indexStopAnalyzed: int = state.indexStartAnalyzed + nonZeroCount

			if indexStopAnalyzed > state.indexStartAnalyzed:
				if arrowAnalyzed.num_rows < indexStopAnalyzed:
					# Extend arrowAnalyzed table
					extensionSize = indexStopAnalyzed - arrowAnalyzed.num_rows
					analyzedExtension = pyarrow.array([0] * extensionSize, type=state.datatypeCurveLocations)
					distinctCrossingsExtension = pyarrow.array([0] * extensionSize, type=state.datatypeDistinctCrossings)
					extensionTable = pyarrow.table([
						analyzedExtension,
						distinctCrossingsExtension
					], names=['analyzed', 'distinctCrossings'])
					arrowAnalyzed = pyarrow.concat_tables([arrowAnalyzed, extensionTable])
					warn(f"Lengthened `arrowAnalyzed` to {indexStopAnalyzed=}; n={state.n}, {state.kOfMatrix=}.", stacklevel=2)

				filteredRows = arrowCurveLocations.filter(nonZeroMask)

				# Update the analyzed and distinctCrossings columns more efficiently
				if filteredRows.num_rows > 0:
					# Convert only the filtered non-zero data to Python lists (much smaller dataset)
					filteredAnalyzed = filteredRows['analyzed'].to_pylist()
					filteredDistinctCrossings = filteredRows['distinctCrossings'].to_pylist()

					# Create replacement arrays for just the slice we need to update
					updatedAnalyzed = pyarrow.array(filteredAnalyzed, type=state.datatypeCurveLocations)
					updatedDistinctCrossings = pyarrow.array(filteredDistinctCrossings, type=state.datatypeDistinctCrossings)

					# Use set_column to update efficiently
					newAnalyzedColumn = pyarrow.concat_arrays([
						arrowAnalyzed['analyzed'].slice(0, state.indexStartAnalyzed).combine_chunks(),
						updatedAnalyzed,
						arrowAnalyzed['analyzed'].slice(indexStopAnalyzed).combine_chunks()
					])

					newDistinctCrossingsColumn = pyarrow.concat_arrays([
						arrowAnalyzed['distinctCrossings'].slice(0, state.indexStartAnalyzed).combine_chunks(),
						updatedDistinctCrossings,
						arrowAnalyzed['distinctCrossings'].slice(indexStopAnalyzed).combine_chunks()
					])

					arrowAnalyzed = pyarrow.table([
						newAnalyzedColumn,
						newDistinctCrossingsColumn
					], names=['analyzed', 'distinctCrossings'])

				state.indexStartAnalyzed = indexStopAnalyzed

		arrowCurveLocations = pyarrow.table([
			arrowAnalyzed['analyzed']
			, pyarrow.array([0] * arrowAnalyzed.num_rows, type=state.datatypeCurveLocations)
			, arrowAnalyzed['distinctCrossings']
		], names=['curveLocations', 'analyzed', 'distinctCrossings'])

		state.bitWidth = int(pyarrowCompute.max(arrowCurveLocations['curveLocations']).as_py()).bit_length()

		del arrowAnalyzed
		goByeBye()

		length: int = getBucketsTotal(state)
		arrowAnalyzed = pyarrow.table([
			pyarrow.array([0] * length, type=state.datatypeCurveLocations)
			, pyarrow.array([0] * length, type=state.datatypeDistinctCrossings)
		], names=['analyzed', 'distinctCrossings'])

		state.indexStartAnalyzed = 0

		state.kOfMatrix -= 1

		analyzeCurveLocationsSimple()
		recordCurveLocations()

		analyzeCurveLocationsAlpha()
		recordCurveLocations()

		analyzeCurveLocationsZulu()
		recordCurveLocations()

		analyzeCurveLocationsAligned()
		recordCurveLocations()
		del arrowCurveLocations
		goByeBye()

		aggregateCurveLocations()

	# Convert back to dictionary
	analyzedList = arrowAnalyzed['analyzed'].to_pylist()
	distinctCrossingsList = arrowAnalyzed['distinctCrossings'].to_pylist()
	state.dictionaryCurveLocations = {analyzed: distinctCrossings for analyzed, distinctCrossings in zip(analyzedList, distinctCrossingsList, strict=True) if analyzed != 0}
	return state

def doTheNeedful(state: MatrixMeandersState) -> int:
	"""Compute a(n) meanders with the transfer matrix algorithm.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing current `kOfMatrix`, `dictionaryCurveLocations`, and thresholds.

	Returns
	-------
	a(n) : int
		The computed value of a(n).

	Notes
	-----
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex

	See Also
	--------
	https://oeis.org/A000682
	https://oeis.org/A005316
	"""
	while state.kOfMatrix > 0:
		bitWidthCurveLocations: int = max(state.dictionaryCurveLocations.keys()).bit_length()
		bitWidthDistinctCrossings: int = max(state.dictionaryCurveLocations.values()).bit_length()

		goByeBye()

		if ((bitWidthCurveLocations > raiseIfNone(state.bitWidthCurveLocationsMaximum))
			or (state.MAXIMUMcurveLocations.bit_length() > raiseIfNone(state.bitWidthCurveLocationsMaximum))
			or (bitWidthDistinctCrossings > raiseIfNone(state.bitWidthDistinctCrossingsMaximum))):
			state = countBigInt(state)
		else:
			state = countArrow(state)

	return sum(state.dictionaryCurveLocations.values())

@cache
def A000682(n: int) -> int:
	"""Compute A000682(n)."""
	oeisID = 'A000682'

	kOfMatrix: int = n - 1

	if n & 0b1:
		curveLocations: int = 5
	else:
		curveLocations = 1
	listCurveLocations: list[int] = [(curveLocations << 1) | curveLocations]

	MAXIMUMcurveLocations: int = 1 << (2 * kOfMatrix + 4)
	while listCurveLocations[-1] < MAXIMUMcurveLocations:
		curveLocations = (curveLocations << 4) | 0b101 # == curveLocations * 2**4 + 5
		listCurveLocations.append((curveLocations << 1) | curveLocations)

	dictionaryCurveLocations: dict[int, int] = dict.fromkeys(listCurveLocations, 1)

	state = MatrixMeandersState(n, oeisID, kOfMatrix, dictionaryCurveLocations
		, datatypeCurveLocations, datatypeDistinctCrossings, bitWidthCurveLocationsMaximum, bitWidthDistinctCrossingsMaximum)

	return doTheNeedful(state)

@cache
def A005316(n: int) -> int:
	"""Compute A005316(n)."""
	oeisID = 'A005316'

	kOfMatrix: int = n - 1

	if n & 0b1:
		dictionaryCurveLocations: dict[int, int] = {15: 1}
	else:
		dictionaryCurveLocations = {22: 1}

	state = MatrixMeandersState(n, oeisID, kOfMatrix, dictionaryCurveLocations
		, datatypeCurveLocations, datatypeDistinctCrossings, bitWidthCurveLocationsMaximum, bitWidthDistinctCrossingsMaximum)

	return doTheNeedful(state)
