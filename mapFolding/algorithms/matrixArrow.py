"""Count meanders with matrix transfer algorithm.

Notes
-----
- Odd/even of `groupAlpha` == the odd/even of `curveLocations`. Proof: `groupAlphaIsEven = curveLocations & 1 & 1 ^ 1`.
- Odd/even of `groupZulu` == `curveLocations` second-least significant bit. So `groupZuluIsEven = bool(curveLocations & 2 ^ 2)`.
"""
from functools import cache
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding.algorithms.getBucketsTotal import getBucketsTotal
from pyarrow.compute import bit_wise_and, bit_wise_or, shift_left  # pyright: ignore[reportUnknownVariableType]
from typing import cast
from warnings import warn
import dataclasses
import numpy
import pyarrow
import pyarrow.compute as pyarrowCompute

@dataclasses.dataclass
class MatrixArrowState:
	"""State for matrix meanders pyarrow."""

	n: int
	oeisID: str
	kOfMatrix: int
	dictionaryCurveLocations: dict[int, int]

	datatypeCurveLocations: pyarrow.DataType = dataclasses.field(default_factory=pyarrow.uint64)
	datatypeDistinctCrossings: pyarrow.DataType = dataclasses.field(default_factory=pyarrow.uint64)

	bitWidthCurveLocationsMaximum: int | None = None
	bitWidthDistinctCrossingsMaximum: int | None = None

	bitWidth: int = 0
	indexStartAnalyzed: int = 0

	def __post_init__(self) -> None:
		"""Post init."""
		if self.bitWidthCurveLocationsMaximum is None:
			_bitWidthOfFixedSizeInteger = self.datatypeCurveLocations.bit_width

			_offsetNecessary: int = 3 # For example, `groupZulu << 3`.
			_offsetSafety: int = 1 # I don't have mathematical proof of how many extra bits I need.
			_offset: int = _offsetNecessary + _offsetSafety

			self.bitWidthCurveLocationsMaximum = _bitWidthOfFixedSizeInteger - _offset

			del _bitWidthOfFixedSizeInteger, _offsetNecessary, _offsetSafety, _offset

		if self.bitWidthDistinctCrossingsMaximum is None:
			_bitWidthOfFixedSizeInteger = self.datatypeDistinctCrossings.bit_width

			_offsetNecessary: int = 0 # I don't know of any.
			_offsetEstimation: int = 3 # See reference directory.
			_offsetSafety: int = 1
			_offset: int = _offsetNecessary + _offsetEstimation + _offsetSafety

			self.bitWidthDistinctCrossingsMaximum = _bitWidthOfFixedSizeInteger - _offset

			del _bitWidthOfFixedSizeInteger, _offsetNecessary, _offsetEstimation, _offsetSafety, _offset

	@property
	def MAXIMUMcurveLocations(self) -> int:
		"""Compute the maximum value of `curveLocations` for the current iteration of the transfer matrix."""
		return 1 << (2 * self.kOfMatrix + 4)

	@property
	def arrowMAXIMUMcurveLocations(self) -> pyarrow.Scalar: # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
		"""Compute the maximum value of `curveLocations` for the current iteration of the transfer matrix."""
		return pyarrow.scalar(self.MAXIMUMcurveLocations, self.datatypeCurveLocations)

	@property
	def locatorGroupAlpha(self) -> int:
		"""Compute an odd-parity bit-mask with `bitWidth` bits.

		Notes
		-----
		In binary, `locatorGroupAlpha` has alternating 0s and 1s and ends with a 1, such as '101', '0101', and '10101'. The last
		digit is in the 1's column, but programmers usually call it the "least significant bit" (LSB). If we count the columns
		from the right, the 1's column is column 1, the 2's column is column 2, the 4's column is column 3, and so on. When
		counting this way, `locatorGroupAlpha` has 1s in the columns with odd index numbers. Mathematicians and programmers,
		therefore, tend to call `locatorGroupAlpha` something like the "odd bit-mask", the "odd-parity numbers", or simply "odd
		mask" or "odd numbers". In addition to "odd" being inherently ambiguous in this context, this algorithm also segregates
		odd numbers from even numbers, so I avoid using "odd" and "even" in the names of these bit-masks.

		"""
		return sum(1 << one for one in range(0, self.bitWidth, 2))

	@property
	def locatorGroupZulu(self) -> int:
		"""Compute an even-parity bit-mask with `bitWidth` bits."""
		return sum(1 << one for one in range(1, self.bitWidth, 2))

	@property
	def arrowLocatorAlpha(self) -> pyarrow.Scalar: # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
		"""Compute an odd-parity bit-mask with `bitWidth` bits.

		Notes
		-----
		In binary, `locatorGroupAlpha` has alternating 0s and 1s and ends with a 1, such as '101', '0101', and '10101'. The last
		digit is in the 1's column, but programmers usually call it the "least significant bit" (LSB). If we count the columns
		from the right, the 1's column is column 1, the 2's column is column 2, the 4's column is column 3, and so on. When
		counting this way, `locatorGroupAlpha` has 1s in the columns with odd index numbers. Mathematicians and programmers,
		therefore, tend to call `locatorGroupAlpha` something like the "odd bit-mask", the "odd-parity numbers", or simply "odd
		mask" or "odd numbers". In addition to "odd" being inherently ambiguous in this context, this algorithm also segregates
		odd numbers from even numbers, so I avoid using "odd" and "even" in the names of these bit-masks.

		"""
		return pyarrow.scalar(self.locatorGroupAlpha, self.datatypeCurveLocations)

	@property
	def arrowLocatorZulu(self) -> pyarrow.Scalar: # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
		"""Compute an even-parity bit-mask with `bitWidth` bits."""
		return pyarrow.scalar(self.locatorGroupZulu, self.datatypeCurveLocations)

@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def outfitDictionaryCurveGroups(state: MatrixArrowState) -> dict[tuple[int, int], int]:
	"""Outfit `dictionaryCurveGroups` so it may manage the computations for one iteration of the transfer matrix.

	Parameters
	----------
	state : MatrixArrowState
		The current state of the computation, including `dictionaryCurveLocations`.

	Returns
	-------
	dictionaryCurveGroups : dict[tuple[int, int], int]
		A dictionary of `(groupAlpha, groupZulu)` to `distinctCrossings`.
	"""
	state.bitWidth = max(state.dictionaryCurveLocations.keys()).bit_length()
	return {(curveLocations & state.locatorGroupAlpha, (curveLocations & state.locatorGroupZulu) >> 1): distinctCrossings
		for curveLocations, distinctCrossings in state.dictionaryCurveLocations.items()}

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

def countBigInt(state: MatrixArrowState) -> MatrixArrowState:
	"""Count meanders with matrix transfer algorithm using Python primitive `int` contained in a Python primitive `dict`.

	Parameters
	----------
	state : MatrixArrowState
		The algorithm state containing current `kOfMatrix`, `dictionaryCurveLocations`, and thresholds.

	Notes
	-----
	The algorithm is sophisticated, but this implementation is straightforward. Compute each index one at a time, compute each
	`curveLocations` one at a time, and compute each type of analysis one at a time.
	"""
	dictionaryCurveGroups: dict[tuple[int, int], int] = {}

	while (state.kOfMatrix > 0
		and ((max(state.dictionaryCurveLocations.keys()).bit_length() > raiseIfNone(state.bitWidthCurveLocationsMaximum))
		or (state.MAXIMUMcurveLocations.bit_length() > raiseIfNone(state.bitWidthCurveLocationsMaximum))
		or (max(state.dictionaryCurveLocations.values()).bit_length() > raiseIfNone(state.bitWidthDistinctCrossingsMaximum)))):

		state.kOfMatrix -= 1

		dictionaryCurveGroups = outfitDictionaryCurveGroups(state)
		state.dictionaryCurveLocations.clear()
		LessThanMaximumTotal: int = 0 # for data collection

		for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
			groupAlphaCurves: bool = groupAlpha > 1
			groupZuluHasCurves: bool = groupZulu > 1
			groupAlphaIsEven = groupZuluIsEven = 0

			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			# simple
			if curveLocationAnalysis < state.MAXIMUMcurveLocations:
				state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings
				LessThanMaximumTotal += 1

			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 1)) << 1)
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings
					LessThanMaximumTotal += 1

			if groupZuluHasCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings
					LessThanMaximumTotal += 1

			if groupAlphaCurves and groupZuluHasCurves and (groupAlphaIsEven or groupZuluIsEven):
				# aligned
				if groupAlphaIsEven and not groupZuluIsEven:
					groupAlpha ^= walkDyckPath(groupAlpha)  # noqa: PLW2901
				elif groupZuluIsEven and not groupAlphaIsEven:
					groupZulu ^= walkDyckPath(groupZulu)  # noqa: PLW2901

				curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings
					LessThanMaximumTotal += 1

		if state.n >= 45: # for data collection
			print(state.n, state.kOfMatrix+1, LessThanMaximumTotal, sep=',')  # noqa: T201

	return state

def countArrow(state: MatrixArrowState) -> MatrixArrowState:
	"""Count meanders with matrix transfer algorithm using pyarrow.

	Parameters
	----------
	state : MatrixArrowState
		The computation state and settings.

	Returns
	-------
	state : MatrixArrowState
		The computation state.
	"""
	schemaAnalyzed: pyarrow.Schema = pyarrow.schema([
		pyarrow.field('analyzed', state.datatypeCurveLocations, nullable=False)
		, pyarrow.field('distinctCrossings', state.datatypeDistinctCrossings, nullable=False)
	])

	schemaCurveLocations: pyarrow.Schema = pyarrow.schema([
		pyarrow.field('curveLocations', state.datatypeCurveLocations, nullable=False)
		, pyarrow.field('analyzed', state.datatypeCurveLocations, nullable=False)
		, pyarrow.field('distinctCrossings', state.datatypeDistinctCrossings, nullable=False)
	])

	arrowAnalyzed: pyarrow.Table = pyarrow.table({
		'analyzed': list(state.dictionaryCurveLocations.keys())
		, 'distinctCrossings': list(state.dictionaryCurveLocations.values())
	}, schema=schemaAnalyzed)
	state.dictionaryCurveLocations.clear()

	a0 = pyarrow.scalar(0, type=state.datatypeCurveLocations)
	a1 = pyarrow.scalar(1, type=state.datatypeCurveLocations)
	a2 = pyarrow.scalar(2, type=state.datatypeCurveLocations)
	a3 = pyarrow.scalar(3, type=state.datatypeCurveLocations)

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

			arrowCurveLocations = arrowCurveLocations.filter(pyarrowCompute.greater(ImaGroupZulpha, pyarrow.scalar(1))) # pyright: ignore[reportUnknownMemberType] # if groupAlphaHasCurves

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupZulu)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)

			arrowCurveLocations = arrowCurveLocations.filter(pyarrowCompute.greater(ImaGroupZulpha, pyarrow.scalar(1))) # pyright: ignore[reportUnknownMemberType] # if groupZuluHasCurves

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(0b10)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_and(ImaGroupZulpha, pyarrow.scalar(1)) # (groupZulu & 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_xor(ImaGroupZulpha, pyarrow.scalar(1)) # (1 - (groupZulu ...))
			arrowCurveLocations = arrowCurveLocations.set_column(1, 'analyzed', ImaGroupZulpha) # selectorGroupZuluAtEven

			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(1)) # (groupAlpha & 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_xor(ImaGroupZulpha, pyarrow.scalar(1)) # (1 - (groupAlpha ...))
			selectorGroupAlphaAtODD = pyarrowCompute.cast(ImaGroupZulpha, pyarrow.bool_()) # pyright: ignore[reportUnknownMemberType] # selectorGroupAlphaAtODD

			filterCondition = pyarrowCompute.or_(selectorGroupAlphaAtODD, pyarrowCompute.cast(arrowCurveLocations['analyzed'], pyarrow.bool_())) # pyright: ignore[reportArgumentType, reportCallIssue, reportUnknownMemberType, reportUnknownVariableType] # if (groupAlphaIsEven or groupZuluIsEven)
			arrowCurveLocations = arrowCurveLocations.filter(filterCondition) # pyright: ignore[reportUnknownArgumentType]

			# NOTE Step 2 modify rows

			# Make a selector for groupZuluAtEven, so you can modify groupAlpha
			ImaGroupZulpha = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(0b10)) # Ima `groupZulu`.
			ImaGroupZulpha = pyarrowCompute.shift_right(ImaGroupZulpha, pyarrow.scalar(1)) # Ima `groupZulu` (groupZulu >> 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_and(ImaGroupZulpha, pyarrow.scalar(1)) # (groupZulu & 1)
			ImaGroupZulpha = pyarrowCompute.bit_wise_xor(ImaGroupZulpha, pyarrow.scalar(1)) # (1 - (groupZulu ...))
			selectorGroupZuluAtEven = pyarrowCompute.cast(ImaGroupZulpha, pyarrow.bool_()) # pyright: ignore[reportUnknownMemberType] # selectorGroupZuluAtEven

			analyzedColumn = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], pyarrow.scalar(state.locatorGroupAlpha)) # (groupAlpha)

			# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
			notGroupZuluAtEven = pyarrowCompute.invert(selectorGroupZuluAtEven) # pyright: ignore[reportArgumentType, reportCallIssue, reportUnknownVariableType]
			# Handle the conditional modification more carefully for array length matching
			modifiedAnalyzedColumn = []
			analyzedColumnList = analyzedColumn.to_pylist()
			notGroupZuluAtEvenList = notGroupZuluAtEven.to_pylist() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

			for analyzed_val, should_modify in zip(analyzedColumnList, notGroupZuluAtEvenList, strict=True): # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
				if should_modify:
					modifiedAnalyzedColumn.append(int(analyzed_val) ^ walkDyckPath(int(analyzed_val))) # pyright: ignore[reportUnknownMemberType]
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
			"""
			nonlocal arrowCurveLocations
			analyzed =	cast('pyarrow.UInt64Array', bit_wise_or( # pyright: ignore[reportUnknownVariableType, reportCallIssue, reportUnknownArgumentType, reportArgumentType]
				shift_left(  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
					bit_wise_or(  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
						bit_wise_and(  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
							arrowCurveLocations['curveLocations']  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
							, state.arrowLocatorAlpha # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
						)  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
						, shift_left(  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
							bit_wise_and(  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
								arrowCurveLocations['curveLocations']  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
								, state.arrowLocatorZulu # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
							)  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
							, a1  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
						)  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
					)  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
					, a2  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
				)  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
				, a3  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
			))  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]

			analyzed = pyarrowCompute.if_else(
				pyarrowCompute.greater_equal(analyzed, state.arrowMAXIMUMcurveLocations), # pyright: ignore[reportUnknownMemberType]
				a0,
				analyzed
			)

			arrowCurveLocations = arrowCurveLocations.set_column(1, 'analyzed', analyzed)

			# analyzedColumn = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], state.arrowLocatorAlpha) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

			# groupZulu = pyarrowCompute.bit_wise_and(arrowCurveLocations['curveLocations'], state.arrowLocatorZulu) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
			# groupZulu = pyarrowCompute.shift_right(groupZulu, pyarrow.scalar(1)) # (groupZulu >> 1)
			# groupZulu = pyarrowCompute.shift_left(groupZulu, pyarrow.scalar(1)) # (groupZulu << 1)

			# analyzedColumn = pyarrowCompute.bit_wise_or(analyzedColumn, groupZulu) # ((groupAlpha | (groupZulu ...))

			# analyzedColumn = pyarrowCompute.shift_left(analyzedColumn, pyarrow.scalar(2)) # (... << 2)
			# analyzedColumn = pyarrowCompute.add(analyzedColumn, pyarrow.scalar(3)) # (...) | 3

			# analyzedColumn = pyarrowCompute.if_else(
			# 	pyarrowCompute.greater_equal(analyzedColumn, pyarrow.scalar(state.MAXIMUMcurveLocations)),
			# 	pyarrow.scalar(0, type=state.datatypeCurveLocations),
			# 	analyzedColumn
			# )
			# arrowCurveLocations = arrowCurveLocations.set_column(1, 'analyzed', analyzedColumn)

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

		# def recordCurveLocations() -> None:
		# 	nonlocal dataframeAnalyzed

		# 	indexStopAnalyzed: int = state.indexStartAnalyzed + int((dataframeCurveLocations['analyzed'] > 0).sum()) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

		# 	if indexStopAnalyzed > state.indexStartAnalyzed:
		# 		dataframeAnalyzed.loc[state.indexStartAnalyzed:indexStopAnalyzed - 1, ['analyzed', 'distinctCrossings']] = (
		# 			dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0), ['analyzed', 'distinctCrossings']
		# 						].to_numpy(dtype=state.datatypeCurveLocations, copy=False)
		# 		)

		# 		state.indexStartAnalyzed = indexStopAnalyzed

		# 	del indexStopAnalyzed

		def effYouClaude() -> None:
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

		arrowCurveLocations = pyarrow.table({
			'curveLocations': cast('pyarrow.UInt64Array', arrowAnalyzed['analyzed'])
			, 'analyzed': [0] * arrowAnalyzed.num_rows
			, 'distinctCrossings': cast('pyarrow.UInt64Array', arrowAnalyzed['distinctCrossings'])
		}, schema=schemaCurveLocations)

		del arrowAnalyzed
		goByeBye()

		state.bitWidth = int(pyarrowCompute.max(arrowCurveLocations['curveLocations'])).bit_length()

		length: int = getBucketsTotal(state)
		arrowAnalyzed: pyarrow.Table = pyarrow.table({
			'analyzed': [0] * length
			, 'distinctCrossings': [0] * length
		}, schema=schemaAnalyzed)

		state.indexStartAnalyzed = 0

		state.kOfMatrix -= 1

		analyzeCurveLocationsSimple()
		effYouClaude()

		analyzeCurveLocationsAlpha()
		effYouClaude()

		analyzeCurveLocationsZulu()
		effYouClaude()

		analyzeCurveLocationsAligned()
		effYouClaude()
		del arrowCurveLocations
		goByeBye()

		aggregateCurveLocations()

	# Garbage
	analyzedList = arrowAnalyzed['analyzed'].to_pylist()
	distinctCrossingsList = arrowAnalyzed['distinctCrossings'].to_pylist()
	state.dictionaryCurveLocations = {analyzed: distinctCrossings for analyzed, distinctCrossings in zip(analyzedList, distinctCrossingsList, strict=True) if analyzed != 0}
	return state

def doTheNeedful(state: MatrixArrowState) -> int:
	"""Compute a(n) meanders with the transfer matrix algorithm.

	Parameters
	----------
	state : MatrixArrowState
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

	state = MatrixArrowState(n, oeisID, kOfMatrix, dictionaryCurveLocations)

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

	state = MatrixArrowState(n, oeisID, kOfMatrix, dictionaryCurveLocations)

	return doTheNeedful(state)
	return doTheNeedful(state)
