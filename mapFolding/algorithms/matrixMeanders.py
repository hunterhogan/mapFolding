"""Count meanders with matrix transfer algorithm."""
# ruff: noqa: D103
# ERA001
from functools import cache
from gc import collect as goByeBye, set_threshold
from tqdm import tqdm
from typing import Any
import gc
import numpy
import pandas
import sys

# ----------------- environment configuration -------------------------------------------------------------------------

_bitWidthOfFixedSizeInteger: int = 64

_bitWidthOffsetCurveLocationsNecessary: int = 3 # `curveLocations` analysis may need 3 extra bits. For example, `groupZulu << 3`.
_bitWidthOffsetCurveLocationsSafety: int = 1 # I don't have mathematical proof of how many extra bits I need.
_bitWidthOffsetCurveLocations: int = _bitWidthOffsetCurveLocationsNecessary + _bitWidthOffsetCurveLocationsSafety

bitWidthCurveLocationsMaximum: int = _bitWidthOfFixedSizeInteger - _bitWidthOffsetCurveLocations

del _bitWidthOffsetCurveLocationsNecessary, _bitWidthOffsetCurveLocationsSafety, _bitWidthOffsetCurveLocations

_bitWidthOffsetDistinctCrossingsNecessary: int = 0 # I don't know of any.
_bitWidthOffsetDistinctCrossingsEstimation: int = 3 # See reference directory.
_bitWidthOffsetDistinctCrossingsSafety: int = 3
_bitWidthOffsetDistinctCrossings: int = _bitWidthOffsetDistinctCrossingsNecessary + _bitWidthOffsetDistinctCrossingsEstimation + _bitWidthOffsetDistinctCrossingsSafety

bitWidthDistinctCrossingsMaximum: int = _bitWidthOfFixedSizeInteger - _bitWidthOffsetDistinctCrossings

del _bitWidthOffsetDistinctCrossingsNecessary, _bitWidthOffsetDistinctCrossingsEstimation, _bitWidthOffsetDistinctCrossingsSafety, _bitWidthOffsetDistinctCrossings
del _bitWidthOfFixedSizeInteger

datatypeCurveLocations = numpy.uint64
datatypeDistinctCrossings = numpy.uint64

# ----------------- environment configuration: NumPy ------------------------------------------------------------------

datatypeCurveLocationsNumPy = numpy.uint64
type DataArray1D = numpy.ndarray[tuple[int, ...], numpy.dtype[datatypeCurveLocationsNumPy | numpy.signedinteger[Any]]]
type DataArray2columns = numpy.ndarray[tuple[int, ...], numpy.dtype[datatypeCurveLocationsNumPy]]
type DataArray3columns = numpy.ndarray[tuple[int, ...], numpy.dtype[datatypeCurveLocationsNumPy]]
type SelectorBoolean = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]]
type SelectorIndices = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]]

# NOTE This code blocks enables semantic references to your NumPy arrays.
columnsArrayCurveGroups = columnsTotal = 3
columnΩ: int = (columnsTotal - columnsTotal) - 1
columnDistinctCrossings = columnΩ = columnΩ + 1
columnGroupAlpha = columnΩ = columnΩ + 1
columnGroupZulu = columnΩ = columnΩ + 1
if columnΩ != columnsTotal - 1:
	message = f"Please inspect the code above this `if` check. '{columnsTotal = }', therefore '{columnΩ = }' must be '{columnsTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del columnsTotal, columnΩ

columnsArrayCurveLocations = columnsTotal = 2
columnΩ: int = (columnsTotal - columnsTotal) - 1
columnDistinctCrossings = columnΩ = columnΩ + 1
columnCurveLocations = columnΩ = columnΩ + 1
if columnΩ != columnsTotal - 1:
	message = f"Please inspect the code above this `if` check. '{columnsTotal = }', therefore '{columnΩ = }' must be '{columnsTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del columnsTotal, columnΩ

locatorGroupAlphaNumPy: int = 0x5555555555555555
locatorGroupZuluNumPy: int = 0xaaaaaaaaaaaaaaaa

# ----------------- support functions: NumPy only ---------------------------------------------------------------------

def aggregateBridgesSimple2CurveLocations(arrayCurveLocations: DataArray2columns, indexStart: int, curveLocations: DataArray1D, distinctCrossings: DataArray1D) -> int:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`."""
	miniCurveLocations, indices = numpy.unique_inverse(curveLocations[numpy.flatnonzero(curveLocations)])

	indexStop: int = indexStart + int(miniCurveLocations.size)
	arrayCurveLocations[indexStart:indexStop, columnCurveLocations] = miniCurveLocations

	miniCurveLocations = None; del miniCurveLocations  # noqa: E702
	goByeBye()

	numpy.add.at(arrayCurveLocations[indexStart:indexStop, columnDistinctCrossings], indices, distinctCrossings[numpy.flatnonzero(curveLocations)])

	return indexStop

def aggregateCurveLocations2CurveGroups(arrayCurveLocations: DataArray2columns) -> DataArray3columns:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`; create curve groups."""
	curveLocations, indices = numpy.unique_inverse(arrayCurveLocations[:, columnCurveLocations])
	arrayCurveGroups: DataArray3columns = numpy.zeros((len(curveLocations), columnsArrayCurveGroups), dtype=datatypeCurveLocationsNumPy)
	numpy.bitwise_and(curveLocations, locatorGroupAlphaNumPy, out=arrayCurveGroups[:, columnGroupAlpha])
	numpy.bitwise_and(curveLocations, locatorGroupZuluNumPy, out=arrayCurveGroups[:, columnGroupZulu])

	curveLocations = None; del curveLocations  # noqa: E702
	goByeBye()

	arrayCurveGroups[:, columnGroupZulu] >>= 1
	numpy.add.at(arrayCurveGroups[:, columnDistinctCrossings], indices, arrayCurveLocations[:, columnDistinctCrossings])
	return arrayCurveGroups

def aggregateData2CurveLocations(arrayCurveLocations: DataArray2columns, indexStart: int, curveLocations: DataArray1D, distinctCrossings: DataArray1D, selector: SelectorBoolean, limiter: int) -> int:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`."""
	miniCurveLocations, indices = numpy.unique_inverse(curveLocations[numpy.flatnonzero(curveLocations < limiter)])

	indexStop: int = indexStart + int(miniCurveLocations.size)
	arrayCurveLocations[indexStart:indexStop, columnCurveLocations] = miniCurveLocations

	miniCurveLocations = None; del miniCurveLocations  # noqa: E702
	goByeBye()

	numpy.add.at(arrayCurveLocations[indexStart:indexStop, columnDistinctCrossings], indices, distinctCrossings[numpy.flatnonzero(selector)[numpy.flatnonzero(curveLocations < limiter)]])

	return indexStop

def convertDictionaryCurveLocations2array(dictionaryCurveLocations: dict[int, int]) -> DataArray2columns:
	return numpy.column_stack((numpy.fromiter(dictionaryCurveLocations.values(), dtype=datatypeDistinctCrossings), numpy.fromiter(dictionaryCurveLocations.keys(), dtype=datatypeCurveLocationsNumPy)))

# ----------------- support functions ---------------------------------------------------------------------------------

@cache
def flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

# Probably NumPy
flipTheExtra_0b1AsUfunc = numpy.frompyfunc(flipTheExtra_0b1, 1, 1)
# Probably Pandas
flipTheExtra_0b1vectorized = numpy.vectorize(flipTheExtra_0b1, otypes=[datatypeCurveLocations])

def getLocatorGroupAlpha(bitWidth: int) -> int:
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
	return sum(1 << one for one in range(0, bitWidth, 2))

def getLocatorGroupZulu(bitWidth: int) -> int:
	"""Compute an even-parity bit-mask with `bitWidth` bits."""
	return sum(1 << one for one in range(1, bitWidth, 2))

def getMAXIMUMcurveLocations(indexTransferMatrix: int) -> int:
	return 1 << (2 * indexTransferMatrix + 4)

def outfitDictionaryCurveGroups(dictionaryCurveLocations: dict[int, int]) -> dict[tuple[int, int], int]:
	bitWidth: int = max(dictionaryCurveLocations.keys()).bit_length()
	locatorGroupAlpha: int = getLocatorGroupAlpha(bitWidth)
	locatorGroupZulu: int = getLocatorGroupZulu(bitWidth)
	return {(curveLocations & locatorGroupAlpha, (curveLocations & locatorGroupZulu) >> 1): distinctCrossings
		for curveLocations, distinctCrossings in dictionaryCurveLocations.items()}

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
	"""NOTE `gc.set_threshold`: Low numbers nullify the `walkDyckPath` cache."""
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

# ----------------- counting functions --------------------------------------------------------------------------------

def countBigInt(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int], indexTransferMatrixMinimum: int = 0) -> tuple[int, dict[int, int]]:
	dictionaryCurveGroups: dict[tuple[int, int], int] = {}

	gc.disable()  # Disable the garbage collector inside this loop to maximize the `walkDyckPath` cache hits.

# TODO `listIndicesTransferMatrix` exists because ... I thought it would help with something in `tqdm`. But I don't remember. However, the unrelated issue of `sys.stdout.write(f"Switching at {indexTransferMatrix =} .")` might cause changes that will make `listIndicesTransferMatrix` very useful, so wait and see.
	listIndicesTransferMatrix: list[int] = list(range(indexTransferMatrix -1, indexTransferMatrixMinimum -1, -1))
	for indexTransferMatrix in tqdm(listIndicesTransferMatrix, leave=False):  # noqa: PLR1704
		if (indexTransferMatrixMinimum > 0) and (max(dictionaryCurveLocations.keys()).bit_length() <= bitWidthCurveLocationsMaximum):
			indexTransferMatrix += 1  # noqa: PLW2901
			sys.stdout.write(f"Switching at {indexTransferMatrix =} .")
			break
		MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(indexTransferMatrix)
		dictionaryCurveGroups = outfitDictionaryCurveGroups(dictionaryCurveLocations)
		dictionaryCurveLocations = {}

		for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
			groupAlphaCurves: bool = groupAlpha > 1
			groupZuluCurves: bool = groupZulu > 1
			groupAlphaIsEven = groupZuluIsEven = 0

			# simple
			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			if curveLocationAnalysis < MAXIMUMcurveLocations:
				dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 1)) << 1)
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupZuluCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			# aligned
			if groupAlphaCurves and groupZuluCurves and (groupAlphaIsEven or groupZuluIsEven):
				if groupAlphaIsEven and not groupZuluIsEven:
					groupAlpha ^= walkDyckPath(groupAlpha)  # noqa: PLW2901
				elif groupZuluIsEven and not groupAlphaIsEven:
					groupZulu ^= walkDyckPath(groupZulu)  # noqa: PLW2901

				curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

	gc.enable()  # Re-enable the garbage collector.

	return (indexTransferMatrix, dictionaryCurveLocations)

def countNumPy(indexTransferMatrix: int, arrayCurveLocations: DataArray2columns, indexTransferMatrixMinimum: int = 0) -> tuple[int, DataArray2columns]:
	listIndicesTransferMatrix: list[int] = list(range(indexTransferMatrix -1, indexTransferMatrixMinimum -1, -1))
	for indexTransferMatrix in listIndicesTransferMatrix:  # noqa: PLR1704
		if int(arrayCurveLocations[:, columnDistinctCrossings].max()).bit_length() > bitWidthDistinctCrossingsMaximum:
			indexTransferMatrix += 1  # noqa: PLW2901
			sys.stdout.write(f"Switching at {indexTransferMatrix =} .")
			break

		MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(indexTransferMatrix)
		arrayCurveGroups = aggregateCurveLocations2CurveGroups(arrayCurveLocations)

		arrayCurveLocations = None; del arrayCurveLocations # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

		set_threshold(1, 1, 1)  # Re-enable the garbage collector.

		allocateGroupAlphaCurves: int = (arrayCurveGroups[:, columnGroupAlpha] > datatypeCurveLocationsNumPy(1)).sum()
		allocateGroupZuluCurves: int = (arrayCurveGroups[:, columnGroupZulu] > datatypeCurveLocationsNumPy(1)).sum()

		selectBridgesAligned: SelectorBoolean = numpy.empty_like(arrayCurveGroups[:, columnGroupAlpha], dtype=bool)
		numpy.equal(numpy.bitwise_and(arrayCurveGroups[:, columnGroupAlpha], 1), 0, out=selectBridgesAligned, dtype=bool)
		numpy.bitwise_or(selectBridgesAligned, (numpy.equal(numpy.bitwise_and(arrayCurveGroups[:, columnGroupZulu], 1), 0, dtype=bool)), out=selectBridgesAligned)
		numpy.bitwise_and(selectBridgesAligned, (arrayCurveGroups[:, columnGroupAlpha] > datatypeCurveLocationsNumPy(1)), out=selectBridgesAligned)
		numpy.bitwise_and(selectBridgesAligned, (arrayCurveGroups[:, columnGroupZulu] > datatypeCurveLocationsNumPy(1)), out=selectBridgesAligned)

		allocateBridgesAligned: int = int(numpy.count_nonzero(selectBridgesAligned))

# ----------------------------------------------- bridgesSimple -------------------------------------------------------
		curveLocationsBridgesSimpleLessThanMaximum: DataArray1D = arrayCurveGroups[:, columnGroupZulu].copy()
		numpy.left_shift(curveLocationsBridgesSimpleLessThanMaximum, 1, out=curveLocationsBridgesSimpleLessThanMaximum)
		numpy.bitwise_or(curveLocationsBridgesSimpleLessThanMaximum, arrayCurveGroups[:, columnGroupAlpha], out=curveLocationsBridgesSimpleLessThanMaximum)
		numpy.left_shift(curveLocationsBridgesSimpleLessThanMaximum, 2, out=curveLocationsBridgesSimpleLessThanMaximum)
		numpy.bitwise_or(curveLocationsBridgesSimpleLessThanMaximum, 3, out=curveLocationsBridgesSimpleLessThanMaximum)
		curveLocationsBridgesSimpleLessThanMaximum[curveLocationsBridgesSimpleLessThanMaximum >= MAXIMUMcurveLocations] = 0

		allocateBridgesSimple: int = int(numpy.count_nonzero(curveLocationsBridgesSimpleLessThanMaximum))

# ----------------------------------------------- arrayCurveLocations -------------------------------------------------
		rowsAllocatedTotal: int = allocateGroupAlphaCurves + allocateGroupZuluCurves + allocateBridgesSimple + allocateBridgesAligned
		arrayCurveLocations = numpy.zeros((rowsAllocatedTotal, columnsArrayCurveLocations), dtype=datatypeCurveLocationsNumPy)

		rowsAggregatedTotal: int = 0
		rowsDeallocatedTotal: int = 0

# ----------------------------------------------- bridgesSimple -------------------------------------------------------
		rowsAggregatedTotal = aggregateBridgesSimple2CurveLocations(arrayCurveLocations
			, rowsAggregatedTotal
			, curveLocationsBridgesSimpleLessThanMaximum
			, arrayCurveGroups[:, columnDistinctCrossings]
		)

		rowsDeallocatedTotal += allocateBridgesSimple
		arrayCurveLocations.resize((((rowsAllocatedTotal - rowsDeallocatedTotal) + rowsAggregatedTotal), columnsArrayCurveLocations))

		curveLocationsBridgesSimpleLessThanMaximum = None; del curveLocationsBridgesSimpleLessThanMaximum  # pyright: ignore[reportAssignmentType] # noqa: E702
		del allocateBridgesSimple
		goByeBye()

# ----------------------------------------------- groupAlpha ----------------------------------------------------------
		selectGroupAlphaCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupAlpha] > datatypeCurveLocationsNumPy(1)
		curveLocationsGroupAlpha: DataArray1D = arrayCurveGroups[selectGroupAlphaCurves, columnGroupAlpha].copy()

		numpy.bitwise_and(curveLocationsGroupAlpha, 1, out=curveLocationsGroupAlpha)
		numpy.subtract(datatypeCurveLocationsNumPy(1), curveLocationsGroupAlpha, out=curveLocationsGroupAlpha)
		numpy.left_shift(curveLocationsGroupAlpha, 3, out=curveLocationsGroupAlpha)
		numpy.bitwise_or(curveLocationsGroupAlpha, arrayCurveGroups[selectGroupAlphaCurves, columnGroupAlpha], out=curveLocationsGroupAlpha)
		numpy.right_shift(curveLocationsGroupAlpha, 2, out=curveLocationsGroupAlpha)
# NOTE (groupAlpha >> 2) | (groupZulu << 3) | ((1 - (groupAlpha & 1)) << 1)
		arrayLockbox: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]] = numpy.full_like(a=curveLocationsGroupAlpha, fill_value=numpy.uint8(0b111), dtype=numpy.uint8)
		numpy.bitwise_and(arrayLockbox, curveLocationsGroupAlpha, out=arrayLockbox, dtype=numpy.uint8)
		numpy.right_shift(curveLocationsGroupAlpha, 3, out=curveLocationsGroupAlpha)
		numpy.bitwise_or(curveLocationsGroupAlpha, arrayCurveGroups[selectGroupAlphaCurves, columnGroupZulu], out=curveLocationsGroupAlpha)
		numpy.left_shift(curveLocationsGroupAlpha, 3, out=curveLocationsGroupAlpha)
		numpy.bitwise_or(curveLocationsGroupAlpha, arrayLockbox, out=curveLocationsGroupAlpha)

		arrayLockbox = None; del arrayLockbox  # pyright: ignore[reportAssignmentType] # noqa: E702
		goByeBye()

		rowsAggregatedTotal = aggregateData2CurveLocations(arrayCurveLocations
			, rowsAggregatedTotal
			, curveLocationsGroupAlpha
			, arrayCurveGroups[:, columnDistinctCrossings]
			, selectGroupAlphaCurves
			, MAXIMUMcurveLocations
		)

		rowsDeallocatedTotal += allocateGroupAlphaCurves
		arrayCurveLocations.resize((((rowsAllocatedTotal - rowsDeallocatedTotal) + rowsAggregatedTotal), columnsArrayCurveLocations))

		curveLocationsGroupAlpha = None; del curveLocationsGroupAlpha  # pyright: ignore[reportAssignmentType] # noqa: E702
		del allocateGroupAlphaCurves
		selectGroupAlphaCurves = None; del selectGroupAlphaCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

# ----------------------------------------------- groupZulu -----------------------------------------------------------
		selectGroupZuluCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupZulu] > datatypeCurveLocationsNumPy(1)
		curveLocationsGroupZulu: DataArray1D = arrayCurveGroups[selectGroupZuluCurves, columnGroupAlpha].copy()
		numpy.left_shift(curveLocationsGroupZulu, 2, out=curveLocationsGroupZulu)
# NOTE (groupAlpha << 2)

		numpy.bitwise_or(curveLocationsGroupZulu, numpy.subtract(datatypeCurveLocationsNumPy(1), numpy.bitwise_and(arrayCurveGroups[selectGroupZuluCurves, columnGroupZulu], 1)), out=curveLocationsGroupZulu)

# NOTE | (groupZulu >> 1)
		numpy.left_shift(curveLocationsGroupZulu, 1, out=curveLocationsGroupZulu)
		numpy.bitwise_or(curveLocationsGroupZulu, arrayCurveGroups[selectGroupZuluCurves, columnGroupZulu], out=curveLocationsGroupZulu)
		numpy.right_shift(curveLocationsGroupZulu, 1, out=curveLocationsGroupZulu)

		rowsAggregatedTotal = aggregateData2CurveLocations(arrayCurveLocations
			, rowsAggregatedTotal
			, curveLocationsGroupZulu
			, arrayCurveGroups[:, columnDistinctCrossings]
			, selectGroupZuluCurves
			, MAXIMUMcurveLocations
		)

		rowsDeallocatedTotal += allocateGroupZuluCurves
		arrayCurveLocations.resize((((rowsAllocatedTotal - rowsDeallocatedTotal) + rowsAggregatedTotal), columnsArrayCurveLocations))

		curveLocationsGroupZulu = None; del curveLocationsGroupZulu  # pyright: ignore[reportAssignmentType] # noqa: E702
		del allocateGroupZuluCurves
		selectGroupZuluCurves = None; del selectGroupZuluCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

# ----------------------------------------------- bridgesAligned ------------------------------------------------------
# `bridgesAligned` = `bridgesGroupAlphaPairedToOdd` UNION WITH `bridgesGroupZuluPairedToOdd` UNION WITH `bridgesAlignedAtEven`

# bridgesAligned -------------------------------- bridgesGroupAlphaPairedToOdd ----------------------------------------
# NOTE this code block MODIFIES `arrayCurveGroups` NOTE
		set_threshold(0, 0, 0)  # Disable the garbage collector inside this loop to maximize the `walkDyckPath` cache hits.

		selectGroupAlphaAtEven: SelectorBoolean = numpy.empty_like(arrayCurveGroups[:, columnGroupAlpha], dtype=bool)
		numpy.equal(numpy.bitwise_and(arrayCurveGroups[:, columnGroupAlpha], 1), 0, out=selectGroupAlphaAtEven, dtype=bool)

		selectGroupZuluAtEven: SelectorBoolean = numpy.empty_like(arrayCurveGroups[:, columnGroupZulu], dtype=bool)
		numpy.equal(numpy.bitwise_and(arrayCurveGroups[:, columnGroupZulu], 1), 0, out=selectGroupZuluAtEven, dtype=bool)

		selectBridgesGroupAlphaPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & selectGroupAlphaAtEven & (~selectGroupZuluAtEven))
		arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha] = flipTheExtra_0b1AsUfunc(arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha])
# Without changing `flipTheExtra_0b1`, above works, but `out=` does not. Why? Elephino.
# NOTE flipTheExtra_0b1(arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha], casting='unsafe', out=arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha])

		selectBridgesGroupAlphaPairedToOdd = None; del selectBridgesGroupAlphaPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702

# bridgesAligned -------------------------------- bridgesGroupZuluPairedToOdd ------------------------------------------
# NOTE this code block MODIFIES `arrayCurveGroups` NOTE
		set_threshold(0, 0, 0)  # Disable the garbage collector inside this loop to maximize the `walkDyckPath` cache hits.
		selectBridgesGroupZuluPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & (~selectGroupAlphaAtEven) & selectGroupZuluAtEven)
		arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, columnGroupZulu] = flipTheExtra_0b1AsUfunc(arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, columnGroupZulu])

		set_threshold(1, 1, 1)  # Re-enable the garbage collector.
		selectBridgesGroupZuluPairedToOdd = None; del selectBridgesGroupZuluPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaAtEven = None; del selectGroupAlphaAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluAtEven = None; del selectGroupZuluAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

# NOTE: All computations for `bridgesAlignedAtEven` are handled by the computations for `bridgesAligned`.

# ----------------------------------------------- bridgesAligned ------------------------------------------------------

		curveLocationsBridgesAlignedLessThanMaximum: DataArray1D = numpy.zeros((selectBridgesAligned.sum(),), dtype=datatypeCurveLocationsNumPy)
		numpy.right_shift(arrayCurveGroups[selectBridgesAligned, columnGroupZulu], 2, out=curveLocationsBridgesAlignedLessThanMaximum)
		numpy.left_shift(curveLocationsBridgesAlignedLessThanMaximum, 3, out=curveLocationsBridgesAlignedLessThanMaximum)
		numpy.bitwise_or(curveLocationsBridgesAlignedLessThanMaximum, arrayCurveGroups[selectBridgesAligned, columnGroupAlpha], out=curveLocationsBridgesAlignedLessThanMaximum)
		numpy.right_shift(curveLocationsBridgesAlignedLessThanMaximum, 2, out=curveLocationsBridgesAlignedLessThanMaximum)
		curveLocationsBridgesAlignedLessThanMaximum[curveLocationsBridgesAlignedLessThanMaximum >= MAXIMUMcurveLocations] = 0

		Z0Z_indexStart: int = rowsAggregatedTotal
		rowsAggregatedTotal += int(numpy.count_nonzero(curveLocationsBridgesAlignedLessThanMaximum))

		arrayCurveLocations[Z0Z_indexStart:rowsAggregatedTotal, columnCurveLocations] = curveLocationsBridgesAlignedLessThanMaximum[numpy.flatnonzero(curveLocationsBridgesAlignedLessThanMaximum)]
		arrayCurveLocations[Z0Z_indexStart:rowsAggregatedTotal, columnDistinctCrossings] = arrayCurveGroups[(numpy.flatnonzero(selectBridgesAligned)[numpy.flatnonzero(curveLocationsBridgesAlignedLessThanMaximum)]), columnDistinctCrossings]

		rowsDeallocatedTotal += allocateBridgesAligned
		arrayCurveLocations.resize((((rowsAllocatedTotal - rowsDeallocatedTotal) + rowsAggregatedTotal), columnsArrayCurveLocations))

		arrayCurveGroups = None; del arrayCurveGroups # pyright: ignore[reportAssignmentType]  # noqa: E702
		curveLocationsBridgesAlignedLessThanMaximum = None; del curveLocationsBridgesAlignedLessThanMaximum  # pyright: ignore[reportAssignmentType] # noqa: E702
		del allocateBridgesAligned
		del MAXIMUMcurveLocations
		del rowsAllocatedTotal
		del rowsDeallocatedTotal
		del Z0Z_indexStart
		del rowsAggregatedTotal
		selectBridgesAligned = None; del selectBridgesAligned  # pyright: ignore[reportAssignmentType] # noqa: E702
		goByeBye()

	return (indexTransferMatrix, arrayCurveLocations)

def countPandas(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int], indexTransferMatrixMinimum: int = 0) -> tuple[int, dict[Any, Any]]:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame."""
	def aggregateCurveLocations(MAXIMUMcurveLocations: int) -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = 0
		dataframeAnalyzed = pandas.concat(
			[dataframeAnalyzed
			, dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0)].groupby('analyzed')['distinctCrossings'].aggregate('sum').reset_index()
		], ignore_index=True)
		dataframeAnalyzed = dataframeAnalyzed.groupby('analyzed')['distinctCrossings'].aggregate('sum').reset_index()

		dataframeCurveLocations.loc[:, 'analyzed'] = 0

	def analyzeCurveLocationsAligned(MAXIMUMcurveLocations: int) -> None:
		"""Compute `curveLocations` from `groupAlpha` and `groupZulu` if at least one is an even number.

		Before computing `curveLocations`, some values of `groupAlpha` and `groupZulu` are modified.

		Formula
		-------
		```python
		if groupAlpha > 1 and groupZulu > 1 and (groupAlphaIsEven or groupZuluIsEven):
			curveLocations = (groupAlpha >> 2) | ((groupZulu >> 2) << 1)
		```

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.
		"""
		nonlocal dataframeCurveLocations

		# if groupAlphaIsEven or groupZuluIsEven
		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations[dataframeCurveLocations['alignAt'] == 'oddBoth'].index)

		# if groupAlphaCurves and groupZuluCurves
		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations[(dataframeCurveLocations['groupAlpha'] <= 1) | (dataframeCurveLocations['groupZulu'] <= 1)].index)

		# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
		dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenAlpha', 'groupAlpha'] = flipTheExtra_0b1vectorized(dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenAlpha', 'groupAlpha'])

		# if groupZuluIsEven and not groupAlphaIsEven, modifyGroupZuluPairedToOdd
		dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenZulu', 'groupZulu'] = flipTheExtra_0b1vectorized(dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenZulu', 'groupZulu'])

		dataframeCurveLocations.loc[:, 'groupAlpha'] //= 2**2 # (groupAlpha >> 2)
		dataframeCurveLocations.loc[:, 'groupZulu'] //= 2**2 # (groupZulu >> 2)
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**1 # ((groupZulu ...) << 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupAlpha'] | dataframeCurveLocations['groupZulu'] # (groupZulu ...) | (groupAlpha ...)

		aggregateCurveLocations(MAXIMUMcurveLocations)

	def analyzeCurveLocationsAlpha(MAXIMUMcurveLocations: int) -> None:
		"""Compute `curveLocations` from `groupAlpha`.

		Formula
		-------
		```python
		if groupAlpha > 1:
			curveLocations = ((1 - (groupAlpha & 1)) << 1) | (groupZulu << 3) | (groupAlpha >> 2)
		```

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupAlpha'] & 1 # (groupAlpha & 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupAlpha ...))

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] == 1, 'alignAt'] = 'evenAlpha' # groupAlphaIsEven

		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**1 # ((groupAlpha ...) << 1)
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**3 # (groupZulu << 3)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # ... | (groupZulu ...)
		dataframeCurveLocations.loc[:, 'analyzed'] |= (dataframeCurveLocations['groupAlpha'] // 2**2) # ... | (groupAlpha >> 2)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupAlpha'] <= 1, 'analyzed'] = 0 # if groupAlpha > 1

		aggregateCurveLocations(MAXIMUMcurveLocations)
		computeCurveGroups(alpha=False)

	def analyzeCurveLocationsSimple(MAXIMUMcurveLocations: int) -> None:
		"""Compute curveLocations with the 'simple' bridges formula.

		Formula
		-------
		```python
		curveLocations = ((groupAlpha | (groupZulu << 1)) << 2) | 3
		```

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.

		Notes
		-----
		About substituting `+= 3` for `|= 3`:

		- Givens
			1. "n" is a Python `int` >= 0
			2. "0bk" = `bin(n)`

		- Claims
			1. n * 2**2 == n << 2
			2. bin(n * 2**2) == 0bk00
			3. 0b11 = 0b00 | 0b11
			4. 0bk11 = 0bk00 | 0b11
			5. 0b11 = 0bk11 - 0bk00
			6. 0b11 == int(3)

		- Therefore
			- For any non-zero integer, 0bk00, the operation 0bk00 | 0b11 is equivalent to 0bk00 + 0b11.
			- I hope my substitution is valid!

		Why substitute? I've been having problems implementing bitwise operations in pandas, so I am avoiding them until I learn
		how to implement them in pandas.
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['groupAlpha']
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**1 # (groupZulu << 1)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # ((groupAlpha | (groupZulu ...))
		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # (... << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] += 3 # (...) | 3

		aggregateCurveLocations(MAXIMUMcurveLocations)
		computeCurveGroups(alpha=False)

	def analyzeCurveLocationsZulu(MAXIMUMcurveLocations: int) -> None:
		"""Compute `curveLocations` from `groupZulu`.

		Formula
		-------
		```python
		if groupZulu > 1:
			curveLocations = (1 - (groupZulu & 1)) | (groupAlpha << 2) | (groupZulu >> 1)
		```

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupZulu'] & 1 # (groupZulu & 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupZulu ...))

		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1) & (dataframeCurveLocations['alignAt'] == 'evenAlpha'), 'alignAt'] = 'evenBoth' # groupZuluIsEven
		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1) & (dataframeCurveLocations['alignAt'] == 'oddBoth'), 'alignAt'] = 'evenZulu' # groupZuluIsEven

		dataframeCurveLocations.loc[:, 'groupAlpha'] *= 2**2 # (groupAlpha << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupAlpha'] # ... | (groupAlpha ...)
# NOTE potential memory optimization
		dataframeCurveLocations.loc[:, 'analyzed'] |= (dataframeCurveLocations['groupZulu'] // 2**1) # ... | (groupZulu >> 1)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupZulu'] <= 1, 'analyzed'] = 0 # if groupZulu > 1

		aggregateCurveLocations(MAXIMUMcurveLocations)
		computeCurveGroups(zulu=False)

	def computeCurveGroups(*, alpha: bool = True, zulu: bool = True) -> None:
		"""Compute `groupAlpha` and `groupZulu` with 'bit-masks' on `curveLocations`.

		Parameters
		----------
		alpha : bool = True
			Should column `groupAlpha` be computed?

		zulu : bool = True
			Should column `groupZulu` be computed?

		3L33T H@X0R
		-----------
		- `groupAlpha`: odd-parity bit-masked `curveLocations`
		- `groupZulu`: even-parity bit-masked `curveLocations`
		"""
		nonlocal dataframeCurveLocations
		bitWidth: int = int(dataframeCurveLocations['curveLocations'].max()).bit_length()
# TODO scratch this itch: I _feel_ it must be possible to bifurcate `curvesLocations` with one formula. Even if implementation is infeasible, I want to know.
		if alpha:
			dataframeCurveLocations['groupAlpha'] = dataframeCurveLocations['curveLocations']
			dataframeCurveLocations.loc[:, 'groupAlpha'] &= getLocatorGroupAlpha(bitWidth)
		if zulu:
			dataframeCurveLocations['groupZulu'] = dataframeCurveLocations['curveLocations']
			dataframeCurveLocations.loc[:, 'groupZulu'] &= getLocatorGroupZulu(bitWidth)
			dataframeCurveLocations.loc[:, 'groupZulu'] //= 2**1 # (groupZulu >> 1)

	def outfitDataframeCurveLocations() -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations = dataframeCurveLocations.iloc[0:0]
		dataframeCurveLocations['curveLocations'] = dataframeAnalyzed['analyzed']
		dataframeCurveLocations['distinctCrossings'] = dataframeAnalyzed['distinctCrossings']
		dataframeCurveLocations['alignAt'] = 'oddBoth'
		dataframeCurveLocations['analyzed'] = 0
		dataframeAnalyzed = dataframeAnalyzed.iloc[0:0]
		computeCurveGroups()
		goByeBye()

	CategoriesAlignAt = pandas.CategoricalDtype(categories=['evenAlpha', 'evenZulu', 'evenBoth', 'oddBoth'], ordered=False)

	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=list(dictionaryCurveLocations.keys()), dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=list(dictionaryCurveLocations.values()), dtype=datatypeDistinctCrossings)
		}
	)

	dataframeCurveLocations = pandas.DataFrame({
		'curveLocations': pandas.Series(name='curveLocations', data=0, dtype=datatypeCurveLocations)
		, 'groupAlpha': pandas.Series(name='groupAlpha', data=0, dtype=datatypeCurveLocations)
		, 'groupZulu': pandas.Series(name='groupZulu', data=0, dtype=datatypeCurveLocations)
		, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=0, dtype=datatypeDistinctCrossings)
		, 'alignAt': pandas.Series(name='alignAt', data='oddBoth', dtype=CategoriesAlignAt)
		}
	)

	listIndicesTransferMatrix: list[int] = list(range(indexTransferMatrix -1, indexTransferMatrixMinimum -1, -1))
	for indexTransferMatrix in tqdm(listIndicesTransferMatrix, leave=False):  # noqa: PLR1704
		if int(dataframeAnalyzed['distinctCrossings'].max()).bit_length() > bitWidthDistinctCrossingsMaximum:
			indexTransferMatrix += 1  # noqa: PLW2901
			sys.stdout.write(f"Switching at {indexTransferMatrix =} .")
			break
		MAXIMUMcurveLocations: int = 1 << (2 * indexTransferMatrix + 4)
		outfitDataframeCurveLocations()

		analyzeCurveLocationsSimple(MAXIMUMcurveLocations)
		analyzeCurveLocationsAlpha(MAXIMUMcurveLocations)
		analyzeCurveLocationsZulu(MAXIMUMcurveLocations)
		analyzeCurveLocationsAligned(MAXIMUMcurveLocations)

	return (indexTransferMatrix, dataframeAnalyzed.set_index('analyzed')['distinctCrossings'].to_dict())

# ----------------- doTheNeedful --------------------------------------------------------------------------------------

def doTheNeedful(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int]) -> int:
	"""Compute a(n) meanders with the transfer matrix algorithm.

	Parameters
	----------
	n : int
		The index in the OEIS ID sequence.
	dictionaryCurveLocations : dict[int, int]
		A dictionary mapping curve locations to their counts.

	Returns
	-------
	a(n) : int
		The computed value of a(n).

	Making sausage
	--------------

	As first computed by Iwan Jensen in 2000, A000682(41) = 6664356253639465480.
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex
	See also https://oeis.org/A000682

	I'm sure you instantly observed that A000682(41) = (6664356253639465480).bit_length() = 63 bits. And A005316(44) =
	(18276178714484582264).bit_length() = 64 bits.

	If you ask NumPy 2.3, "What is your relationship with integers with more than 64 bits?"
	NumPy will say, "It's complicated."

	Therefore, to take advantage of the computational excellence of NumPy when computing A000682(n) for n > 41, I must make some
	adjustments at the total count approaches 64 bits.

	The second complication is bit-packed integers. I use a loop that starts at `bridges = n` and decrements (`bridges -= 1`)
	`until bridges = 0`. If `bridges > 29`, some of the bit-packed integers have more than 64 bits. "Hey NumPy, can I use
	bit-packed integers with more than 64 bits?" NumPy: "It's complicated." Therefore, while `bridges` is decrementing, I don't
	use NumPy until I believe the bit-packed integers will be less than 64 bits.

	A third factor that works in my favor is that peak memory usage occurs when all types of integers are well under 64-bits wide.

	In total, to compute a(n) for "large" n, I use three-stages.
	1. I use Python primitive `int` contained in a Python primitive `dict`.
	2. When the bit width of the bit-packed integers connected to `bridges` is small enough to use `numpy.uint64`, I switch to NumPy for the heavy lifting.
	3. When `distinctCrossings` subtotals might exceed 64 bits, I must switch back to Python primitives.
	"""
	bitWidth: int = max(dictionaryCurveLocations.keys()).bit_length()

	indexTransferMatrixMinimum = max(0, (indexTransferMatrix + 1 - 41 + 2))

	# NOTE Stage 1 if `curveLocations` bit-width is too wide for `numpy.uint64`.
	if bitWidth > bitWidthCurveLocationsMaximum:
		indexTransferMatrixMinimumEstimatedA000682 = 28
		indexTransferMatrix, dictionaryCurveLocations = countBigInt(indexTransferMatrix, dictionaryCurveLocations, indexTransferMatrixMinimumEstimatedA000682)
		goByeBye()

	# NOTE Stage 2 Goldilocks

	# NumPy
	# indexTransferMatrix, arrayCurveLocations = countNumPy(indexTransferMatrix, convertDictionaryCurveLocations2array(dictionaryCurveLocations))  # noqa: ERA001

	# Pandas
# TODO memory crash A000682(45) with pandas, which is the same limit as NumPy. The good news: I have put zero effort into optimizing memory usage with pandas.
# Even better, I disabled the garbage collector, but never re-enabled it.

	indexTransferMatrix, dictionaryCurveLocations = countPandas(indexTransferMatrix, dictionaryCurveLocations, indexTransferMatrixMinimum)

	# NOTE Stage 3 if `distinctCrossings` bit-width is too wide for `numpy.uint64`.
	if indexTransferMatrix > 0:
		goByeBye()
		indexTransferMatrix, dictionaryCurveLocations = countBigInt(indexTransferMatrix, dictionaryCurveLocations)
	return sum(dictionaryCurveLocations.values())
"""	NumPy Stage 3
		distinctCrossingsTotal: int = sum(dictionaryCurveLocations.values())
	else:
		distinctCrossingsTotal = int(arrayCurveLocations[:, columnDistinctCrossings].sum())
	return distinctCrossingsTotal

"""
@cache
def A000682(n: int) -> int:
	"""Compute A000682(n)."""
	if n & 0b1:
		curveLocations: int = 5
	else:
		curveLocations = 1
	listCurveLocations: list[int] = [(curveLocations << 1) | curveLocations]

	MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(n-1)
	while listCurveLocations[-1] < MAXIMUMcurveLocations:
		curveLocations = (curveLocations << 4) | 0b101 # == curveLocations * 2**4 + 5
		listCurveLocations.append((curveLocations << 1) | curveLocations)
	return doTheNeedful(n - 1, dict.fromkeys(listCurveLocations, 1))

@cache
def A005316(n: int) -> int:
	"""Compute A005316(n)."""
	if n & 0b1:
		dictionaryCurveLocations: dict[int, int] = {15: 1}
	else:
		dictionaryCurveLocations = {22: 1}
	return doTheNeedful(n - 1, dictionaryCurveLocations)
