# ruff: noqa: D100 D103
from functools import cache
from gc import collect as goByeBye, set_threshold
from typing import Any, Literal
import gc
import numpy

# DEVELOPMENT INSTRUCTIONS FOR THIS MODULE
#
# Avoid early-return guard clauses, short-circuit returns, and multiple exit points. This codebase enforces a
# single-return-per-function pattern with stable shapes/dtypes due to AST transforms. An empty input is a problem, so allow it to
# fail early.
#
# If an algorithm has potential for infinite loops, fix the root cause: do NOT add artificial safety limits (e.g., maxIterations
# counters) to prevent infinite loops.
#
# Always use semantic column, index, or slice identifiers: Never hardcode the locations.

# TODO `set_threshold`: I know 0 means disabled, but I don't even understand if 1 means "as frequently as possible" or "almost never".
set_threshold(1, 1, 1)
Z0Z_bit_lengthSafetyLimit: int = 61

type DataArray1D = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64 | numpy.signedinteger[Any]]]
type DataArray2columns = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]
type DataArray3columns = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]
type SelectorBoolean = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]]
type SelectorIndices = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]]

# NOTE This code blocks enables semantic references to your data.
columnsArrayCurveGroups = columnsArrayTotal = 3
columnΩ: int = (columnsArrayTotal - columnsArrayTotal) - 1  # Something _feels_ right about this instead of `= -1`.
columnDistinctCrossings = columnΩ = columnΩ + 1
columnGroupAlpha = columnΩ = columnΩ + 1
columnGroupZulu = columnΩ = columnΩ + 1
if columnΩ != columnsArrayTotal - 1:
	message = f"Please inspect the code above this `if` check. '{columnsArrayTotal = }', therefore '{columnΩ = }' must be '{columnsArrayTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del columnsArrayTotal, columnΩ

columnsArrayCurveLocations = columnsArrayTotal = 2
columnΩ: int = (columnsArrayTotal - columnsArrayTotal) - 1
columnDistinctCrossings = columnΩ = columnΩ + 1
columnCurveLocations = columnΩ = columnΩ + 1
if columnΩ != columnsArrayTotal - 1:
	message = f"Please inspect the code above this `if` check. '{columnsArrayTotal = }', therefore '{columnΩ = }' must be '{columnsArrayTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del columnsArrayTotal, columnΩ

groupAlphaLocator: int = 0x55555555555555555555555555555555
groupAlphaLocator64: int = 0x5555555555555555
groupZuluLocator: int = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
groupZuluLocator64: int = 0xaaaaaaaaaaaaaaaa

def convertDictionaryCurveLocations2CurveGroups(dictionaryCurveLocations: dict[int, int]) -> dict[tuple[int, int], int]:
	return {(curveLocations & groupAlphaLocator, (curveLocations & groupZuluLocator) >> 1): distinctCrossings
		for curveLocations, distinctCrossings in dictionaryCurveLocations.items()}

def count(bridges: int, dictionaryCurveGroups: dict[tuple[int, int], int], bridgesMinimum: int = 0) -> tuple[int, dict[tuple[int, int], int]]:

	dictionaryCurveLocations: dict[int, int] = {}
	while bridges > bridgesMinimum:
		bridges -= 1

		curveLocationsMAXIMUM: int = 1 << (2 * bridges + 4)

		for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
			groupAlphaCurves = groupAlpha != 1
			groupZuluCurves = groupZulu != 1

			# bridgesSimple
			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			if curveLocationAnalysis < curveLocationsMAXIMUM:
				dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 0b1)) << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupZuluCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			# bridgesAligned
			if groupZuluCurves and groupAlphaCurves:
				# One Truth-check to select a code path
				groupsCanBePairedTogether = (groupZuluIsEven << 1) | groupAlphaIsEven # pyright: ignore[reportPossiblyUnboundVariable]

				if groupsCanBePairedTogether != 0:  # Case 0 (False, False)
					XOrHere2makePair = 0b1
					findUnpaired_0b1 = 0

					if groupsCanBePairedTogether == 1:  # Case 1: (False, True)
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (groupAlpha & XOrHere2makePair) == 0 else -1
						groupAlpha ^= XOrHere2makePair  # noqa: PLW2901
					elif groupsCanBePairedTogether == 2:  # Case 2: (True, False)
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (groupZulu & XOrHere2makePair) == 0 else -1
						groupZulu ^= XOrHere2makePair  # noqa: PLW2901

					# Cases 1, 2, and 3 all compute curveLocationAnalysis
					curveLocationAnalysis = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

		dictionaryCurveGroups = convertDictionaryCurveLocations2CurveGroups(dictionaryCurveLocations)
		dictionaryCurveLocations = {}

	return (bridges, dictionaryCurveGroups)

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
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

@cache
def _flipTheExtra_0b1(avoidingLookupsInPerRowLoop: int) -> numpy.uint64:
	"""Be a docstring."""
	return numpy.uint64(avoidingLookupsInPerRowLoop ^ walkDyckPath(avoidingLookupsInPerRowLoop))

# TODO there is a better way to do this.
flipTheExtra_0b1 = numpy.vectorize(_flipTheExtra_0b1, otypes=[numpy.uint64])
"""The vectorize function is provided primarily for convenience, not for performance. The implementation is essentially a for loop."""

def aggregateCurveLocations(arrayCurveLocations: DataArray2columns) -> DataArray3columns:
	arrayCurveGroups: DataArray3columns = numpy.tile(
		A=numpy.unique(arrayCurveLocations[:, columnCurveLocations])
		, reps=(columnsArrayCurveGroups, 1)
		).T
	arrayCurveGroups[:, columnDistinctCrossings] = 0
	numpy.add.at(
		arrayCurveGroups[:, columnDistinctCrossings]
		, numpy.searchsorted(
			a=arrayCurveGroups[:, columnCurveLocations]
			, v=arrayCurveLocations[:, columnCurveLocations])
		, arrayCurveLocations[:, columnDistinctCrossings]
	)
	# I'm computing groupZulu from curveLocations that are physically in `arrayCurveGroups`, so I'm using `columnCurveLocations`.
	numpy.bitwise_and(arrayCurveGroups[:, columnCurveLocations], numpy.uint64(groupZuluLocator64), out=arrayCurveGroups[:, columnGroupZulu])
	numpy.right_shift(arrayCurveGroups[:, columnGroupZulu], 1, out=arrayCurveGroups[:, columnGroupZulu])
	# NOTE Do not alphabetize these operations. This column has curveLocations data that groupZulu needs.
	arrayCurveGroups[:, columnGroupAlpha] &= groupAlphaLocator64
	return arrayCurveGroups

def convertDictionaryCurveGroups2array(dictionaryCurveGroups: dict[tuple[int, int], int]) -> DataArray3columns:
	arrayCurveGroups: DataArray3columns = numpy.tile(numpy.fromiter(dictionaryCurveGroups.values(), dtype=numpy.uint64), (columnsArrayCurveGroups, 1)).T
	arrayKeys: DataArray2columns = numpy.array(list(dictionaryCurveGroups.keys()), dtype=numpy.uint64)
	arrayCurveGroups[:, columnGroupAlpha] = arrayKeys[:, 0]
	arrayCurveGroups[:, columnGroupZulu] = arrayKeys[:, 1]
	return arrayCurveGroups

def count64(bridges: int, arrayCurveGroups: DataArray3columns, bridgesMinimum: int = 0) -> tuple[int, DataArray3columns]:

	while bridges > bridgesMinimum and int(arrayCurveGroups[:, columnDistinctCrossings].max()).bit_length() < Z0Z_bit_lengthSafetyLimit:
		bridges -= 1
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1 << (2 * bridges + 4))

		selectGroupAlphaCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupAlpha] > numpy.uint64(1)
		curveLocationsGroupAlpha: DataArray1D = ((arrayCurveGroups[selectGroupAlphaCurves, columnGroupAlpha] >> 2)
			| (arrayCurveGroups[selectGroupAlphaCurves, columnGroupZulu] << 3)
			| ((numpy.uint64(1) - (arrayCurveGroups[selectGroupAlphaCurves, columnGroupAlpha] & 1)) << 1)
		)
		selectGroupAlphaCurvesLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectGroupAlphaCurves)[numpy.flatnonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)]

		selectGroupZuluCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupZulu] > numpy.uint64(1)
		curveLocationsGroupZulu: DataArray1D = (arrayCurveGroups[selectGroupZuluCurves, columnGroupZulu] >> 1
			| arrayCurveGroups[selectGroupZuluCurves, columnGroupAlpha] << 2
			| (numpy.uint64(1) - (arrayCurveGroups[selectGroupZuluCurves, columnGroupZulu] & 1))
		)
		selectGroupZuluCurvesLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectGroupZuluCurves)[numpy.flatnonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)]

		selectBridgesSimpleLessThanMaximum: SelectorIndices = numpy.flatnonzero(
			((arrayCurveGroups[:, columnGroupAlpha] << 2) | (arrayCurveGroups[:, columnGroupZulu] << 3) | 3) < curveLocationsMAXIMUM
		) # Computation, but including `< curveLocationsMAXIMUM` is ~2% of total time.

		# Selectors for bridgesAligned -------------------------------------------------
		selectGroupAlphaAtEven: SelectorBoolean = (arrayCurveGroups[:, columnGroupAlpha] & 1) == numpy.uint64(0)
		selectGroupZuluAtEven: SelectorBoolean = (arrayCurveGroups[:, columnGroupZulu] & 1) == numpy.uint64(0)
		selectBridgesAligned: SelectorBoolean = selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven | selectGroupZuluAtEven)

		SliceΩ: slice[int, int, Literal[1]] = slice(0,0)
		sliceAllocateGroupAlpha = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectGroupAlphaCurvesLessThanMaximum.size)
		sliceAllocateGroupZulu = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectGroupZuluCurvesLessThanMaximum.size)
		sliceAllocateBridgesSimple = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectBridgesSimpleLessThanMaximum.size)
		sliceAllocateBridgesAligned = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectBridgesAligned.size)

		arrayCurveLocations: DataArray2columns = numpy.zeros((SliceΩ.stop, columnsArrayCurveLocations), dtype=arrayCurveGroups.dtype)

		arrayCurveLocations[sliceAllocateGroupAlpha, columnCurveLocations] = curveLocationsGroupAlpha[numpy.flatnonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)]
		arrayCurveLocations[sliceAllocateGroupAlpha, columnDistinctCrossings] = arrayCurveGroups[selectGroupAlphaCurvesLessThanMaximum, columnDistinctCrossings]

		arrayCurveLocations[sliceAllocateGroupZulu, columnCurveLocations] = curveLocationsGroupZulu[numpy.flatnonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)]
		arrayCurveLocations[sliceAllocateGroupZulu, columnDistinctCrossings] = arrayCurveGroups[selectGroupZuluCurvesLessThanMaximum, columnDistinctCrossings]

# TODO Uh, it sure looks like I am doing this computation twice. Computation (without assignment) ~ 1.5% of total time.
		arrayCurveLocations[sliceAllocateBridgesSimple, columnCurveLocations] = (
			(arrayCurveGroups[selectBridgesSimpleLessThanMaximum, columnGroupAlpha] << 2)
			| (arrayCurveGroups[selectBridgesSimpleLessThanMaximum, columnGroupZulu] << 3)
			| 3
		)
		arrayCurveLocations[sliceAllocateBridgesSimple, columnDistinctCrossings] = arrayCurveGroups[selectBridgesSimpleLessThanMaximum, columnDistinctCrossings]

		curveLocationsGroupAlpha = None; del curveLocationsGroupAlpha  # pyright: ignore[reportAssignmentType] # noqa: E702
		curveLocationsGroupZulu = None; del curveLocationsGroupZulu  # pyright: ignore[reportAssignmentType] # noqa: E702
		selectBridgesSimpleLessThanMaximum = None; del selectBridgesSimpleLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaCurvesLessThanMaximum = None; del selectGroupAlphaCurvesLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluCurvesLessThanMaximum = None; del selectGroupZuluCurvesLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

		# NOTE this MODIFIES `arrayCurveGroups` for bridgesPairedToOdd ---------------------------------------------------------------------------------------
		selectBridgesGroupAlphaPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & selectGroupAlphaAtEven & (~selectGroupZuluAtEven))
		arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha] = flipTheExtra_0b1(
			arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha]
		)

		selectBridgesGroupZuluPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & (~selectGroupAlphaAtEven) & selectGroupZuluAtEven)
		arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, columnGroupZulu] = flipTheExtra_0b1(
			arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, columnGroupZulu]
		)

		selectBridgesGroupAlphaPairedToOdd = None; del selectBridgesGroupAlphaPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectBridgesGroupZuluPairedToOdd = None; del selectBridgesGroupZuluPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaAtEven = None; del selectGroupAlphaAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaCurves = None; del selectGroupAlphaCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluAtEven = None; del selectGroupZuluAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluCurves = None; del selectGroupZuluCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

		# bridgesAligned; bridgesAlignedAtEven, bridgesGroupAlphaPairedToOdd, bridgesGroupZuluPairedToOdd ------------------------------------------------------------------
		curveLocationsBridgesAligned: DataArray1D = (((arrayCurveGroups[selectBridgesAligned, columnGroupZulu] >> 2) << 1)
			| (arrayCurveGroups[selectBridgesAligned, columnGroupAlpha] >> 2)
		)
		selectBridgesAlignedLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectBridgesAligned)[numpy.flatnonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)]

		sliceAllocateBridgesAligned = SliceΩ  = slice(sliceAllocateBridgesAligned.start, sliceAllocateBridgesAligned.stop - selectBridgesAligned.size + selectBridgesAlignedLessThanMaximum.size)
		arrayCurveLocations[sliceAllocateBridgesAligned, columnDistinctCrossings] = arrayCurveGroups[selectBridgesAlignedLessThanMaximum, columnDistinctCrossings]
		arrayCurveLocations[sliceAllocateBridgesAligned, columnCurveLocations] = curveLocationsBridgesAligned[numpy.flatnonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)]

		arrayCurveGroups = None; del arrayCurveGroups # pyright: ignore[reportAssignmentType]  # noqa: E702
		curveLocationsBridgesAligned = None; del curveLocationsBridgesAligned  # pyright: ignore[reportAssignmentType] # noqa: E702
		del curveLocationsMAXIMUM
		selectBridgesAligned = None; del selectBridgesAligned  # pyright: ignore[reportAssignmentType] # noqa: E702
		selectBridgesAlignedLessThanMaximum = None; del selectBridgesAlignedLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

		arrayCurveLocations.resize((SliceΩ.stop, columnsArrayCurveLocations))
		arrayCurveGroups = aggregateCurveLocations(arrayCurveLocations)

		arrayCurveLocations = None; del arrayCurveLocations # pyright: ignore[reportAssignmentType]  # noqa: E702
		del sliceAllocateBridgesAligned
		del sliceAllocateBridgesSimple
		del sliceAllocateGroupAlpha
		del sliceAllocateGroupZulu
		del SliceΩ
		goByeBye()

	return (bridges, arrayCurveGroups)

def convertArrayCurveGroups2dictionaryCurveGroups(arrayCurveGroups: DataArray3columns) -> dict[tuple[int, int], int]:
	return {(int(row[columnGroupAlpha]), int(row[columnGroupZulu])): int(row[columnDistinctCrossings]) for row in arrayCurveGroups}

def doTheNeedful(n: int, dictionaryCurveLocations: dict[int, int]) -> int:
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
# NOTE '29' is based on two things. 1) `bridges = 29`, groupZuluLocator = 0xaaaaaaaaaaaaaaaa.bit_length() = 64. 2) If `bridges =
# 30` or a larger number, `OverflowError: int too big to convert`. Conclusion: '29' isn't necessarily correct or the best value:
# it merely fits within my limited ability to assess the correct value.
# NOTE the above was written when I had the `bridges >= bridgesMinimum` bug. So, apply '-1' to everything.
# NOTE This default value is necessary: it prevents `count64` from returning an incomplete dictionary when that is not necessary.
# TODO `count64_bridgesMaximum` might be a VERY good idea as a second safeguard against overflowing distinctCrossingsTotal. But
# I'm pretty sure I should use an actual check on maximum bit-width in arrayCurveGroups[:, columnDistinctCrossings] at the start
# of each while loop. Tests on A000682 showed that the max bit-width of arrayCurveGroups[:, columnDistinctCrossings] always
# increased by 1 or 2 bits on each iteration: never 0 and never 3. I did not test A005316. And I do not have a mathematical proof of the limit.

	count64_bridgesMaximum = 28
	bridgesMinimum = 0
	distinctCrossings64bitLimitAsValueOf_n = 41
	distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG = distinctCrossings64bitLimitAsValueOf_n - 3
	distinctCrossings64bitLimitSafetyMargin = 4

	dictionaryCurveGroups: dict[tuple[int, int], int] = convertDictionaryCurveLocations2CurveGroups(dictionaryCurveLocations)

	if n >= count64_bridgesMaximum:
		if n >= distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG:
			bridgesMinimum = n - distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG + distinctCrossings64bitLimitSafetyMargin
		n, dictionaryCurveGroups = count(n, dictionaryCurveGroups, count64_bridgesMaximum)
		gc.collect()
	n, arrayCurveGroups = count64(n, convertDictionaryCurveGroups2array(dictionaryCurveGroups), bridgesMinimum)
	if n > 0:
		gc.collect()
		n, dictionaryCurveGroups = count(n, convertArrayCurveGroups2dictionaryCurveGroups(arrayCurveGroups), bridgesMinimum=0)
		distinctCrossingsTotal = sum(dictionaryCurveGroups.values())
	else:
		distinctCrossingsTotal = int(arrayCurveGroups[0, columnDistinctCrossings])
	return distinctCrossingsTotal
