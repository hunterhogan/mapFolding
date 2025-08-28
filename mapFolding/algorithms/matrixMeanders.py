# ruff: noqa: D100 D103
from functools import cache
from gc import collect as goByeBye
from tqdm import tqdm
from typing import Any
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

# TODO `set_threshold`: Low numbers nullify the `walkDyckPath` cache. Can I use this more effectively than merely disabled or 1,1,1?
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

def count(bridges: int, dictionaryCurveGroups: dict[tuple[int, int], int], bridgesMinimum: int = 0) -> tuple[int, dict[tuple[int, int], int]]:

	dictionaryCurveLocations: dict[int, int] = {}
	with tqdm(total=bridges, initial=bridges) as tqdmBar:
		while bridges > bridgesMinimum:
			bridges -= 1

			curveLocationsMAXIMUM: int = 1 << (2 * bridges + 4)

			for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
				groupAlphaCurves: bool = groupAlpha != 1
				groupZuluCurves: bool = groupZulu != 1
				groupAlphaIsEven = groupZuluIsEven = 0

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
				if groupAlphaCurves and groupZuluCurves and (groupAlphaIsEven or groupZuluIsEven):
					if groupAlphaIsEven and not groupZuluIsEven:
						groupAlpha ^= walkDyckPath(groupAlpha)  # noqa: PLW2901
					elif groupZuluIsEven and not groupAlphaIsEven:
						groupZulu ^= walkDyckPath(groupZulu)  # noqa: PLW2901

					curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			dictionaryCurveGroups = convertDictionaryCurveLocations2CurveGroups(dictionaryCurveLocations)
			dictionaryCurveLocations = {}
			tqdmBar.update(-1)
	return (bridges, dictionaryCurveGroups)

@cache
def _flipTheExtra_0b1(avoidingLookupsInPerRowLoop: int) -> numpy.uint64:
	"""Be a docstring."""
	return numpy.uint64(avoidingLookupsInPerRowLoop ^ walkDyckPath(avoidingLookupsInPerRowLoop))

# TODO there is a better way to do this. If I decorate `_flipTheExtra_0b1`, I think that nullifies the cache advantage, which is huge.
# But, I think I can use more parameters in `numpy.vectorize` for better results and/or documentation.
# OR, ideally, since this is faux-vectorization, I can find real vectorization.
# Nevertheless, while this section of the code is more expensive than the other three selectors, it is trivial compared to the aggregation problems.
flipTheExtra_0b1 = numpy.vectorize(_flipTheExtra_0b1, otypes=[numpy.uint64])
"""NumPy docs: The vectorize function is provided primarily for convenience, not for performance. The implementation is essentially a for loop."""

def aggregateCurveLocations2CurveGroups(arrayCurveLocations: DataArray2columns) -> DataArray3columns:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`; create curve groups."""
	curveLocations, indices = numpy.unique_inverse(arrayCurveLocations[:, columnCurveLocations])
	arrayCurveGroups: DataArray3columns = numpy.zeros((len(curveLocations), columnsArrayCurveGroups), dtype=numpy.uint64)
	numpy.bitwise_and(curveLocations, groupAlphaLocator64, out=arrayCurveGroups[:, columnGroupAlpha])
	numpy.right_shift(numpy.bitwise_and(curveLocations, groupZuluLocator64), 1, out=arrayCurveGroups[:, columnGroupZulu])

	del curveLocations
	goByeBye()

	numpy.add.at(arrayCurveGroups[:, columnDistinctCrossings], indices, arrayCurveLocations[:, columnDistinctCrossings])
	return arrayCurveGroups

def aggregateColumns2CurveLocations(arrayCurveLocations: DataArray2columns, indexStart: int, curveLocations: DataArray1D, distinctCrossings: DataArray1D) -> int:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`."""
	miniCurveLocations, indices = numpy.unique_inverse(curveLocations)

	indexStop: int = indexStart + int(miniCurveLocations.size)
	arrayCurveLocations[indexStart:indexStop, columnCurveLocations] = miniCurveLocations
	numpy.add.at(arrayCurveLocations[indexStart:indexStop, columnDistinctCrossings], indices, distinctCrossings)

	return indexStop

def aggregateData2CurveLocations(arrayCurveLocations: DataArray2columns, indexStart: int, curveLocations: DataArray1D, distinctCrossings: DataArray1D) -> int:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`."""
	miniCurveLocations, indices = numpy.unique_inverse(curveLocations)

	indexStop: int = indexStart + int(miniCurveLocations.size)
	arrayCurveLocations[indexStart:indexStop, columnCurveLocations] = miniCurveLocations
	numpy.add.at(arrayCurveLocations[indexStart:indexStop, columnDistinctCrossings], indices, distinctCrossings)

	return indexStop

def convertDictionaryCurveGroups2array(dictionaryCurveGroups: dict[tuple[int, int], int]) -> DataArray3columns:
	arrayCurveGroups: DataArray3columns = numpy.tile(numpy.fromiter(dictionaryCurveGroups.values(), dtype=numpy.uint64), (columnsArrayCurveGroups, 1)).T
	arrayKeys: DataArray2columns = numpy.array(list(dictionaryCurveGroups.keys()), dtype=numpy.uint64)
	arrayCurveGroups[:, columnGroupAlpha] = arrayKeys[:, 0]
	arrayCurveGroups[:, columnGroupZulu] = arrayKeys[:, 1]
	return arrayCurveGroups

def count64(bridges: int, arrayCurveGroups: DataArray3columns, bridgesMinimum: int = 0) -> tuple[int, DataArray3columns]:
	with tqdm(total=bridges, initial=bridges) as tqdmBar:
		while bridges > bridgesMinimum and int(arrayCurveGroups[:, columnDistinctCrossings].max()).bit_length() < Z0Z_bit_lengthSafetyLimit:
			bridges -= 1
			curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1 << (2 * bridges + 4))

# ----------------------------------------------- groupAlpha ----------------------------------------------------------
			selectGroupAlphaCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupAlpha] > numpy.uint64(1)
			curveLocationsGroupAlpha: DataArray1D = ((arrayCurveGroups[selectGroupAlphaCurves, columnGroupAlpha] >> 2)
				| (arrayCurveGroups[selectGroupAlphaCurves, columnGroupZulu] << 3)
				| ((numpy.uint64(1) - (arrayCurveGroups[selectGroupAlphaCurves, columnGroupAlpha] & 1)) << 1)
			)
			selectGroupAlphaCurvesLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectGroupAlphaCurves)[numpy.flatnonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)]
			allocateGroupAlphaCurves: int = selectGroupAlphaCurvesLessThanMaximum.size

# ----------------------------------------------- groupZulu -----------------------------------------------------------
			selectGroupZuluCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupZulu] > numpy.uint64(1)
			curveLocationsGroupZulu: DataArray1D = (arrayCurveGroups[selectGroupZuluCurves, columnGroupZulu] >> 1
				| arrayCurveGroups[selectGroupZuluCurves, columnGroupAlpha] << 2
				| (numpy.uint64(1) - (arrayCurveGroups[selectGroupZuluCurves, columnGroupZulu] & 1))
			)
			selectGroupZuluCurvesLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectGroupZuluCurves)[numpy.flatnonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)]
			allocateGroupZuluCurves: int = selectGroupZuluCurvesLessThanMaximum.size

# ----------------------------------------------- bridgesSimple -------------------------------------------------------
			curveLocationsBridgesSimple: DataArray1D = ((arrayCurveGroups[:, columnGroupAlpha] << 2) | (arrayCurveGroups[:, columnGroupZulu] << 3) | 3)
			selectBridgesSimpleLessThanMaximum: SelectorIndices = numpy.flatnonzero(curveLocationsBridgesSimple < curveLocationsMAXIMUM)
			allocateBridgesSimple: int = selectBridgesSimpleLessThanMaximum.size

# ----------------------------------------------- bridgesAligned ------------------------------------------------------
			selectGroupAlphaAtEven: SelectorBoolean = (arrayCurveGroups[:, columnGroupAlpha] & 1) == numpy.uint64(0)
			selectGroupZuluAtEven: SelectorBoolean = (arrayCurveGroups[:, columnGroupZulu] & 1) == numpy.uint64(0)
			selectBridgesAligned: SelectorBoolean = selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven | selectGroupZuluAtEven)
			allocateBridgesAligned: int = int(numpy.count_nonzero(selectBridgesAligned))

# ----------------------------------------------- arrayCurveLocations -------------------------------------------------
			rowsAllocatedTotal: int = allocateGroupAlphaCurves + allocateGroupZuluCurves + allocateBridgesSimple + allocateBridgesAligned
			arrayCurveLocations: DataArray2columns = numpy.zeros((rowsAllocatedTotal, columnsArrayCurveLocations), dtype=arrayCurveGroups.dtype)

			Z0Z_indexStop: int = 0

# ----------------------------------------------- groupAlpha ----------------------------------------------------------
			Z0Z_indexStop = aggregateColumns2CurveLocations(arrayCurveLocations
				, Z0Z_indexStop
				, curveLocationsGroupAlpha[numpy.flatnonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)]
				, arrayCurveGroups[selectGroupAlphaCurvesLessThanMaximum, columnDistinctCrossings]
			)
			rowsAllocatedTotal += Z0Z_indexStop
			rowsAllocatedTotal -= allocateGroupAlphaCurves
			arrayCurveLocations.resize((rowsAllocatedTotal, columnsArrayCurveLocations))

			curveLocationsGroupAlpha = None; del curveLocationsGroupAlpha  # pyright: ignore[reportAssignmentType] # noqa: E702
			selectGroupAlphaCurves = None; del selectGroupAlphaCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
			selectGroupAlphaCurvesLessThanMaximum = None; del selectGroupAlphaCurvesLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

# ----------------------------------------------- groupZulu -----------------------------------------------------------
			Z0Z_indexStop = aggregateColumns2CurveLocations(arrayCurveLocations
				, Z0Z_indexStop
				, curveLocationsGroupZulu[numpy.flatnonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)]
				, arrayCurveGroups[selectGroupZuluCurvesLessThanMaximum, columnDistinctCrossings]
			)

			rowsAllocatedTotal += Z0Z_indexStop
			rowsAllocatedTotal -= allocateGroupAlphaCurves
			rowsAllocatedTotal -= allocateGroupZuluCurves
			arrayCurveLocations.resize((rowsAllocatedTotal, columnsArrayCurveLocations))

			curveLocationsGroupZulu = None; del curveLocationsGroupZulu  # pyright: ignore[reportAssignmentType] # noqa: E702
			selectGroupZuluCurves = None; del selectGroupZuluCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
			selectGroupZuluCurvesLessThanMaximum = None; del selectGroupZuluCurvesLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

# ----------------------------------------------- bridgesSimple -------------------------------------------------------
			Z0Z_indexStop = aggregateColumns2CurveLocations(arrayCurveLocations
				, Z0Z_indexStop
				, curveLocationsBridgesSimple[numpy.flatnonzero(curveLocationsBridgesSimple < curveLocationsMAXIMUM)]
				, arrayCurveGroups[selectBridgesSimpleLessThanMaximum, columnDistinctCrossings]
			)
			rowsAllocatedTotal += Z0Z_indexStop
			rowsAllocatedTotal -= allocateGroupAlphaCurves
			rowsAllocatedTotal -= allocateGroupZuluCurves
			rowsAllocatedTotal -= allocateBridgesSimple
			arrayCurveLocations.resize((rowsAllocatedTotal, columnsArrayCurveLocations))

			curveLocationsBridgesSimple = None; del curveLocationsBridgesSimple  # pyright: ignore[reportAssignmentType] # noqa: E702
			selectBridgesSimpleLessThanMaximum = None; del selectBridgesSimpleLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

# ----------------------------------------------- bridgesAligned ------------------------------------------------------
# `bridgesAligned` = `bridgesGroupAlphaPairedToOdd` UNION WITH `bridgesGroupZuluPairedToOdd` UNION WITH `bridgesAlignedAtEven`

# bridgesAligned -------------------------------- bridgesGroupAlphaPairedToOdd ----------------------------------------
# NOTE this code block MODIFIES `arrayCurveGroups` NOTE
			selectBridgesGroupAlphaPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & selectGroupAlphaAtEven & (~selectGroupZuluAtEven))
			arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha] = flipTheExtra_0b1(arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha])
			selectBridgesGroupAlphaPairedToOdd = None; del selectBridgesGroupAlphaPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702

# bridgesAligned -------------------------------- bridgesGroupZuluPairedToOdd ------------------------------------------
# NOTE this code block MODIFIES `arrayCurveGroups` NOTE
			selectBridgesGroupZuluPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & (~selectGroupAlphaAtEven) & selectGroupZuluAtEven)
			arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, columnGroupZulu] = flipTheExtra_0b1(arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, columnGroupZulu])
			selectBridgesGroupZuluPairedToOdd = None; del selectBridgesGroupZuluPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702

			selectGroupAlphaAtEven = None; del selectGroupAlphaAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
			selectGroupZuluAtEven = None; del selectGroupZuluAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

# NOTE: All computations for `bridgesAlignedAtEven` are handled by the computations for `bridgesAligned`.

# ----------------------------------------------- bridgesAligned ------------------------------------------------------
			curveLocationsBridgesAligned: DataArray1D = (((arrayCurveGroups[selectBridgesAligned, columnGroupZulu] >> 2) << 1)
				| (arrayCurveGroups[selectBridgesAligned, columnGroupAlpha] >> 2)
			)
			selectBridgesAlignedLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectBridgesAligned)[numpy.flatnonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)]
			selectBridgesAligned = None; del selectBridgesAligned  # pyright: ignore[reportAssignmentType] # noqa: E702

			Z0Z_indexStop = aggregateColumns2CurveLocations(arrayCurveLocations
				, Z0Z_indexStop
				, curveLocationsBridgesAligned[numpy.flatnonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)]
				, arrayCurveGroups[selectBridgesAlignedLessThanMaximum, columnDistinctCrossings]
			)
			rowsAllocatedTotal += Z0Z_indexStop
			rowsAllocatedTotal -= allocateGroupAlphaCurves
			rowsAllocatedTotal -= allocateGroupZuluCurves
			rowsAllocatedTotal -= allocateBridgesSimple
			rowsAllocatedTotal -= allocateBridgesAligned
			arrayCurveLocations.resize((rowsAllocatedTotal, columnsArrayCurveLocations))

			arrayCurveGroups = None; del arrayCurveGroups # pyright: ignore[reportAssignmentType]  # noqa: E702
			curveLocationsBridgesAligned = None; del curveLocationsBridgesAligned  # pyright: ignore[reportAssignmentType] # noqa: E702
			del curveLocationsMAXIMUM
			selectBridgesAlignedLessThanMaximum = None; del selectBridgesAlignedLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

# ----------------------------------------------- aggregation ---------------------------------------------------------
			arrayCurveGroups = aggregateCurveLocations2CurveGroups(arrayCurveLocations)

			arrayCurveLocations = None; del arrayCurveLocations # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

			tqdmBar.update(-1)

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
# NOTE `count64_bridgesMaximum = 28` is based on empirical evidence. I do not have a mathematical proof that it is correct.
	count64_bridgesMaximum = 28
	bridgesMinimum = 0

# TODO Setting `bridgesMinimum` > 0 in `count64` might be a VERY good idea as a second safeguard against overflowing
# distinctCrossingsTotal. As a the first safeguard I've added an actual check on maximum bit-width in arrayCurveGroups[:,
# columnDistinctCrossings] at the start of each while loop. The computational cost seems to be too low to measure. `count`, when
# `bridges` have decremented to < ~8 is very fast. So the real issue is avoiding overflow during peak memory usage. Tests on
# A000682 showed that the max bit-width of arrayCurveGroups[:, columnDistinctCrossings] always increased by 1 or 2 bits on each
# iteration: never 0 and never 3. I did not test A005316. And I do not have a mathematical proof of the limit. And I prefer less complex code.
# So, for `count64`, I might just set `bridgesMinimum = 0` and I check the bit-width at the start of each while loop. That would allow me to remove
# `distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG` code. Testing this, however, requires large values of n, which take a long time to compute.
	distinctCrossings64bitLimitAsValueOf_n = 41
	distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG = distinctCrossings64bitLimitAsValueOf_n - 3
	distinctCrossings64bitLimitSafetyMargin = 4

	dictionaryCurveGroups: dict[tuple[int, int], int] = convertDictionaryCurveLocations2CurveGroups(dictionaryCurveLocations)

	if n >= count64_bridgesMaximum:
		if n >= distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG:
			bridgesMinimum: int = n - distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG + distinctCrossings64bitLimitSafetyMargin

		n, dictionaryCurveGroups = count(n, dictionaryCurveGroups, count64_bridgesMaximum)
		goByeBye()
	n, arrayCurveGroups = count64(n, convertDictionaryCurveGroups2array(dictionaryCurveGroups), bridgesMinimum)
	if n > 0:
		goByeBye()

		n, dictionaryCurveGroups = count(n, convertArrayCurveGroups2dictionaryCurveGroups(arrayCurveGroups))
		distinctCrossingsTotal: int = sum(dictionaryCurveGroups.values())
	else:
		distinctCrossingsTotal = int(arrayCurveGroups[0, columnDistinctCrossings])
	return distinctCrossingsTotal

def A000682getCurveLocations(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)

	curveStart: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveStart << 1) | curveStart]

	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveStart = (curveStart << 4) | 0b101
		listCurveLocations.append((curveStart << 1) | curveStart)

	return dict.fromkeys(listCurveLocations, 1)

@cache
def A000682(n: int) -> int:
	return doTheNeedful(n - 1, A000682getCurveLocations(n - 1))

def A005316getCurveLocations(n: int) -> dict[int, int]:
	if n & 0b1:
		return {22: 1}
	else:
		return {15: 1}

@cache
def A005316(n: int) -> int:
	return doTheNeedful(n - 1, A005316getCurveLocations(n - 1))
