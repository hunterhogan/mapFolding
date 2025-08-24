from functools import cache
from gc import collect as goByeBye, set_threshold
from mapFolding.algorithms.matrixMeandersSimple import count as countSimple
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
# Always use semantic index identifiers: Never hardcode the indices.

set_threshold(1, 1, 1)

def count(bridges: int, dictionaryCurveLocationsStarting: dict[int, int], bridgesMinimum: int = 0) -> tuple[int, dict[int, int]]:
	listCurveMaximums: list[tuple[int, int, int]] = [
(0x15, 0x2a, 0x10),
(0x55, 0xaa, 0x40),
(0x155, 0x2aa, 0x100),
(0x555, 0xaaa, 0x400),
(0x1555, 0x2aaa, 0x1000),
(0x5555, 0xaaaa, 0x4000),
(0x15555, 0x2aaaa, 0x10000),
(0x55555, 0xaaaaa, 0x40000),
(0x155555, 0x2aaaaa, 0x100000),
(0x555555, 0xaaaaaa, 0x400000),
(0x1555555, 0x2aaaaaa, 0x1000000),
(0x5555555, 0xaaaaaaa, 0x4000000),
(0x15555555, 0x2aaaaaaa, 0x10000000),
(0x55555555, 0xaaaaaaaa, 0x40000000), # `bridges = 13`, 0xaaaaaaaa.bit_length() = 32, 0x40000000.bit_length() = 31
(0x155555555, 0x2aaaaaaaa, 0x100000000),
(0x555555555, 0xaaaaaaaaa, 0x400000000),
(0x1555555555, 0x2aaaaaaaaa, 0x1000000000),
(0x5555555555, 0xaaaaaaaaaa, 0x4000000000),
(0x15555555555, 0x2aaaaaaaaaa, 0x10000000000),
(0x55555555555, 0xaaaaaaaaaaa, 0x40000000000),
(0x155555555555, 0x2aaaaaaaaaaa, 0x100000000000),
(0x555555555555, 0xaaaaaaaaaaaa, 0x400000000000),
(0x1555555555555, 0x2aaaaaaaaaaaa, 0x1000000000000),
(0x5555555555555, 0xaaaaaaaaaaaaa, 0x4000000000000),
(0x15555555555555, 0x2aaaaaaaaaaaaa, 0x10000000000000),
(0x55555555555555, 0xaaaaaaaaaaaaaa, 0x40000000000000),
(0x155555555555555, 0x2aaaaaaaaaaaaaa, 0x100000000000000),
(0x555555555555555, 0xaaaaaaaaaaaaaaa, 0x400000000000000),
(0x1555555555555555, 0x2aaaaaaaaaaaaaaa, 0x1000000000000000),
(0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x4000000000000000), # 0x5000000000000000.bit_length() = 63; 0xaaaaaaaaaaaaaaaa.bit_length() = 64; 0x5555555555555555.bit_length() = 63
(0x15555555555555555, 0x2aaaaaaaaaaaaaaaa, 0x10000000000000000),
(0x55555555555555555, 0xaaaaaaaaaaaaaaaaa, 0x40000000000000000),
(0x155555555555555555, 0x2aaaaaaaaaaaaaaaaa, 0x100000000000000000),
(0x555555555555555555, 0xaaaaaaaaaaaaaaaaaa, 0x400000000000000000),
(0x1555555555555555555, 0x2aaaaaaaaaaaaaaaaaa, 0x1000000000000000000),
(0x5555555555555555555, 0xaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000),
(0x15555555555555555555, 0x2aaaaaaaaaaaaaaaaaaa, 0x10000000000000000000),
(0x55555555555555555555, 0xaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000),
(0x155555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000),
(0x555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000),
(0x1555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000),
(0x5555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000),
(0x15555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000),
(0x55555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000),
(0x155555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000),
(0x555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000),
(0x1555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000),
(0x5555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000),
(0x15555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000),
(0x55555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000),
(0x155555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000000),
(0x555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000000),
(0x1555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000000),
(0x5555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000000),
(0x15555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000000),
(0x55555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000000),
(0x155555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000000000),
(0x555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000000000),
(0x1555555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000000000),
(0x5555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000000000),
(0x15555555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000000000),
(0x55555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000000000),
]
	"""`bridges = 29`
	0x5000000000000000.bit_length() = 63;
	0xaaaaaaaaaaaaaaaa.bit_length() = 64;
	0x5555555555555555.bit_length() = 63

	a(41) = 63 bits
	print((6664356253639465480).bit_length())
	"""

	listCurveMaximums = listCurveMaximums[0:bridges]

	dictionaryCurveLocationsAnalyzed: dict[int, int] = {}
	while bridges >= bridgesMinimum:
		bridges -= 1

		groupAlphaLocator, groupZuluLocator, curveLocationsMAXIMUM = listCurveMaximums[bridges]
		curveLocationsMAXIMUM: int = 1 << (2 * bridges + 4)
		groupAlphaLocator: int = 0x55555555555555555555555555555555
		groupZuluLocator: int = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

		for curveLocations, distinctCrossings in dictionaryCurveLocationsStarting.items():
			groupAlpha = (curveLocations & groupAlphaLocator)
			groupZulu = (curveLocations & groupZuluLocator) >> 1

			groupAlphaCurves = groupAlpha != 1
			groupZuluCurves = groupZulu != 1

			# bridgesSimple
			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			if curveLocationAnalysis < curveLocationsMAXIMUM:
				dictionaryCurveLocationsAnalyzed[curveLocationAnalysis] = dictionaryCurveLocationsAnalyzed.get(curveLocationAnalysis, 0) + distinctCrossings

			# bifurcationAlphaCurves
			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 0b1)) << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsAnalyzed[curveLocationAnalysis] = dictionaryCurveLocationsAnalyzed.get(curveLocationAnalysis, 0) + distinctCrossings

			# bifurcationZuluCurves
			if groupZuluCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsAnalyzed[curveLocationAnalysis] = dictionaryCurveLocationsAnalyzed.get(curveLocationAnalysis, 0) + distinctCrossings

			# Z0Z_alignedBridges
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
						groupAlpha ^= XOrHere2makePair
					elif groupsCanBePairedTogether == 2:  # Case 2: (True, False)
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (groupZulu & XOrHere2makePair) == 0 else -1
						groupZulu ^= XOrHere2makePair

					# Cases 1, 2, and 3 all compute curveLocationAnalysis
					curveLocationAnalysis = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocationsAnalyzed[curveLocationAnalysis] = dictionaryCurveLocationsAnalyzed.get(curveLocationAnalysis, 0) + distinctCrossings

		dictionaryCurveLocationsStarting.clear()
		dictionaryCurveLocationsStarting, dictionaryCurveLocationsAnalyzed = dictionaryCurveLocationsAnalyzed, dictionaryCurveLocationsStarting

	return (bridges, dictionaryCurveLocationsStarting)

type DataArray1D = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64 | numpy.signedinteger[Any]]]
type DataArray2D = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]
type DataArray3D = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]
type SelectorBoolean = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]]
type SelectorIndices = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]]

columnsArrayCurveGroups: int = 3
indexDistinctCrossings: int = 0
indexGroupAlpha: int = 1
indexGroupZulu: int = 2

columnsArrayCurveLocations: int = 2
indexDistinctCrossings: int = 0
indexCurveLocations: int = 1

groupAlphaLocator: int = 0x5555555555555555
groupZuluLocator: int = 0xaaaaaaaaaaaaaaaa

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
	return numpy.uint64(avoidingLookupsInPerRowLoop ^ walkDyckPath(avoidingLookupsInPerRowLoop))

flipTheExtra_0b1 = numpy.vectorize(_flipTheExtra_0b1, otypes=[numpy.uint64])
"""The vectorize function is provided primarily for convenience, not for performance. The implementation is essentially a for loop."""

def aggregateCurveLocations(arrayCurveLocations: DataArray2D) -> DataArray3D:
	arrayCurveGroups: DataArray3D = numpy.tile(numpy.unique(arrayCurveLocations[:, indexCurveLocations]), (columnsArrayCurveGroups, 1)).T
	arrayCurveGroups[:, indexDistinctCrossings] = 0
	numpy.add.at(arrayCurveGroups[:, indexDistinctCrossings], numpy.searchsorted(arrayCurveGroups[:, indexCurveLocations], arrayCurveLocations[:, indexCurveLocations]), arrayCurveLocations[:, indexDistinctCrossings])
	# I'm computing groupZulu from curveLocations that are physically in `arrayCurveGroups`, so I'm using `indexCurveLocations`.
	numpy.bitwise_and(arrayCurveGroups[:, indexCurveLocations], numpy.uint64(groupZuluLocator), out=arrayCurveGroups[:, indexGroupZulu])
	numpy.right_shift(arrayCurveGroups[:, indexGroupZulu], 1, out=arrayCurveGroups[:, indexGroupZulu])
	# NOTE Do not alphabetize these operations. This column has curveLocations data that groupZulu needs.
	arrayCurveGroups[:, indexGroupAlpha] &= groupAlphaLocator

	return arrayCurveGroups

def convertArrayCurveGroups2dictionary(arrayCurveGroups: DataArray3D) -> dict[int, int]:
	"""'Smush' `groupAlpha` and `groupZulu` back together into `curveLocations`."""
	arrayCurveGroups[:, indexCurveLocations] = arrayCurveGroups[:, indexGroupAlpha] | (arrayCurveGroups[:, indexGroupZulu] << 1)
	return {int(row[indexCurveLocations]): int(row[indexDistinctCrossings]) for row in arrayCurveGroups[:, [indexDistinctCrossings, indexCurveLocations]]}

def convertDictionaryCurveLocations2array(dictionaryCurveLocations: dict[int, int]) -> DataArray3D:
	arrayCurveGroups: DataArray3D = numpy.tile(numpy.fromiter(dictionaryCurveLocations.values(), dtype=numpy.uint64), (columnsArrayCurveGroups, 1)).T

	# I'm putting curveLocations into `arrayCurveGroups`, so I'm using `indexCurveLocations` even though it's not an index for this array.
	arrayCurveGroups[:, indexCurveLocations] = numpy.fromiter(dictionaryCurveLocations.keys(), dtype=numpy.uint64)
	# I'm computing groupZulu from curveLocations that are physically in `arrayCurveGroups`, so I'm using `indexCurveLocations`.
	arrayCurveGroups[:, indexGroupZulu] = (arrayCurveGroups[:, indexCurveLocations] & groupZuluLocator) >> 1
	# NOTE Do not alphabetize these operations. This column has curveLocations data that groupZulu needs.
	arrayCurveGroups[:, indexGroupAlpha] &= groupAlphaLocator

	return arrayCurveGroups

def count64(bridges: int, dictionaryCurveLocations: dict[int, int], bridgesMinimum: int = 0) -> tuple[int, dict[int, int]]:
	arrayCurveGroups: DataArray3D = convertDictionaryCurveLocations2array(dictionaryCurveLocations)
	del dictionaryCurveLocations

	while bridges > bridgesMinimum:
		bridges -= 1
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1 << (2 * bridges + 4))

		selectGroupAlphaCurves: SelectorBoolean = arrayCurveGroups[:, indexGroupAlpha] > numpy.uint64(1)
		curveLocationsGroupAlpha: DataArray1D = ((arrayCurveGroups[selectGroupAlphaCurves, indexGroupAlpha] >> 2)
			| (arrayCurveGroups[selectGroupAlphaCurves, indexGroupZulu] << 3)
			| ((numpy.uint64(1) - (arrayCurveGroups[selectGroupAlphaCurves, indexGroupAlpha] & 1)) << 1)
		)
		selectGroupAlphaCurvesLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectGroupAlphaCurves)[numpy.flatnonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)]

		selectGroupZuluCurves: SelectorBoolean = arrayCurveGroups[:, indexGroupZulu] > numpy.uint64(1)
		curveLocationsGroupZulu: DataArray1D = (arrayCurveGroups[selectGroupZuluCurves, indexGroupZulu] >> 1
			| arrayCurveGroups[selectGroupZuluCurves, indexGroupAlpha] << 2
			| (numpy.uint64(1) - (arrayCurveGroups[selectGroupZuluCurves, indexGroupZulu] & 1))
		)
		selectGroupZuluCurvesLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectGroupZuluCurves)[numpy.flatnonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)]

		selectBridgesSimpleLessThanMaximum: SelectorIndices = numpy.flatnonzero(
			((arrayCurveGroups[:, indexGroupAlpha] << 2) | (arrayCurveGroups[:, indexGroupZulu] << 3) | 3) < curveLocationsMAXIMUM
		)

		# Selectors for bridgesAligned -------------------------------------------------
		selectGroupAlphaAtEven: SelectorBoolean = (arrayCurveGroups[:, indexGroupAlpha] & 1) == numpy.uint64(0)
		selectGroupZuluAtEven: SelectorBoolean = (arrayCurveGroups[:, indexGroupZulu] & 1) == numpy.uint64(0)
		selectBridgesAligned: SelectorBoolean = selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven | selectGroupZuluAtEven)

		SliceΩ: slice[int, int, Literal[1]] = slice(0,0)
		sliceGroupAlpha = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectGroupAlphaCurvesLessThanMaximum.size)
		sliceGroupZulu = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectGroupZuluCurvesLessThanMaximum.size)
		sliceBridgesSimple = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectBridgesSimpleLessThanMaximum.size)
		# NOTE Maximum size, not actual size.
		sliceBridgesAligned = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectBridgesAligned.size)

		arrayCurveLocations: DataArray2D = numpy.zeros((SliceΩ.stop, columnsArrayCurveLocations), dtype=arrayCurveGroups.dtype)

		arrayCurveLocations[sliceGroupAlpha, indexDistinctCrossings] = arrayCurveGroups[selectGroupAlphaCurvesLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceGroupAlpha, indexCurveLocations] = curveLocationsGroupAlpha[numpy.flatnonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)]

		arrayCurveLocations[sliceGroupZulu, indexDistinctCrossings] = arrayCurveGroups[selectGroupZuluCurvesLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceGroupZulu, indexCurveLocations] = curveLocationsGroupZulu[numpy.flatnonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)]

		arrayCurveLocations[sliceBridgesSimple, indexDistinctCrossings] = arrayCurveGroups[selectBridgesSimpleLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceBridgesSimple, indexCurveLocations] = (
			(arrayCurveGroups[selectBridgesSimpleLessThanMaximum, indexGroupAlpha] << 2)
			| (arrayCurveGroups[selectBridgesSimpleLessThanMaximum, indexGroupZulu] << 3)
			| 3
		)

		curveLocationsGroupAlpha = None; del curveLocationsGroupAlpha  # pyright: ignore[reportAssignmentType] # noqa: E702
		curveLocationsGroupZulu = None; del curveLocationsGroupZulu  # pyright: ignore[reportAssignmentType] # noqa: E702
		selectBridgesSimpleLessThanMaximum = None; del selectBridgesSimpleLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaCurvesLessThanMaximum = None; del selectGroupAlphaCurvesLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluCurvesLessThanMaximum = None; del selectGroupZuluCurvesLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

		# NOTE this MODIFIES `arrayCurveGroups` for bridgesPairedToOdd ---------------------------------------------------------------------------------------
		selectBridgesGroupAlphaPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & selectGroupAlphaAtEven & (~selectGroupZuluAtEven))
		arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, indexGroupAlpha] = flipTheExtra_0b1(
			arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, indexGroupAlpha]
		)

		selectBridgesGroupZuluPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & (~selectGroupAlphaAtEven) & selectGroupZuluAtEven)
		arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, indexGroupZulu] = flipTheExtra_0b1(
			arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, indexGroupZulu]
		)

		selectBridgesGroupAlphaPairedToOdd = None; del selectBridgesGroupAlphaPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectBridgesGroupZuluPairedToOdd = None; del selectBridgesGroupZuluPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaAtEven = None; del selectGroupAlphaAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaCurves = None; del selectGroupAlphaCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluAtEven = None; del selectGroupZuluAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluCurves = None; del selectGroupZuluCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

		# bridgesAligned; bridgesAlignedAtEven, bridgesGroupAlphaPairedToOdd, bridgesGroupZuluPairedToOdd ------------------------------------------------------------------
		curveLocationsBridgesAligned: DataArray1D = (((arrayCurveGroups[selectBridgesAligned, indexGroupZulu] >> 2) << 1)
			| (arrayCurveGroups[selectBridgesAligned, indexGroupAlpha] >> 2)
		)
		selectBridgesAlignedLessThanMaximum: SelectorIndices = numpy.flatnonzero(selectBridgesAligned)[numpy.flatnonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)]

		sliceBridgesAligned = SliceΩ  = slice(sliceBridgesAligned.start, sliceBridgesAligned.stop - selectBridgesAligned.size + selectBridgesAlignedLessThanMaximum.size)
		arrayCurveLocations[sliceBridgesAligned, indexDistinctCrossings] = arrayCurveGroups[selectBridgesAlignedLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceBridgesAligned, indexCurveLocations] = curveLocationsBridgesAligned[numpy.flatnonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)]

		arrayCurveGroups = None; del arrayCurveGroups # pyright: ignore[reportAssignmentType]  # noqa: E702
		curveLocationsBridgesAligned = None; del curveLocationsBridgesAligned  # pyright: ignore[reportAssignmentType] # noqa: E702
		del curveLocationsMAXIMUM
		selectBridgesAligned = None; del selectBridgesAligned  # pyright: ignore[reportAssignmentType] # noqa: E702
		selectBridgesAlignedLessThanMaximum = None; del selectBridgesAlignedLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

		arrayCurveGroups = aggregateCurveLocations(arrayCurveLocations[0:SliceΩ.stop])

		arrayCurveLocations = None; del arrayCurveLocations # pyright: ignore[reportAssignmentType]  # noqa: E702
		del sliceBridgesAligned
		del sliceBridgesSimple
		del sliceGroupAlpha
		del sliceGroupZulu
		del SliceΩ
		goByeBye()

	return (bridges, convertArrayCurveGroups2dictionary(arrayCurveGroups))


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

	As first computed by Iwan Jensen in 2000, a(41) = 6664356253639465480.
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex
	See also https://oeis.org/A000682

	I'm sure you instantly observed that a(41) = 6664356253639465480.bit_length() = 63 bits.

	If you ask NumPy 2.3, "What is your relationship with integers with more than 64 bits?"
	NumPy will say, "It's complicated."

	Therefore, to take advantage of the computational excellence of NumPy when computing a(n) for n > 41, I must make some
	adjustments at the total count approaches 64 bits.

	The second complication is bit-packed integers. I use a loop that starts at `bridges = n` and decrements (`bridges -= 1`)
	`until bridges = 0`. If `bridges > 29`, some of the bit-packed integers have more than 64 bits. "Hey NumPy, can I use
	bit-packed integers with more than 64 bits?" NumPy: "It's complicated." Therefore, while `bridges` is decrementing, I don't
	use NumPy until I believe the bit-packed integers will be less than 64 bits.

	A third fact that works in my favor is that peak memory usage occurs when all of the integers are well under 64 bits wide.

	In total, to compute a(n) for "large" n, I use three-stages.
	1. I use Python primitive `int` contained in a Python primitive `dict`.
	2. When the bit width of `bridges` is small enough to use `numpy.uint64`, switch to `numpy` for the heavy lifting.
	3. When distinctCrossings subtotals exceed 64 bits, I must switch back to Python primitives.
	"""
# NOTE '29' is based on two things. 1) `bridges = 29`, groupZuluLocator = 0xaaaaaaaaaaaaaaaa.bit_length() = 64. 2) If `bridges =
# 30` or a larger number, `OverflowError: int too big to convert`. Conclusion: '29' isn't necessarily correct or the best value:
# it merely fits within my limited ability to assess the correct value.
	count64_bridgesMaximum = 29
	bridgesMinimum = 0  # NOTE This default value is necessary: it prevents `count64` from returning an incomplete dictionary when that is not necessary.

	# Oh, uh, I suddenly had an intuition that this method of computing 64bitLimitAsValueOf_n is, at best, wrong.
	distinctCrossings64bitLimitAsValueOf_n = 41
	distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG = distinctCrossings64bitLimitAsValueOf_n - 3
	distinctCrossings64bitLimitSafetyMargin = 4
	if n >= count64_bridgesMaximum:
		if n >= distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG:
			bridgesMinimum = n - distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG + distinctCrossings64bitLimitSafetyMargin
		n, dictionaryCurveLocations = count(n, dictionaryCurveLocations, count64_bridgesMaximum)
		gc.collect()
	n, dictionaryCurveLocations = count64(n, dictionaryCurveLocations, bridgesMinimum)
	if n > 0:
		gc.collect()
# TODO eliminate `countSimple`. My previous attempts have failed. In each case, the logic I used had at least one error that I
# could not find: the new result would be that the values returned by the second call to `count` would be wrong. There might be
# one or more errors in THIS module that are hidden until I try to eliminate `countSimple`. Or, the way I am thinking about the
# second call to `count` might be flawed. Or, bad luck. Whatever the case, when implementing this TODO, use it as an opportunity
# to scrutinize all of the code.
		return countSimple(n, dictionaryCurveLocations)
	return sum(dictionaryCurveLocations.values())
