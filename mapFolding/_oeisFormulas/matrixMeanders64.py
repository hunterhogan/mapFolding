from gc import collect as goByeBye
from typing import Literal
import numpy

# DEVELOPMENT INSTRUCTIONS FOR THIS MODULE
#
# Avoid early-return guard clauses (also called short-circuit returns or multiple exit points)
# that immediately return a literal result for empty inputs (e.g., returning an empty array or
# empty dictionary as soon as size==0).
#
# Why this is harmful here:
# - This codebase enforces a single-return-per-function pattern; multiple exits make
#   control flow harder to analyze and transform.
# - Our AST/Numba job generator assumes one consistent exit path with stable shapes/dtypes;
#   early returns often create shape/type divergence that breaks specialization.
#
# Do NOT add artificial safety limits (e.g., maxIterations counters) to prevent infinite loops.
# Such limits mask underlying algorithmic problems and create non-deterministic behavior.
# If an algorithm has potential for infinite loops, fix the root cause in the mathematical logic,
# not by adding arbitrary iteration caps that could truncate valid computations.
#
# Summary: No early-return guard clauses for empty inputs; preserve a single return statement
# and uniform shapes/dtypes throughout the function body; an empty input is a problem: fail early.
# Always use semantic index identifiers: Never hardcode the indices.

# NOTE `arrayCurveGroups`
indexDistinctCrossings: int = 0
indexGroupAlpha: int = 1
indexGroupZulu: int = 2

# NOTE `arrayCurveLocations`
indexDistinctCrossings: int = 0
indexCurveLocations: int = 1

groupAlphaLocator: int = 0x5555555555555555
groupZuluLocator: int = 0xaaaaaaaaaaaaaaaa

def aggregateCurveLocations(arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	uniqueness = numpy.unique_inverse(arrayCurveLocations[:, indexCurveLocations])

	arrayCurveGroups = numpy.zeros((uniqueness.values.shape[0], 3), dtype=numpy.uint64)
	numpy.add.at(arrayCurveGroups[:, indexDistinctCrossings], uniqueness.inverse_indices, arrayCurveLocations[:, indexDistinctCrossings])
	arrayCurveGroups[:, indexGroupAlpha] = uniqueness.values & groupAlphaLocator
	arrayCurveGroups[:, indexGroupZulu] = (uniqueness.values & groupZuluLocator) >> 1

	return arrayCurveGroups

def convertArrayCurveGroups2dictionary(arrayCurveGroups: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]) -> dict[int, int]:
	"""'Smush' `groupAlpha` and `groupZulu` back together into `curveLocations`."""
	arrayCurveGroups[:, indexCurveLocations] = arrayCurveGroups[:, indexGroupAlpha] | (arrayCurveGroups[:, indexGroupZulu] << 1)
	return {int(row[indexCurveLocations]): int(row[indexDistinctCrossings]) for row in arrayCurveGroups[:, [indexDistinctCrossings, indexCurveLocations]]}

def convertDictionaryCurveLocations2array(dictionaryCurveLocations: dict[int, int]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	arrayCurveGroups = numpy.tile(numpy.fromiter(dictionaryCurveLocations.values(), dtype=numpy.uint64), (3, 1)).T

	# I'm putting curveLocations into `arrayCurveGroups`, so I'm using `indexCurveLocations` even though it's not an index for this array.
	arrayCurveGroups[:, indexCurveLocations] = numpy.fromiter(dictionaryCurveLocations.keys(), dtype=numpy.uint64)
	# I'm computing groupZulu from curveLocations that are physically in `arrayCurveGroups`, so I'm using `indexCurveLocations`.
	arrayCurveGroups[:, indexGroupZulu] = (arrayCurveGroups[:, indexCurveLocations] & groupZuluLocator) >> 1
	# NOTE Do not alphabetize these operations. This column has curveLocations data that groupZulu needs.
	arrayCurveGroups[:, indexGroupAlpha] &= groupAlphaLocator

	return arrayCurveGroups

def count64(bridges: int, dictionaryCurveLocations: dict[int, int], bridgesMinimum: int = 0) -> tuple[int, dict[int, int]]:
	arrayCurveGroups: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = convertDictionaryCurveLocations2array(dictionaryCurveLocations)
	del dictionaryCurveLocations

	while bridges > bridgesMinimum:
		bridges -= 1
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1 << (2 * bridges + 4))
# TODO Are there any 0 values in `arrayCurveGroups[:, indexGroupAlpha]`?
		selectGroupAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupAlpha] > numpy.uint64(1)
		curveLocationsGroupAlpha = ((arrayCurveGroups[selectGroupAlphaCurves, indexGroupAlpha] >> 2)
			| (arrayCurveGroups[selectGroupAlphaCurves, indexGroupZulu] << 3)
			| ((numpy.uint64(1) - (arrayCurveGroups[selectGroupAlphaCurves, indexGroupAlpha] & 1)) << 1)
		)
		selectGroupAlphaCurvesLessThanMaximum: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]] = numpy.flatnonzero(selectGroupAlphaCurves)[numpy.nonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)[0]]

		selectGroupZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupZulu] > numpy.uint64(1)
		curveLocationsGroupZulu = (arrayCurveGroups[selectGroupZuluCurves, indexGroupZulu] >> 1
			| arrayCurveGroups[selectGroupZuluCurves, indexGroupAlpha] << 2
			| (numpy.uint64(1) - (arrayCurveGroups[selectGroupZuluCurves, indexGroupZulu] & 1))
		)
		selectGroupZuluCurvesLessThanMaximum: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]] = numpy.flatnonzero(selectGroupZuluCurves)[numpy.nonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)[0]]

		selectBridgesSimpleLessThanMaximum: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]] = numpy.nonzero(
			((arrayCurveGroups[:, indexGroupAlpha] << 2) | (arrayCurveGroups[:, indexGroupZulu] << 3) | 3) < curveLocationsMAXIMUM
		)[0]

		# Selectors for bridgesAligned -------------------------------------------------
# TODO What is the type returned by `(arrayCurveGroups[:, indexGroupAlpha] & 1)`?
		selectGroupAlphaAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveGroups[:, indexGroupAlpha] & 1) == numpy.uint64(0)
		selectGroupZuluAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveGroups[:, indexGroupZulu] & 1) == numpy.uint64(0)
		selectBridgesAligned = selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven | selectGroupZuluAtEven)

		SliceΩ: slice[int, int, Literal[1]] = slice(0,0)
		sliceGroupAlpha = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectGroupAlphaCurvesLessThanMaximum.size)
		sliceGroupZulu = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectGroupZuluCurvesLessThanMaximum.size)
		sliceBridgesSimple = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectBridgesSimpleLessThanMaximum.size)
		# NOTE Maximum size, not actual size.
		sliceBridgesAligned = SliceΩ  = slice(SliceΩ.stop, SliceΩ.stop + selectBridgesAligned.size)

		arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.zeros((sliceBridgesAligned.stop, 2), dtype=arrayCurveGroups.dtype)

		arrayCurveLocations[sliceGroupAlpha, indexDistinctCrossings] = arrayCurveGroups[selectGroupAlphaCurvesLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceGroupAlpha, indexCurveLocations] = curveLocationsGroupAlpha[numpy.nonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)[0]]

		arrayCurveLocations[sliceGroupZulu, indexDistinctCrossings] = arrayCurveGroups[selectGroupZuluCurvesLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceGroupZulu, indexCurveLocations] = curveLocationsGroupZulu[numpy.nonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)[0]]

		arrayCurveLocations[sliceBridgesSimple, indexDistinctCrossings] = arrayCurveGroups[selectBridgesSimpleLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceBridgesSimple, indexCurveLocations] = (
			(arrayCurveGroups[selectBridgesSimpleLessThanMaximum, indexGroupAlpha] << 2)
			| (arrayCurveGroups[selectBridgesSimpleLessThanMaximum, indexGroupZulu] << 3)
			| 3
		)

		curveLocationsGroupAlpha = None; del curveLocationsGroupAlpha  # noqa: E702
		curveLocationsGroupZulu = None; del curveLocationsGroupZulu  # noqa: E702
		selectGroupAlphaCurvesLessThanMaximum = None; del selectGroupAlphaCurvesLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluCurvesLessThanMaximum = None; del selectGroupZuluCurvesLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectBridgesSimpleLessThanMaximum = None; del selectBridgesSimpleLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702

		# NOTE this MODIFIES `arrayCurveGroups` for bridgesPairedToOdd ---------------------------------------------------------------------------------------
		# I don't think Claude's changes are an improvement. But, I'll get rid of these UNNECESSARY intermediates and see if that is better.
# OPTIMIZED bridgesPairedToOdd: Memory-optimized replacement eliminates .tolist() conversion and Python loop overhead

		# Process each target row with memory-optimized operations
		for indexRow in numpy.nonzero(selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven ^ selectGroupZuluAtEven))[0]:
			if selectGroupAlphaAtEven[indexRow]:
				indexGroupToModify: int = indexGroupAlpha
			else:
				indexGroupToModify = indexGroupZulu

			XOrHere2makePair = 0b1
			findUnpaired_0b1 = 0

			while findUnpaired_0b1 >= 0:
				XOrHere2makePair <<= 2
				findUnpaired_0b1 += 1 if (arrayCurveGroups[indexRow, indexGroupToModify] & XOrHere2makePair) == 0 else -1

			arrayCurveGroups[indexRow, indexGroupToModify] ^= XOrHere2makePair

# TODO END refactor area ----------------------------------------------------------------------------------------------------------------------------------------------------

		selectGroupAlphaCurves = None; del selectGroupAlphaCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluCurves = None; del selectGroupZuluCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaAtEven = None; del selectGroupAlphaAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluAtEven = None; del selectGroupZuluAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702

		# bridgesAligned; bridgesAlignedAtEven, bridgesGroupAlphaPairedToOdd, bridgesGroupZuluPairedToOdd -----------------
		curveLocationsBridgesAligned = (((arrayCurveGroups[selectBridgesAligned, indexGroupZulu] >> 2) << 1)
			| (arrayCurveGroups[selectBridgesAligned, indexGroupAlpha] >> 2)
		)

		selectBridgesAlignedLessThanMaximum: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]] = numpy.flatnonzero(selectBridgesAligned)[numpy.nonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)[0]]
		sliceBridgesAligned = SliceΩ  = slice(sliceBridgesAligned.start, sliceBridgesAligned.stop - selectBridgesAligned.size + selectBridgesAlignedLessThanMaximum.size)
		arrayCurveLocations[sliceBridgesAligned, indexDistinctCrossings] = arrayCurveGroups[selectBridgesAlignedLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceBridgesAligned, indexCurveLocations] = curveLocationsBridgesAligned[numpy.nonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)[0]]

		selectBridgesAligned = None; del selectBridgesAligned  # noqa: E702
		curveLocationsBridgesAligned = None; del curveLocationsBridgesAligned  # noqa: E702
		selectBridgesAlignedLessThanMaximum = None; del selectBridgesAlignedLessThanMaximum # pyright: ignore[reportAssignmentType]  # noqa: E702
		del curveLocationsMAXIMUM

		arrayCurveGroups = None; del arrayCurveGroups # pyright: ignore[reportAssignmentType]  # noqa: E702

		arrayCurveGroups = aggregateCurveLocations(arrayCurveLocations[0:SliceΩ.stop])
		arrayCurveLocations = None; del arrayCurveLocations # pyright: ignore[reportAssignmentType]  # noqa: E702
		del SliceΩ
		del sliceGroupAlpha
		del sliceGroupZulu
		del sliceBridgesSimple
		del sliceBridgesAligned

		goByeBye()

	return (bridges, convertArrayCurveGroups2dictionary(arrayCurveGroups))

