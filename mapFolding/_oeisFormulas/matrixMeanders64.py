from typing import Any
import gc
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
# - Short-circuit returns in this module have led to unbound intermediates downstream and
#   inconsistent accumulator initialization across branches.
#
# Summary: No early-return guard clauses for empty inputs; preserve a single return statement
# and uniform shapes/dtypes throughout the function body.

"""Ideas, whether plausible or crazy

MEMORY MANAGEMENT is the ONLY issue that matters until we fix the crashes when trying to compute `bridges = 46` (or a lower value).

- `for indexRow in` "selectBridgesPairedToOdd": vectorize or add concurrency.
- numba
- codon for multiple modules, but I don't think I have figure out how to do that yet.
- codon compiled executable.
- ast construction of modules for only one value of n, especially converting the module to an executable with codon.
- The best, but craziest idea: fuck this shit: Somehow get healthcare so I can be healthy, not poor, not alone, and doing something I am good at.
- Use numpy bit shift operations.

"""

# NOTE `arrayCurveGroups`: Always use semantic index identifiers: Never hardcode the indices.
indexDistinctCrossings: int = 0
indexGroupAlpha: int = 1
indexGroupZulu: int = 2

# NOTE `arrayCurveLocations`: Always use semantic index identifiers: Never hardcode the indices.
indexDistinctCrossings: int = 0
indexCurveLocations: int = 1

groupAlphaLocator: numpy.uint64 = numpy.uint64(0x5555555555555555)
groupZuluLocator: numpy.uint64 = numpy.uint64(0xaaaaaaaaaaaaaaaa)

def aggregateCurveLocations(arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	uniqueness = numpy.unique_inverse(arrayCurveLocations[:, indexCurveLocations])

	arrayCurveGroups = numpy.zeros((uniqueness.values.shape[0], 3), dtype=numpy.uint64)
	numpy.add.at(arrayCurveGroups[:, indexDistinctCrossings], uniqueness.inverse_indices, arrayCurveLocations[:, indexDistinctCrossings])
	arrayCurveGroups[:, indexGroupAlpha] = uniqueness.values & groupAlphaLocator
	arrayCurveGroups[:, indexGroupZulu] = (uniqueness.values & groupZuluLocator) >> numpy.uint64(1)

	return arrayCurveGroups

def convertArrayCurveLocations2dictionary(arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]) -> dict[int, int]:
	uniqueness = numpy.unique_inverse(arrayCurveLocations[:, indexCurveLocations])

	arrayCurveGroups = numpy.zeros((uniqueness.values.shape[0], 2), dtype=numpy.uint64)
	numpy.add.at(arrayCurveGroups[:, indexDistinctCrossings], uniqueness.inverse_indices, arrayCurveLocations[:, indexDistinctCrossings])
	return {int(row[indexCurveLocations]): int(row[indexDistinctCrossings]) for row in arrayCurveGroups}

def convertDictionaryCurveLocations2array(dictionaryCurveLocations: dict[int, int]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	arrayKeys: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.fromiter(dictionaryCurveLocations.keys(), dtype=numpy.uint64)
	return numpy.column_stack((
		numpy.fromiter(dictionaryCurveLocations.values(), dtype=numpy.uint64)
		, arrayKeys & groupAlphaLocator
		, (arrayKeys & groupZuluLocator) >> numpy.uint64(1)
	))

def count64(bridges: int, dictionaryCurveLocations: dict[int, int], bridgesMinimum: int = 0) -> tuple[int, numpy.ndarray[tuple[int, ...], numpy.dtype[Any]]]:
	arrayCurveGroups: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = convertDictionaryCurveLocations2array(dictionaryCurveLocations)

	while bridges > bridgesMinimum:
		bridges -= 1
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)

		selectGroupAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupAlpha] > numpy.uint64(1)
		curveLocationsGroupAlpha = ((arrayCurveGroups[selectGroupAlphaCurves, indexGroupAlpha] >> numpy.uint64(2))
			| (arrayCurveGroups[selectGroupAlphaCurves, indexGroupZulu] << numpy.uint64(3))
			| ((numpy.uint64(1) - (arrayCurveGroups[selectGroupAlphaCurves, indexGroupAlpha] & numpy.uint64(1))) << numpy.uint64(1))
		)
		selectGroupAlphaCurvesLessThanMaximum: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]] = numpy.flatnonzero(selectGroupAlphaCurves)[numpy.nonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)[0]]

		selectGroupZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupZulu] > numpy.uint64(1)
		curveLocationsGroupZulu = (arrayCurveGroups[selectGroupZuluCurves, indexGroupZulu] >> numpy.uint64(1)
			| arrayCurveGroups[selectGroupZuluCurves, indexGroupAlpha] << numpy.uint64(2)
			| (numpy.uint64(1) - (arrayCurveGroups[selectGroupZuluCurves, indexGroupZulu] & numpy.uint64(1)))
		)
		selectGroupZuluCurvesLessThanMaximum: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]] = numpy.flatnonzero(selectGroupZuluCurves)[numpy.nonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)[0]]

		selectBridgesSimpleLessThanMaximum: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]] = numpy.nonzero(
			((arrayCurveGroups[:, indexGroupAlpha] << numpy.uint64(2)) | (arrayCurveGroups[:, indexGroupZulu] << numpy.uint64(3)) | numpy.uint64(3)) < curveLocationsMAXIMUM
		)[0]

		# Selectors for bridgesAligned -------------------------------------------------
		selectGroupAlphaAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveGroups[:, indexGroupAlpha] & numpy.uint64(1)) == numpy.uint64(0)
		selectGroupZuluAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveGroups[:, indexGroupZulu] & numpy.uint64(1)) == numpy.uint64(0)
		selectBridgesAligned = selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven | selectGroupZuluAtEven)

		SliceΩ: slice = slice(0,0)
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
			(arrayCurveGroups[selectBridgesSimpleLessThanMaximum, indexGroupAlpha] << numpy.uint64(2))
			| (arrayCurveGroups[selectBridgesSimpleLessThanMaximum, indexGroupZulu] << numpy.uint64(3))
			| numpy.uint64(3)
		)

		del curveLocationsGroupAlpha
		del curveLocationsGroupZulu
		del selectGroupAlphaCurvesLessThanMaximum
		del selectGroupZuluCurvesLessThanMaximum
		del selectBridgesSimpleLessThanMaximum
		gc.collect()

		# NOTE this MODIFIES `arrayCurveGroups` for bridgesPairedToOdd ---------------------------------------------------------------------------------------
		for indexRow in numpy.nonzero(selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven ^ selectGroupZuluAtEven))[0].tolist():
			if (arrayCurveGroups[indexRow, indexGroupAlpha] & 1) == 0:
				indexGroupToModify: int = indexGroupAlpha
			else:
				indexGroupToModify = indexGroupZulu

			XOrHere2makePair = 0b1
			findUnpaired_0b1 = 0

			while findUnpaired_0b1 >= 0:
				XOrHere2makePair <<= 2
				findUnpaired_0b1 += 1 if (arrayCurveGroups[indexRow, indexGroupToModify] & XOrHere2makePair) == 0 else -1

			arrayCurveGroups[indexRow, indexGroupToModify] ^= XOrHere2makePair

		# bridgesAligned; bridgesAlignedAtEven, bridgesGroupAlphaPairedToOdd, bridgesGroupZuluPairedToOdd -----------------
		curveLocationsBridgesAligned = (((arrayCurveGroups[selectBridgesAligned, indexGroupZulu] >> numpy.uint64(2)) << numpy.uint64(1))
			| (arrayCurveGroups[selectBridgesAligned, indexGroupAlpha] >> numpy.uint64(2))
		)

		selectBridgesAlignedLessThanMaximum: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]] = numpy.flatnonzero(selectBridgesAligned)[numpy.nonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)[0]]
		sliceBridgesAligned = SliceΩ  = slice(sliceBridgesAligned.start, sliceBridgesAligned.stop - selectBridgesAligned.size + selectBridgesAlignedLessThanMaximum.size)
		arrayCurveLocations[sliceBridgesAligned, indexDistinctCrossings] = arrayCurveGroups[selectBridgesAlignedLessThanMaximum, indexDistinctCrossings]
		arrayCurveLocations[sliceBridgesAligned, indexCurveLocations] = curveLocationsBridgesAligned[numpy.nonzero(curveLocationsBridgesAligned < curveLocationsMAXIMUM)[0]]

		del curveLocationsBridgesAligned
		del selectBridgesAlignedLessThanMaximum

		arrayCurveGroups = aggregateCurveLocations(arrayCurveLocations[0:SliceΩ.stop])
		gc.collect()

	else:  # noqa: PLW0120
		return (bridges, arrayCurveGroups.astype(__builtins__.int))

