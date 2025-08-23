from collections import defaultdict
from gc import collect as goByeBye
from itertools import repeat
from typing import Any, Literal
import numpy

# DEVELOPMENT INSTRUCTIONS FOR THIS MODULE
#
# Avoid early-return guard clauses, short-circuit returns, or multiple exit points. This codebase enforces a
# single-return-per-function pattern; multiple exits make control flow harder to analyze and transform. Our AST/Numba job
# generator assumes one consistent exit path with stable shapes/dtypes. An empty input is a problem: fail early.
#
# Do NOT add artificial safety limits (e.g., maxIterations counters) to prevent infinite loops. Such limits mask underlying
# algorithmic problems and create non-deterministic behavior. If an algorithm has potential for infinite loops, fix the root cause
# in the mathematical logic, not by adding arbitrary iteration caps that could truncate valid computations.
#
# Always use semantic index identifiers: Never hardcode the indices.

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

def aggregateCurveLocations(arrayCurveLocations: DataArray2D) -> DataArray3D:
	arrayCurveGroups: DataArray3D = numpy.tile(numpy.unique(arrayCurveLocations[:, indexCurveLocations]), (columnsArrayCurveGroups, 1)).T
	arrayCurveGroups[:, indexDistinctCrossings] = 0
	numpy.add.at(arrayCurveGroups[:, indexDistinctCrossings], numpy.searchsorted(arrayCurveGroups[:, indexCurveLocations], arrayCurveLocations[:, indexCurveLocations]), arrayCurveLocations[:, indexDistinctCrossings])
	# I'm computing groupZulu from curveLocations that are physically in `arrayCurveGroups`, so I'm using `indexCurveLocations`.
	arrayCurveGroups[:, indexGroupZulu] = (arrayCurveGroups[:, indexCurveLocations] & groupZuluLocator) >> 1
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

type WhyDoesThisIteratorRefuseToDie = int
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

		"""NOTE Potentially useful code if we figure out vectorization of bridgesPairedToOdd:
		selectBridgesGroupAlphaPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & selectGroupAlphaAtEven & (~selectGroupZuluAtEven))
		selectBridgesGroupZuluPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & (~selectGroupAlphaAtEven) & selectGroupZuluAtEven)
		"""
		# Z0Z_dictionary: dict[int, list[tuple[int, int]]] = defaultdict(list)

# ruff: noqa: ERA001  # noqa: RUF100
# TODO START refactor area 96.24s out of 168.12s = 57.24% of ALL run time in the test ------------------------------------------------------------------------
		# NOTE this MODIFIES `arrayCurveGroups` for bridgesPairedToOdd ---------------------------------------------------------------------------------------
		for indexRow, indexGroupToModify in [*zip(numpy.flatnonzero(selectBridgesAligned & selectGroupAlphaAtEven & (~selectGroupZuluAtEven)), repeat(indexGroupAlpha))
											, *zip(numpy.flatnonzero(selectBridgesAligned & (~selectGroupAlphaAtEven) & selectGroupZuluAtEven), repeat(indexGroupZulu))]:

			XOrHere2makePair: int = 1
			balancePairs: int = 0
			valueGroup: int = int(arrayCurveGroups[indexRow, indexGroupToModify])
			while True:
				XOrHere2makePair <<= 2
				if (valueGroup & XOrHere2makePair) == 0:
					balancePairs += 1
				else:
					balancePairs -= 1
				if balancePairs < 0:
					break

			arrayCurveGroups[indexRow, indexGroupToModify] ^= XOrHere2makePair 

# TODO END refactor area ----------------------------------------------------------------------------------------------------------------------------------------------------




		# 	Z0Z_dictionary[XOrHere2makePair].append((indexRow, indexGroupToModify))
		# for XOrHere2makePair, Z0Z_list in Z0Z_dictionary.items():
		# 	selectXOrHere2makePair: SelectorIndices = numpy.array(Z0Z_list, dtype=int)
		# 	print(selectXOrHere2makePair.shape)
		# 	arrayCurveGroups[selectXOrHere2makePair] ^= XOrHere2makePair



		selectGroupAlphaAtEven = None; del selectGroupAlphaAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupAlphaCurves = None; del selectGroupAlphaCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluAtEven = None; del selectGroupZuluAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
		selectGroupZuluCurves = None; del selectGroupZuluCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
		goByeBye()

		# bridgesAligned; bridgesAlignedAtEven, bridgesGroupAlphaPairedToOdd, bridgesGroupZuluPairedToOdd -----------------
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

