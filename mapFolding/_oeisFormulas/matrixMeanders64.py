import numpy

"""TODO temporary notes while refactoring from `for` to vectorized operations:

Flow:
- mask construction
	- Z0Z_simpleBridges
	- bifurcationAlphaCurves
	- bifurcationZuluCurves
	- Z0Z_alignedBridges
	- Z0Z_bifurcationAlphaPaired
	- Z0Z_bifurcationZuluPaired
- aggregate curveLocationAnalysis and distinctCrossings
	- Details and flow are unclear
	- each mask generates curveLocationAnalysis, distinctCrossings pairs or indices for the distinctCrossings
	- the curveLocationAnalysis, distinctCrossings data is aggregated
	- distinctCrossings are summed for identical curveLocationsAnalysis
- replace `arrayCurveLocations` with new ndarray

"""

indexDistinctCrossings = int(0)  # noqa: RUF046, UP018
indexBifurcationAlpha = int(1)  # noqa: RUF046, UP018
indexBifurcationZulu = int(2)  # noqa: RUF046, UP018

def mergeIntoDictionary(arrayCurveLocationsAnalyzed: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]], LOOPdictionaryCurveLocations: dict[int, int]) -> dict[int, int]: # pyright: ignore[reportReturnType]
	"""Merge."""

def convertStartingCurveLocationsToArray(startingCurveLocations: dict[int, int]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	"""Convert startingCurveLocations dict to ndarray for curve location analysis."""
	bifurcationAlphaLocator: numpy.uint64 = numpy.uint64(0x5555555555555555)
	bifurcationZuluLocator: numpy.uint64 = numpy.uint64(0xaaaaaaaaaaaaaaaa)

	arrayKeys: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.fromiter(startingCurveLocations.keys(), dtype=numpy.uint64)

	return numpy.column_stack((
		numpy.fromiter(startingCurveLocations.values(), dtype=numpy.uint64)
		, arrayKeys & bifurcationAlphaLocator
		, (arrayKeys & bifurcationZuluLocator) >> numpy.uint64(1)
	))

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:

	while bridges > 0:
		bridges -= 1

		arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = convertStartingCurveLocationsToArray(startingCurveLocations)
		listArrayCurveLocationsAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]] = []

		# Vector conditional
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)

		# Z0Z_simpleBridges
		curveLocation_Z0Z_simpleBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[:, indexBifurcationAlpha] | (arrayCurveLocations[:, indexBifurcationZulu] << numpy.uint64(1))) << numpy.uint64(2)) | numpy.uint64(3))

		arrayCurveLocationsAnalyzed_Z0Z_simpleBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[:, indexDistinctCrossings][curveLocation_Z0Z_simpleBridges < curveLocationsMAXIMUM]
			, curveLocation_Z0Z_simpleBridges[curveLocation_Z0Z_simpleBridges < curveLocationsMAXIMUM]
		))
		listArrayCurveLocationsAnalyzed.append(arrayCurveLocationsAnalyzed_Z0Z_simpleBridges)

		# bifurcationAlphaCurves
		bifurcationAlphaHasCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexBifurcationAlpha] > numpy.uint64(1)

		# `bifurcationAlphaShiftRight2` is currently reused in `Z0Z_bifurcationZuluPaired`
		bifurcationAlphaShiftRight2: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationAlpha] >> numpy.uint64(2)
		bifurcatedZuluShiftLeft3: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationZulu] << numpy.uint64(3)
		makeBifurcationAlphaEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (numpy.uint64(1) - (arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationAlpha] & numpy.uint64(1))) << numpy.uint64(1)
		curveLocationComputation_bifurcationAlphaHasCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.bitwise_or.reduce((
			bifurcationAlphaShiftRight2, bifurcatedZuluShiftLeft3, makeBifurcationAlphaEven), axis=0)
		curveLocation_bifurcationAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = curveLocationComputation_bifurcationAlphaHasCurves

		arrayCurveLocationsAnalyzed_bifurcationAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[bifurcationAlphaHasCurves][:, indexDistinctCrossings][curveLocation_bifurcationAlphaCurves < curveLocationsMAXIMUM]
			, curveLocation_bifurcationAlphaCurves[curveLocation_bifurcationAlphaCurves < curveLocationsMAXIMUM]
		))
		listArrayCurveLocationsAnalyzed.append(arrayCurveLocationsAnalyzed_bifurcationAlphaCurves)

		# bifurcationZuluCurves
		bifurcationZuluHasCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexBifurcationZulu] > numpy.uint64(1)

		curveLocationComputation_bifurcationZuluHasCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.bitwise_or.reduce((
			arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationZulu] >> numpy.uint64(1)
			, arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationAlpha] << numpy.uint64(2)
			, ~ arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationZulu] & numpy.uint64(1)
		), axis=0)
		curveLocation_bifurcationZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = curveLocationComputation_bifurcationZuluHasCurves

		arrayCurveLocationsAnalyzed_bifurcationZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[bifurcationZuluHasCurves][:, indexDistinctCrossings][curveLocation_bifurcationZuluCurves < curveLocationsMAXIMUM]
			, curveLocation_bifurcationZuluCurves[curveLocation_bifurcationZuluCurves < curveLocationsMAXIMUM]
		))

		# Defining masks
		bifurcationAlphaIsEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexBifurcationAlpha] & numpy.uint64(1)) == numpy.uint64(0)
		bifurcationZuluIsEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexBifurcationZulu] & numpy.uint64(1)) == numpy.uint64(0)
		anyBifurcationIsEven = numpy.logical_or.reduce((bifurcationAlphaIsEven, bifurcationZuluIsEven))
		Z0Z_allAlignedBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((bifurcationAlphaHasCurves, bifurcationZuluHasCurves, anyBifurcationIsEven))
		Z0Z_bifurcationAlphaPaired: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_allAlignedBridges, ~bifurcationZuluIsEven))
		Z0Z_bifurcationZuluPaired: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_allAlignedBridges, ~bifurcationAlphaIsEven))
		listArrayCurveLocationsAnalyzed.append(arrayCurveLocationsAnalyzed_bifurcationZuluCurves)

		# Z0Z_alignedBridges
		Z0Z_alignedBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_allAlignedBridges, ~Z0Z_bifurcationAlphaPaired, ~Z0Z_bifurcationZuluPaired))

		curveLocation_Z0Z_alignedBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[Z0Z_alignedBridges][:, indexBifurcationZulu] >> numpy.uint64(2)) << numpy.uint64(1))
			| (arrayCurveLocations[Z0Z_alignedBridges][:, indexBifurcationAlpha] >> numpy.uint64(2))
		)

		arrayCurveLocationsAnalyzed_Z0Z_alignedBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[Z0Z_alignedBridges][:, indexDistinctCrossings][curveLocation_Z0Z_alignedBridges < curveLocationsMAXIMUM]
			, curveLocation_Z0Z_alignedBridges[curveLocation_Z0Z_alignedBridges < curveLocationsMAXIMUM]
		))
		listArrayCurveLocationsAnalyzed.append(arrayCurveLocationsAnalyzed_Z0Z_alignedBridges)

		LOOPdictionaryCurveLocations: dict[int, int] = {}
		for LOOPindex in range(len(arrayCurveLocations)):
			# Z0Z_bifurcationAlphaPaired
			LOaPindex = LOOPindex
			LOaPbifurcationAlphaHasCurves = arrayCurveLocations[LOaPindex, indexBifurcationAlpha] != 1
			LOaPbifurcationZuluHasCurves = arrayCurveLocations[LOaPindex, indexBifurcationZulu] != 1
			LOaPbifurcationAlphaIsEven = not (arrayCurveLocations[LOaPindex, indexBifurcationAlpha] & 1)
			LOaPbifurcationZuluIsEven = not (arrayCurveLocations[LOaPindex, indexBifurcationZulu] & 1)

			if LOaPbifurcationAlphaHasCurves and LOaPbifurcationZuluHasCurves and LOaPbifurcationAlphaIsEven and not LOaPbifurcationZuluIsEven:
				LOaPXOrHere2makePair = 0b1
				LOaPfindUnpaired_0b1 = 0
				while LOaPfindUnpaired_0b1 >= 0:
					LOaPXOrHere2makePair <<= 2
					LOaPfindUnpaired_0b1 += 1 if (arrayCurveLocations[LOaPindex, indexBifurcationAlpha] & LOaPXOrHere2makePair) == 0 else -1
				LOaPbifurcationAlphaPaired = (arrayCurveLocations[LOaPindex, indexBifurcationAlpha] ^ LOaPXOrHere2makePair) >> 2

				curveLocation_Z0Z_bifurcationAlphaPaired = ((arrayCurveLocations[LOaPindex, indexBifurcationZulu] >> 2) << 1) | LOaPbifurcationAlphaPaired
				if curveLocation_Z0Z_bifurcationAlphaPaired < curveLocationsMAXIMUM:
					LOOPdictionaryCurveLocations[curveLocation_Z0Z_bifurcationAlphaPaired] = LOOPdictionaryCurveLocations.get(curveLocation_Z0Z_bifurcationAlphaPaired, 0) + arrayCurveLocations[LOaPindex, indexDistinctCrossings]

			# Z0Z_bifurcationZuluPaired
			LOBPindex = LOOPindex
			LOBPbifurcationAlphaHasCurves = arrayCurveLocations[LOBPindex, indexBifurcationAlpha] != 1
			LOBPbifurcationZuluHasCurves = arrayCurveLocations[LOBPindex, indexBifurcationZulu] != 1
			LOBPbifurcationAlphaIsEven = not (arrayCurveLocations[LOBPindex, indexBifurcationAlpha] & 1)
			LOBPbifurcationZuluIsEven = not (arrayCurveLocations[LOBPindex, indexBifurcationZulu] & 1)

			LOBPbifurcationAlphaShiftRight2 = arrayCurveLocations[LOBPindex, indexBifurcationAlpha] >> 2
			if LOBPbifurcationAlphaHasCurves and LOBPbifurcationZuluHasCurves and LOBPbifurcationZuluIsEven and not LOBPbifurcationAlphaIsEven:
				LOBPXOrHere2makePair = 0b1
				LOBPfindUnpaired_0b1 = 0
				while LOBPfindUnpaired_0b1 >= 0:
					LOBPXOrHere2makePair <<= 2
					LOBPfindUnpaired_0b1 += 1 if (arrayCurveLocations[LOBPindex, indexBifurcationZulu] & LOBPXOrHere2makePair) == 0 else -1
				LOBPbifurcationZuluPaired = arrayCurveLocations[LOBPindex, indexBifurcationZulu] ^ LOBPXOrHere2makePair

				curveLocation_Z0Z_bifurcationAlphaPaired = ((LOBPbifurcationZuluPaired >> 2) << 1) | LOBPbifurcationAlphaShiftRight2
				if curveLocation_Z0Z_bifurcationAlphaPaired < curveLocationsMAXIMUM:
					LOOPdictionaryCurveLocations[curveLocation_Z0Z_bifurcationAlphaPaired] = LOOPdictionaryCurveLocations.get(curveLocation_Z0Z_bifurcationAlphaPaired, 0) + arrayCurveLocations[LOBPindex, indexDistinctCrossings]

		# Combine all arrays into single array
		CONVERTarrayCurveLocationsAnalyzedMerged: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.vstack(listArrayCurveLocationsAnalyzed)

		CONVERTcurveLocationColumn: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = CONVERTarrayCurveLocationsAnalyzedMerged[:, 1]
		CONVERTdistinctCrossingsColumn: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = CONVERTarrayCurveLocationsAnalyzedMerged[:, 0]

		CONVERTorderByCurveLocation: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.argsort(CONVERTcurveLocationColumn, kind='mergesort')
		CONVERTsortedCurveLocation: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = CONVERTcurveLocationColumn[CONVERTorderByCurveLocation]
		CONVERTsortedDistinctCrossings: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = CONVERTdistinctCrossingsColumn[CONVERTorderByCurveLocation]

		CONVERTindicesWhereKeyChanges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(numpy.diff(CONVERTsortedCurveLocation) != 0)[0] + 1
		CONVERTgroupStarts: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.concatenate((numpy.array([0], dtype=numpy.int64), CONVERTindicesWhereKeyChanges))
		CONVERTsegmentSums: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.add.reduceat(CONVERTsortedDistinctCrossings, CONVERTgroupStarts)
		CONVERTuniqueCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = CONVERTsortedCurveLocation[CONVERTgroupStarts]

		for CONVERTindexGroup in range(CONVERTuniqueCurveLocations.shape[0]):
			CONVERTkeyCurveLocation = int(CONVERTuniqueCurveLocations[CONVERTindexGroup])
			CONVERTvalueDistinctCrossings = int(CONVERTsegmentSums[CONVERTindexGroup])
			LOOPdictionaryCurveLocations[CONVERTkeyCurveLocation] = LOOPdictionaryCurveLocations.get(CONVERTkeyCurveLocation, 0) + CONVERTvalueDistinctCrossings

		startingCurveLocations.clear()
		startingCurveLocations, LOOPdictionaryCurveLocations = LOOPdictionaryCurveLocations, startingCurveLocations

	return sum(startingCurveLocations.values())

