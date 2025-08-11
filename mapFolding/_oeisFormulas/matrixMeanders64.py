import numpy

"""TODO temporary notes:

Flow:
- mask construction
	- Z0Z_simpleBridges
	- bifurcationAlphaCurves
	- bifurcationZuluCurves
	- Z0Z_alignedBridges
	- Z0Z_LOOPbifurcationAlphaPaired
	- Z0Z_LOOPbifurcationZuluPaired
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

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:

	while bridges > 0:
		bridges -= 1

		# Convert startingCurveLocations to ndarray
		bifurcationAlphaLocator: numpy.uint64 = numpy.uint64(0x5555555555555555)
		bifurcationZuluLocator: numpy.uint64 = numpy.uint64(0xaaaaaaaaaaaaaaaa)
		arrayKeys: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.fromiter(startingCurveLocations.keys(), dtype=numpy.uint64)
		arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			numpy.fromiter(startingCurveLocations.values(), dtype=numpy.uint64)
			, arrayKeys & bifurcationAlphaLocator
			, (arrayKeys & bifurcationZuluLocator) >> numpy.uint64(1)
		))

		# Vector conditional
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)

		# Z0Z_simpleBridges
		curveLocation_Z0Z_simpleBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[:, indexBifurcationAlpha] | (arrayCurveLocations[:, indexBifurcationZulu] << numpy.uint64(1))) << numpy.uint64(2)) | numpy.uint64(3))

		arrayCurveLocationsAnalyzed_Z0Z_simpleBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[:, indexDistinctCrossings][curveLocation_Z0Z_simpleBridges < curveLocationsMAXIMUM]
			, curveLocation_Z0Z_simpleBridges[curveLocation_Z0Z_simpleBridges < curveLocationsMAXIMUM]
		))

		# bifurcationAlphaCurves
		bifurcationAlphaHasCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexBifurcationAlpha] > numpy.uint64(1)

		bifurcationAlphaShiftRight2: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationAlpha] >> numpy.uint64(2)
		bifurcatedZuluShiftLeft3: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationZulu] << numpy.uint64(3)
		makeBifurcationAlphaEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (numpy.uint64(1) - (arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationAlpha] & numpy.uint64(1))) << numpy.uint64(1)
		curveLocation_bifurcationAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			bifurcationAlphaShiftRight2 | bifurcatedZuluShiftLeft3 | makeBifurcationAlphaEven
		)

		arrayCurveLocationsAnalyzed_bifurcationAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[bifurcationAlphaHasCurves][:, indexDistinctCrossings][curveLocation_bifurcationAlphaCurves < curveLocationsMAXIMUM]
			, curveLocation_bifurcationAlphaCurves[curveLocation_bifurcationAlphaCurves < curveLocationsMAXIMUM]
		))

		# bifurcationZuluCurves
		bifurcationZuluHasCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexBifurcationZulu] > numpy.uint64(1)

		bifurcationZuluShiftRight1: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationZulu] >> numpy.uint64(1)
		bifurcationAlphaShiftLeft2: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationAlpha] << numpy.uint64(2)
		makeBifurcationZuluEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = ~ arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationZulu] & numpy.uint64(1)
		curveLocation_bifurcationZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = bifurcationZuluShiftRight1 | bifurcationAlphaShiftLeft2 | makeBifurcationZuluEven

		arrayCurveLocationsAnalyzed_bifurcationZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[bifurcationZuluHasCurves][:, indexDistinctCrossings][curveLocation_bifurcationZuluCurves < curveLocationsMAXIMUM]
			, curveLocation_bifurcationZuluCurves[curveLocation_bifurcationZuluCurves < curveLocationsMAXIMUM]
		))

		# Z0Z_alignedBridges
		# ABABbifurcationAlphaHasCurves = arrayCurveLocations[:, indexBifurcationAlpha] > numpy.uint64(1)
		# ABABbifurcationZuluHasCurves = arrayCurveLocations[:, indexBifurcationZulu] > numpy.uint64(1)
		# ABABbifurcationAlphaIsEven = arrayCurveLocations[:, indexBifurcationAlpha] % 2 == 0
		# ABABbifurcationZuluIsEven = arrayCurveLocations[:, indexBifurcationZulu] % 2 == 0

		# Z0Z_alignedBridges = ABABbifurcationAlphaHasCurves and ABABbifurcationZuluHasCurves and ABABbifurcationAlphaIsEven and ABABbifurcationZuluIsEven

		# curveLocation_Z0Z_alignedBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
		# 	((arrayCurveLocations[Z0Z_alignedBridges][:, indexBifurcationZulu] >> numpy.uint64(2)) << numpy.uint64(1))
		# 	| (arrayCurveLocations[Z0Z_alignedBridges][:, indexBifurcationAlpha] >> numpy.uint64(2))
		# )

		# arrayCurveLocationsAnalyzed_Z0Z_alignedBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
		# 	arrayCurveLocations[bifurcationZuluHasCurves][:, indexDistinctCrossings][curveLocation_Z0Z_alignedBridges < curveLocationsMAXIMUM]
		# 	, curveLocation_Z0Z_alignedBridges[curveLocation_Z0Z_alignedBridges < curveLocationsMAXIMUM]
		# ))

		LOOPdictionaryCurveLocations: dict[int, int] = {}
		for LOOPindex in range(len(arrayCurveLocations)):

			# Z0Z_alignedBridges
			LOOPbifurcationAlphaHasCurves = arrayCurveLocations[LOOPindex, indexBifurcationAlpha] != 1
			LOOPbifurcationZuluHasCurves = arrayCurveLocations[LOOPindex, indexBifurcationZulu] != 1
			LOOPbifurcationAlphaIsEven = not (arrayCurveLocations[LOOPindex, indexBifurcationAlpha] & 1)
			LOOPbifurcationZuluIsEven = not (arrayCurveLocations[LOOPindex, indexBifurcationZulu] & 1)

			LOOPbifurcationAlphaShiftRight2 = arrayCurveLocations[LOOPindex, indexBifurcationAlpha] >> 2
			if LOOPbifurcationAlphaHasCurves and LOOPbifurcationZuluHasCurves and LOOPbifurcationAlphaIsEven and LOOPbifurcationZuluIsEven:

				LOOPcurveLocationAnalysis = ((arrayCurveLocations[LOOPindex, indexBifurcationZulu] >> 2) << 1) | LOOPbifurcationAlphaShiftRight2
				if LOOPcurveLocationAnalysis < curveLocationsMAXIMUM:
					LOOPdictionaryCurveLocations[LOOPcurveLocationAnalysis] = LOOPdictionaryCurveLocations.get(LOOPcurveLocationAnalysis, 0) + arrayCurveLocations[LOOPindex, indexDistinctCrossings]

			# Z0Z_bifurcationAlphaPaired
			LOOPbifurcationAlphaHasCurves = arrayCurveLocations[LOOPindex, indexBifurcationAlpha] != 1
			LOOPbifurcationZuluHasCurves = arrayCurveLocations[LOOPindex, indexBifurcationZulu] != 1
			LOOPbifurcationAlphaIsEven = not (arrayCurveLocations[LOOPindex, indexBifurcationAlpha] & 1)
			LOOPbifurcationZuluIsEven = not (arrayCurveLocations[LOOPindex, indexBifurcationZulu] & 1)

			if LOOPbifurcationAlphaHasCurves and LOOPbifurcationZuluHasCurves and LOOPbifurcationAlphaIsEven and not LOOPbifurcationZuluIsEven:
				LOOPXOrHere2makePair = 0b1
				LOOPfindUnpaired_0b1 = 0
				while LOOPfindUnpaired_0b1 >= 0:
					LOOPXOrHere2makePair <<= 2
					LOOPfindUnpaired_0b1 += 1 if (arrayCurveLocations[LOOPindex, indexBifurcationAlpha] & LOOPXOrHere2makePair) == 0 else -1
				LOOPbifurcationAlphaPaired = (arrayCurveLocations[LOOPindex, indexBifurcationAlpha] ^ LOOPXOrHere2makePair) >> 2

				LOOPcurveLocationAnalysis = ((arrayCurveLocations[LOOPindex, indexBifurcationZulu] >> 2) << 1) | LOOPbifurcationAlphaPaired
				if LOOPcurveLocationAnalysis < curveLocationsMAXIMUM:
					LOOPdictionaryCurveLocations[LOOPcurveLocationAnalysis] = LOOPdictionaryCurveLocations.get(LOOPcurveLocationAnalysis, 0) + arrayCurveLocations[LOOPindex, indexDistinctCrossings]

			# Z0Z_LOOPbifurcationZuluPaired
			LOOPbifurcationAlphaHasCurves = arrayCurveLocations[LOOPindex, indexBifurcationAlpha] != 1
			LOOPbifurcationZuluHasCurves = arrayCurveLocations[LOOPindex, indexBifurcationZulu] != 1
			LOOPbifurcationAlphaIsEven = not (arrayCurveLocations[LOOPindex, indexBifurcationAlpha] & 1)
			LOOPbifurcationZuluIsEven = not (arrayCurveLocations[LOOPindex, indexBifurcationZulu] & 1)

			LOOPbifurcationAlphaShiftRight2 = arrayCurveLocations[LOOPindex, indexBifurcationAlpha] >> 2
			if LOOPbifurcationAlphaHasCurves and LOOPbifurcationZuluHasCurves and LOOPbifurcationZuluIsEven and not LOOPbifurcationAlphaIsEven:
				LOOPXOrHere2makePair = 0b1
				LOOPfindUnpaired_0b1 = 0
				while LOOPfindUnpaired_0b1 >= 0:
					LOOPXOrHere2makePair <<= 2
					LOOPfindUnpaired_0b1 += 1 if (arrayCurveLocations[LOOPindex, indexBifurcationZulu] & LOOPXOrHere2makePair) == 0 else -1
				LOOPbifurcationZuluPaired = arrayCurveLocations[LOOPindex, indexBifurcationZulu] ^ LOOPXOrHere2makePair

				LOOPcurveLocationAnalysis = ((LOOPbifurcationZuluPaired >> 2) << 1) | LOOPbifurcationAlphaShiftRight2
				if LOOPcurveLocationAnalysis < curveLocationsMAXIMUM:
					LOOPdictionaryCurveLocations[LOOPcurveLocationAnalysis] = LOOPdictionaryCurveLocations.get(LOOPcurveLocationAnalysis, 0) + arrayCurveLocations[LOOPindex, indexDistinctCrossings]

		# Merge and aggregate all arrayCurveLocationsAnalyzed mask arrays
		CONVERTarraysMaskAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]] = []
		if arrayCurveLocationsAnalyzed_Z0Z_simpleBridges.size != 0:
			CONVERTarraysMaskAnalyzed.append(arrayCurveLocationsAnalyzed_Z0Z_simpleBridges)
		if arrayCurveLocationsAnalyzed_bifurcationAlphaCurves.size != 0:
			CONVERTarraysMaskAnalyzed.append(arrayCurveLocationsAnalyzed_bifurcationAlphaCurves)
		if arrayCurveLocationsAnalyzed_bifurcationZuluCurves.size != 0:
			CONVERTarraysMaskAnalyzed.append(arrayCurveLocationsAnalyzed_bifurcationZuluCurves)
		# if arrayCurveLocationsAnalyzed_Z0Z_alignedBridges.size != 0:
		# 	CONVERTarraysMaskAnalyzed.append(arrayCurveLocationsAnalyzed_Z0Z_alignedBridges)

		if CONVERTarraysMaskAnalyzed:
			# Combine all arrays into single array
			CONVERTarrayCurveLocationsAnalyzedMerged: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.vstack(CONVERTarraysMaskAnalyzed)

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
