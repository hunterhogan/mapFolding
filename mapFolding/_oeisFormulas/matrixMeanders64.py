import numpy

"""TODO temporary notes:

Flow:
- mask construction
	- Z0Z_simpleBridges
	- bifurcationAlphaCurves
	- bifurcationZuluCurves
	- Z0Z_alreadyEven
	- Z0Z_alignedBridges
- aggregate curveLocationAnalysis and distinctCrossings
	- Details are unclear
	- each mask generates curveLocationAnalysis, distinctCrossings pairs or indices for the distinctCrossings
	- the curveLocationAnalysis, distinctCrossings data is aggregated
	- distinctCrossings are summed for identical curveLocationsAnalysis
- replace `arrayCurveLocations` with new ndarray

"""

indexDistinctCrossings = int(0)  # noqa: RUF046, UP018
indexBifurcationAlpha = int(1)  # noqa: RUF046, UP018
indexBifurcationZulu = int(2)  # noqa: RUF046, UP018

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:

	# Legacy
	dictionaryCurveLocations: dict[int, int] = {}

	while bridges > 0:
		bridges -= 1

		# Vector conditional
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)

		# Convert startingCurveLocations to ndarray
		bifurcationAlphaLocator: numpy.uint64 = numpy.uint64(0x5555555555555555)
		bifurcationZuluLocator: numpy.uint64 = numpy.uint64(0xaaaaaaaaaaaaaaaa)
		arrayKeys: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.fromiter(startingCurveLocations.keys(), dtype=numpy.uint64)
		arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			numpy.fromiter(startingCurveLocations.values(), dtype=numpy.uint64),
			arrayKeys & bifurcationAlphaLocator,
			(arrayKeys & bifurcationZuluLocator) >> numpy.uint64(1),
		))

		# Z0Z_simpleBridges
		curveLocation_Z0Z_simpleBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[:, indexBifurcationAlpha] | (arrayCurveLocations[:, indexBifurcationZulu] << numpy.uint64(1))) << numpy.uint64(2)) | numpy.uint64(3))

		arrayCurveLocationsAnalyzed_Z0Z_simpleBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[:, indexDistinctCrossings][curveLocation_Z0Z_simpleBridges < curveLocationsMAXIMUM],
			curveLocation_Z0Z_simpleBridges[curveLocation_Z0Z_simpleBridges < curveLocationsMAXIMUM]
		))

		# bifurcationAlphaCurves
		bifurcationAlphaHasCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexBifurcationAlpha] > numpy.uint64(1)

		_bifurcationAlphaShiftRight2 = arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationAlpha] >> numpy.uint64(2)
		_bifurcatedZuluShiftLeft3 = arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationZulu] << numpy.uint64(3)
		_bifurcationAlphaIsEven = (numpy.uint64(1) - (arrayCurveLocations[bifurcationAlphaHasCurves][:, indexBifurcationAlpha] & numpy.uint64(1))) << numpy.uint64(1)
		curveLocation_bifurcationAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			_bifurcationAlphaShiftRight2 | _bifurcatedZuluShiftLeft3 | _bifurcationAlphaIsEven
		)

		arrayCurveLocationsAnalyzed_bifurcationAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[bifurcationAlphaHasCurves][:, indexDistinctCrossings][curveLocation_bifurcationAlphaCurves < curveLocationsMAXIMUM],
			curveLocation_bifurcationAlphaCurves[curveLocation_bifurcationAlphaCurves < curveLocationsMAXIMUM]
		))

		# bifurcationZuluCurves
		bifurcationZuluHasCurves = arrayCurveLocations[:, indexBifurcationZulu] > numpy.uint64(1)

		_bifurcationZuluShiftRight1 = arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationZulu] >> numpy.uint64(1)
		_bifurcationAlphaShiftLeft2 = arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationAlpha] << numpy.uint64(2)
		_bifurcationZuluIsEven = ~ arrayCurveLocations[bifurcationZuluHasCurves][:, indexBifurcationZulu] & numpy.uint64(1)
		curveLocation_bifurcationZuluCurves = _bifurcationZuluShiftRight1 | _bifurcationAlphaShiftLeft2 | _bifurcationZuluIsEven

		arrayCurveLocationsAnalyzed_bifurcationZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.column_stack((
			arrayCurveLocations[bifurcationZuluHasCurves][:, indexDistinctCrossings][curveLocation_bifurcationZuluCurves < curveLocationsMAXIMUM],
			curveLocation_bifurcationZuluCurves[curveLocation_bifurcationZuluCurves < curveLocationsMAXIMUM]
		))

		for index in range(len(arrayCurveLocations)):

			# Z0Z_alignedBridges
			_bifurcationAlphaHasCurves = arrayCurveLocations[index, indexBifurcationAlpha] != 1
			_bifurcationZuluHasCurves = arrayCurveLocations[index, indexBifurcationZulu] != 1
			bifurcationAlphaShiftRight2 = arrayCurveLocations[index, indexBifurcationAlpha] >> 2
			bifurcationAlphaIsEven = 1 - (arrayCurveLocations[index, indexBifurcationAlpha] & 0b1)
			bifurcationZuluIsEven = not (arrayCurveLocations[index, indexBifurcationZulu] & 1)
			if _bifurcationAlphaHasCurves and _bifurcationZuluHasCurves:
				# One Truth-check to select a code path
				bifurcationsCanBePairedTogether = (bifurcationZuluIsEven << 1) | bifurcationAlphaIsEven # pyright: ignore[reportPossiblyUnboundVariable]

				if bifurcationsCanBePairedTogether != 0:
					XOrHere2makePair = 0b1
					findUnpaired_0b1 = 0

					if bifurcationsCanBePairedTogether == 1:
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (arrayCurveLocations[index, indexBifurcationAlpha] & XOrHere2makePair) == 0 else -1
						bifurcationAlphaShiftRight2 = (arrayCurveLocations[index, indexBifurcationAlpha] ^ XOrHere2makePair) >> 2
					elif bifurcationsCanBePairedTogether == 2:
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (arrayCurveLocations[index, indexBifurcationZulu] & XOrHere2makePair) == 0 else -1
						arrayCurveLocations[index, indexBifurcationZulu] ^= XOrHere2makePair

					curveLocationAnalysis = ((arrayCurveLocations[index, indexBifurcationZulu] >> 2) << 1) | bifurcationAlphaShiftRight2 # pyright: ignore[reportPossiblyUnboundVariable]
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + arrayCurveLocations[index, indexDistinctCrossings]

		startingCurveLocations.clear()

		# Merge and aggregate all arrayCurveLocationsAnalyzed mask arrays
		arraysMaskAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]] = []
		if arrayCurveLocationsAnalyzed_Z0Z_simpleBridges.size != 0:
			arraysMaskAnalyzed.append(arrayCurveLocationsAnalyzed_Z0Z_simpleBridges)
		if arrayCurveLocationsAnalyzed_bifurcationAlphaCurves.size != 0:
			arraysMaskAnalyzed.append(arrayCurveLocationsAnalyzed_bifurcationAlphaCurves)
		if arrayCurveLocationsAnalyzed_bifurcationZuluCurves.size != 0:
			arraysMaskAnalyzed.append(arrayCurveLocationsAnalyzed_bifurcationZuluCurves)

		if arraysMaskAnalyzed:
			# Combine all arrays into single array
			arrayCurveLocationsAnalyzedMerged: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.vstack(arraysMaskAnalyzed)

			curveLocationColumn: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocationsAnalyzedMerged[:, 1]
			distinctCrossingsColumn: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocationsAnalyzedMerged[:, 0]

			orderByCurveLocation: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.argsort(curveLocationColumn, kind='mergesort')
			sortedCurveLocation: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = curveLocationColumn[orderByCurveLocation]
			sortedDistinctCrossings: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = distinctCrossingsColumn[orderByCurveLocation]

			indicesWhereKeyChanges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(numpy.diff(sortedCurveLocation) != 0)[0] + 1
			groupStarts: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.concatenate((numpy.array([0], dtype=numpy.int64), indicesWhereKeyChanges))
			segmentSums: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.add.reduceat(sortedDistinctCrossings, groupStarts)
			uniqueCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = sortedCurveLocation[groupStarts]

			for indexGroup in range(uniqueCurveLocations.shape[0]):
				keyCurveLocation = int(uniqueCurveLocations[indexGroup])
				valueDistinctCrossings = int(segmentSums[indexGroup])
				dictionaryCurveLocations[keyCurveLocation] = dictionaryCurveLocations.get(keyCurveLocation, 0) + valueDistinctCrossings

		startingCurveLocations, dictionaryCurveLocations = dictionaryCurveLocations, startingCurveLocations

	return sum(startingCurveLocations.values())
