import numpy

# TODO change identifier `arrayCurveLocations`: it doesn't have curve locations.
# NOTE `arrayCurveLocations`: Always use semantic index identifiers: Never hardcode the indices.
indexDistinctCrossings: int = 0
indexGroupAlpha: int = 1
indexGroupZulu: int = 2

# NOTE `arrayCurveLocationsAnalyzed`: Always use semantic index identifiers: Never hardcode the indices.
indexDistinctCrossings: int = 0
indexCurveLocations: int = 1

groupAlphaLocator: numpy.uint64 = numpy.uint64(0x5555555555555555)
groupZuluLocator: numpy.uint64 = numpy.uint64(0xaaaaaaaaaaaaaaaa)

def _aggregate(listArrayCurveLocationsAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]) -> tuple[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]], numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]:
	arrayCurveLocationsAnalyzedMerged: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.vstack(listArrayCurveLocationsAnalyzed)
	curveLocationColumn: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocationsAnalyzedMerged[:, indexCurveLocations]
	distinctCrossingsColumn: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = arrayCurveLocationsAnalyzedMerged[:, indexDistinctCrossings]
	orderByCurveLocation: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.argsort(curveLocationColumn, kind='mergesort')
	sortedCurveLocation: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = curveLocationColumn[orderByCurveLocation]
	sortedDistinctCrossings: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = distinctCrossingsColumn[orderByCurveLocation]
	indicesWhereKeyChanges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(numpy.diff(sortedCurveLocation) != 0)[0] + 1
	groupStarts: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.concatenate((numpy.array([0], dtype=numpy.int64), indicesWhereKeyChanges))
	segmentSums: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.add.reduceat(sortedDistinctCrossings, groupStarts)
	uniqueCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = sortedCurveLocation[groupStarts]
	return (uniqueCurveLocations, segmentSums)

def aggregateAnalyzedCurves(listArrayCurveLocationsAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	uniqueCurveLocations, segmentSums = _aggregate(listArrayCurveLocationsAnalyzed)

	groupAlpha: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = uniqueCurveLocations & groupAlphaLocator
	groupZulu: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (uniqueCurveLocations & groupZuluLocator) >> numpy.uint64(1)
	return numpy.column_stack((segmentSums, groupAlpha, groupZulu))

def convertArrayCurveLocations2dictionary(listArrayCurveLocationsAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]) -> dict[int, int]:
	uniqueCurveLocations, segmentSums = _aggregate(listArrayCurveLocationsAnalyzed)
	dictionaryCurveLocations: dict[int, int] = {}
	for index in range(len(uniqueCurveLocations)):
		dictionaryCurveLocations[int(uniqueCurveLocations[index])] = int(segmentSums[index])
	return dictionaryCurveLocations

def convertDictionaryCurveLocations2array(dictionaryCurveLocations: dict[int, int]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	arrayKeys: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.fromiter(dictionaryCurveLocations.keys(), dtype=numpy.uint64)

	return numpy.column_stack((
		numpy.fromiter(dictionaryCurveLocations.values(), dtype=numpy.uint64)
		, arrayKeys & groupAlphaLocator
		, (arrayKeys & groupZuluLocator) >> numpy.uint64(1)
	))

def count64(bridges: int, dictionaryCurveLocations: dict[int, int], bridgesMinimum: int = 0) -> tuple[int, dict[int, int]]:
	arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = convertDictionaryCurveLocations2array(dictionaryCurveLocations)
	listArrayCurveLocationsAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]] = []

	while bridges > bridgesMinimum:
		bridges -= 1
		listArrayCurveLocationsAnalyzed = []
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)

		# Selectors -------------------------------------------------------------------------------------------
		selectGroupAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexGroupAlpha] > numpy.uint64(1)
		selectGroupZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexGroupZulu] > numpy.uint64(1)
		selectGroupAlphaAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexGroupAlpha] & numpy.uint64(1)) == numpy.uint64(0)
		selectGroupZuluAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexGroupZulu] & numpy.uint64(1)) == numpy.uint64(0)

		# bridgesSimple ----------------------------------------------------------------------------------------------
		curveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[:, indexGroupAlpha] | (arrayCurveLocations[:, indexGroupZulu] << numpy.uint64(1))) << numpy.uint64(2)) | numpy.uint64(3)
		)

		# NOTE convergent code block, `bridgesSimple` does not have a `selector`.
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(curveLocations)[0]
		listArrayCurveLocationsAnalyzed.append(numpy.column_stack((
				arrayCurveLocations[:, indexDistinctCrossings][selectNonZero]
				, curveLocations[selectNonZero])))

		# groupAlphaCurves -----------------------------------------------------------------------------------
		selector: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = selectGroupAlphaCurves
		curveLocations = ((arrayCurveLocations[selector, indexGroupAlpha] >> numpy.uint64(2))
			| (arrayCurveLocations[selector, indexGroupZulu] << numpy.uint64(3))
			| ((numpy.uint64(1) - (arrayCurveLocations[selector, indexGroupAlpha] & numpy.uint64(1))) << numpy.uint64(1))
		)

		# NOTE converged code block
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero = numpy.nonzero(curveLocations)[0]
		listArrayCurveLocationsAnalyzed.append(numpy.column_stack((arrayCurveLocations[selector, indexDistinctCrossings][selectNonZero], curveLocations[selectNonZero])))

		# groupZuluCurves -----------------------------------------------------------------------------------
		selector = selectGroupZuluCurves
		curveLocations = (arrayCurveLocations[selector, indexGroupZulu] >> numpy.uint64(1)
			| arrayCurveLocations[selector, indexGroupAlpha] << numpy.uint64(2)
			| (numpy.uint64(1) - (arrayCurveLocations[selector, indexGroupZulu] & numpy.uint64(1)))
		)

		# NOTE converged code block
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero = numpy.nonzero(curveLocations)[0]
		listArrayCurveLocationsAnalyzed.append(numpy.column_stack((arrayCurveLocations[selector, indexDistinctCrossings][selectNonZero], curveLocations[selectNonZero])))

		# NOTE this MODIFIES `arrayCurveLocations` for bridgesPairedToOdd ---------------------------------------------------------------------------------------
		for indexRow in numpy.nonzero(selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven ^ selectGroupZuluAtEven))[0].tolist():
			if (arrayCurveLocations[indexRow, indexGroupAlpha] & 1) == 0:
				indexGroupToModify: int = indexGroupAlpha
			else:
				indexGroupToModify = indexGroupZulu

			XOrHere2makePair = 0b1
			findUnpaired_0b1 = 0

			while findUnpaired_0b1 >= 0:
				XOrHere2makePair <<= 2
				findUnpaired_0b1 += 1 if (arrayCurveLocations[indexRow, indexGroupToModify] & XOrHere2makePair) == 0 else -1

			arrayCurveLocations[indexRow, indexGroupToModify] ^= XOrHere2makePair

		# bridgesAligned; bridgesAlignedAtEven, bridgesGroupAlphaPairedToOdd, bridgesGroupZuluPairedToOdd -----------------
		selector = selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven | selectGroupZuluAtEven)
		curveLocations = (((arrayCurveLocations[selector, indexGroupZulu] >> numpy.uint64(2)) << numpy.uint64(1))
			| (arrayCurveLocations[selector, indexGroupAlpha] >> numpy.uint64(2))
		)

		# NOTE converged code block
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero = numpy.nonzero(curveLocations)[0]
		listArrayCurveLocationsAnalyzed.append(numpy.column_stack((arrayCurveLocations[selector, indexDistinctCrossings][selectNonZero], curveLocations[selectNonZero])))

		arrayCurveLocations = aggregateAnalyzedCurves(listArrayCurveLocationsAnalyzed)

	return (bridges, convertArrayCurveLocations2dictionary(listArrayCurveLocationsAnalyzed))

