import numpy

# NOTE `arrayCurveGroups`: Always use semantic index identifiers: Never hardcode the indices.
indexDistinctCrossings: int = 0
indexGroupAlpha: int = 1
indexGroupZulu: int = 2

# NOTE `arrayCurveLocations` and tuples in `listOfCoordinates`: Always use semantic index identifiers: Never hardcode the indices.
indexDistinctCrossings: int = 0
indexCurveLocations: int = 1

groupAlphaLocator: numpy.uint64 = numpy.uint64(0x5555555555555555)
groupZuluLocator: numpy.uint64 = numpy.uint64(0xaaaaaaaaaaaaaaaa)

def _aggregate(arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]) -> tuple[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]], numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]:
	orderByCurveLocation: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.argsort(arrayCurveLocations[:, indexCurveLocations])
	groupStarts: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.concatenate(
		(numpy.array([0], dtype=numpy.int64), numpy.nonzero(numpy.diff(arrayCurveLocations[:, indexCurveLocations][orderByCurveLocation]) != 0)[0] + 1))
	return (arrayCurveLocations[:, indexCurveLocations][orderByCurveLocation][groupStarts]
		, numpy.add.reduceat(arrayCurveLocations[:, indexDistinctCrossings][orderByCurveLocation], groupStarts))

def aggregateAnalyzedCurves(arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	curveLocations, distinctCrossings = _aggregate(arrayCurveLocations)
	return numpy.column_stack((distinctCrossings
					, curveLocations & groupAlphaLocator
					, (curveLocations & groupZuluLocator) >> numpy.uint64(1)
	))

def convertArrayCurveLocations2dictionary(arrayCurveLocationsAnalyzed: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]) -> dict[int, int]:
	curveLocations, distinctCrossings = _aggregate(arrayCurveLocationsAnalyzed)
	dictionaryCurveLocations: dict[int, int] = {}
	for index in range(len(curveLocations)):
		dictionaryCurveLocations[int(curveLocations[index])] = int(distinctCrossings[index])
	return dictionaryCurveLocations

def convertDictionaryCurveLocations2array(dictionaryCurveLocations: dict[int, int]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	arrayKeys: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.fromiter(dictionaryCurveLocations.keys(), dtype=numpy.uint64)
	return numpy.column_stack((
		numpy.fromiter(dictionaryCurveLocations.values(), dtype=numpy.uint64)
		, arrayKeys & groupAlphaLocator
		, (arrayKeys & groupZuluLocator) >> numpy.uint64(1)
	))

def count64(bridges: int, dictionaryCurveLocations: dict[int, int], bridgesMinimum: int = 0) -> tuple[int, dict[int, int]]:
	arrayCurveGroups: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = convertDictionaryCurveLocations2array(dictionaryCurveLocations)
	arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.zeros((0, 2), dtype=arrayCurveGroups.dtype)
	usedRows: int = 0

	while bridges > bridgesMinimum:
		bridges -= 1
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)

		# groupAlphaCurves -----------------------------------------------------------------------------------
		selectGroupAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupAlpha] > numpy.uint64(1)
		curveLocationsGroupAlpha = ((arrayCurveGroups[selectGroupAlphaCurves, indexGroupAlpha] >> numpy.uint64(2))
			| (arrayCurveGroups[selectGroupAlphaCurves, indexGroupZulu] << numpy.uint64(3))
			| ((numpy.uint64(1) - (arrayCurveGroups[selectGroupAlphaCurves, indexGroupAlpha] & numpy.uint64(1))) << numpy.uint64(1))
		)

		selectNonZeroGroupAlpha = numpy.flatnonzero(selectGroupAlphaCurves)[
			numpy.nonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)[0]
		]
		sizeGroupAlpha = numpy.nonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)[0].size

		selectGroupZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupZulu] > numpy.uint64(1)
		curveLocationsGroupZulu = (arrayCurveGroups[selectGroupZuluCurves, indexGroupZulu] >> numpy.uint64(1)
			| arrayCurveGroups[selectGroupZuluCurves, indexGroupAlpha] << numpy.uint64(2)
			| (numpy.uint64(1) - (arrayCurveGroups[selectGroupZuluCurves, indexGroupZulu] & numpy.uint64(1)))
		)

		selectNonZeroGroupZulu = numpy.flatnonzero(selectGroupZuluCurves)[numpy.nonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)[0]]
		sizeGroupZulu = numpy.nonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)[0].size

		# bridgesSimple ----------------------------------------------------------------------------------------------
		selectNonZeroBridgesSimple: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(
			((arrayCurveGroups[:, indexGroupAlpha] << numpy.uint64(2)) | (arrayCurveGroups[:, indexGroupZulu] << numpy.uint64(3)) | numpy.uint64(3)) < curveLocationsMAXIMUM
		)[0]
		sizeBridgesSimple = selectNonZeroBridgesSimple.size

		# Selectors for bridgesAligned -------------------------------------------------
		selectGroupAlphaAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveGroups[:, indexGroupAlpha] & numpy.uint64(1)) == numpy.uint64(0)
		selectGroupZuluAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveGroups[:, indexGroupZulu] & numpy.uint64(1)) == numpy.uint64(0)
		selectorBridgesAligned = selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven | selectGroupZuluAtEven)
		sizeBridgesAligned = arrayCurveGroups[selectorBridgesAligned, indexGroupAlpha].size
		arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.zeros((sizeGroupAlpha + sizeGroupZulu + sizeBridgesSimple + sizeBridgesAligned, 2), dtype=arrayCurveGroups.dtype)
		usedRows: int = 0

		arrayCurveLocations[usedRows:usedRows+sizeGroupAlpha, indexDistinctCrossings] = arrayCurveGroups[selectNonZeroGroupAlpha, indexDistinctCrossings]
		arrayCurveLocations[usedRows:usedRows+sizeGroupAlpha, indexCurveLocations] = curveLocationsGroupAlpha[numpy.nonzero(curveLocationsGroupAlpha < curveLocationsMAXIMUM)[0]]
		usedRows += sizeGroupAlpha

		arrayCurveLocations[usedRows:usedRows+sizeGroupZulu, indexDistinctCrossings] = arrayCurveGroups[selectNonZeroGroupZulu, indexDistinctCrossings]
		arrayCurveLocations[usedRows:usedRows+sizeGroupZulu, indexCurveLocations] = curveLocationsGroupZulu[numpy.nonzero(curveLocationsGroupZulu < curveLocationsMAXIMUM)[0]]
		usedRows += sizeGroupZulu

		arrayCurveLocations[usedRows:usedRows+sizeBridgesSimple, indexDistinctCrossings] = arrayCurveGroups[selectNonZeroBridgesSimple, indexDistinctCrossings]
		arrayCurveLocations[usedRows:usedRows+sizeBridgesSimple, indexCurveLocations] = (
			(arrayCurveGroups[selectNonZeroBridgesSimple, indexGroupAlpha] << numpy.uint64(2))
			| (arrayCurveGroups[selectNonZeroBridgesSimple, indexGroupZulu] << numpy.uint64(3))
			| numpy.uint64(3)
		)
		usedRows += sizeBridgesSimple

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
		curveLocationsBridgesAligned = (((arrayCurveGroups[selectorBridgesAligned, indexGroupZulu] >> numpy.uint64(2)) << numpy.uint64(1))
			| (arrayCurveGroups[selectorBridgesAligned, indexGroupAlpha] >> numpy.uint64(2))
		)

		curveLocationsBridgesAligned[curveLocationsBridgesAligned >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZeroBridgesAligned = numpy.nonzero(curveLocationsBridgesAligned)[0]
		arrayCurveLocations[usedRows:usedRows+selectNonZeroBridgesAligned.size, indexDistinctCrossings] = arrayCurveGroups[selectorBridgesAligned, indexDistinctCrossings][selectNonZeroBridgesAligned]
		arrayCurveLocations[usedRows:usedRows+selectNonZeroBridgesAligned.size, indexCurveLocations] = curveLocationsBridgesAligned[selectNonZeroBridgesAligned]
		usedRows += selectNonZeroBridgesAligned.size

		arrayCurveGroups = aggregateAnalyzedCurves(arrayCurveLocations[:usedRows])

	return (bridges, convertArrayCurveLocations2dictionary(arrayCurveLocations[:usedRows]))

