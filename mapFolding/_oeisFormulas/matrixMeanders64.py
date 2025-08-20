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

def _aggregate(listArrayCurveLocations: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]) -> tuple[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]], numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]:
	arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.vstack(listArrayCurveLocations)
	orderByCurveLocation: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.argsort(arrayCurveLocations[:, indexCurveLocations])
	groupStarts: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.concatenate(
		(numpy.array([0], dtype=numpy.int64), numpy.nonzero(numpy.diff(arrayCurveLocations[:, indexCurveLocations][orderByCurveLocation]) != 0)[0] + 1))
	return (arrayCurveLocations[:, indexCurveLocations][orderByCurveLocation][groupStarts]
		, numpy.add.reduceat(arrayCurveLocations[:, indexDistinctCrossings][orderByCurveLocation], groupStarts))

def aggregateAnalyzedCurves(listArrayCurveLocations: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	curveLocations, distinctCrossings = _aggregate(listArrayCurveLocations)
	return numpy.column_stack((distinctCrossings
					, curveLocations & groupAlphaLocator
					, (curveLocations & groupZuluLocator) >> numpy.uint64(1)
	))

def convertArrayCurveLocations2dictionary(listArrayCurveLocationsAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]) -> dict[int, int]:
	curveLocations, distinctCrossings = _aggregate(listArrayCurveLocationsAnalyzed)
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
	listArrayCurveLocations: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]] = []

	while bridges > bridgesMinimum:
		bridges -= 1
		listArrayCurveLocations = []
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)

		# Selectors -------------------------------------------------------------------------------------------
		selectGroupAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupAlpha] > numpy.uint64(1)
		selectGroupZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupZulu] > numpy.uint64(1)
		selectGroupAlphaAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveGroups[:, indexGroupAlpha] & numpy.uint64(1)) == numpy.uint64(0)
		selectGroupZuluAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveGroups[:, indexGroupZulu] & numpy.uint64(1)) == numpy.uint64(0)

		# bridgesSimple ----------------------------------------------------------------------------------------------
		curveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveGroups[:, indexGroupAlpha] | (arrayCurveGroups[:, indexGroupZulu] << numpy.uint64(1))) << numpy.uint64(2)) | numpy.uint64(3)
		)

		# NOTE convergent code block, `bridgesSimple` does not have a `selector`.
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(curveLocations)[0]
		listArrayCurveLocations.append(numpy.column_stack((
				arrayCurveGroups[:, indexDistinctCrossings][selectNonZero]
				, curveLocations[selectNonZero])))

		# groupAlphaCurves -----------------------------------------------------------------------------------
		selector: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = selectGroupAlphaCurves
		curveLocations = ((arrayCurveGroups[selector, indexGroupAlpha] >> numpy.uint64(2))
			| (arrayCurveGroups[selector, indexGroupZulu] << numpy.uint64(3))
			| ((numpy.uint64(1) - (arrayCurveGroups[selector, indexGroupAlpha] & numpy.uint64(1))) << numpy.uint64(1))
		)

		# NOTE converged code block
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero = numpy.nonzero(curveLocations)[0]
		listArrayCurveLocations.append(numpy.column_stack((arrayCurveGroups[selector, indexDistinctCrossings][selectNonZero], curveLocations[selectNonZero])))

		# groupZuluCurves -----------------------------------------------------------------------------------
		selector = selectGroupZuluCurves
		curveLocations = (arrayCurveGroups[selector, indexGroupZulu] >> numpy.uint64(1)
			| arrayCurveGroups[selector, indexGroupAlpha] << numpy.uint64(2)
			| (numpy.uint64(1) - (arrayCurveGroups[selector, indexGroupZulu] & numpy.uint64(1)))
		)

		# NOTE converged code block
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero = numpy.nonzero(curveLocations)[0]
		listArrayCurveLocations.append(numpy.column_stack((arrayCurveGroups[selector, indexDistinctCrossings][selectNonZero], curveLocations[selectNonZero])))

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
		selector = selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven | selectGroupZuluAtEven)
		curveLocations = (((arrayCurveGroups[selector, indexGroupZulu] >> numpy.uint64(2)) << numpy.uint64(1))
			| (arrayCurveGroups[selector, indexGroupAlpha] >> numpy.uint64(2))
		)

		# NOTE converged code block
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero = numpy.nonzero(curveLocations)[0]
		listArrayCurveLocations.append(numpy.column_stack((arrayCurveGroups[selector, indexDistinctCrossings][selectNonZero], curveLocations[selectNonZero])))

		arrayCurveGroups = aggregateAnalyzedCurves(listArrayCurveLocations)

	return (bridges, convertArrayCurveLocations2dictionary(listArrayCurveLocations))

