import numpy

"""TODO temporary notes while refactoring from `for` to vectorized operations:

Compute a(n) from n and a pre-populated dictionary.

"n" is the initial number of bridges.

function `count` flow:
- loop by decrementing `bridges` to 0
	- Convert dictionaryCurveLocationsStarting to arrayCurveLocations
		- arrayCurveLocations is used as the source of data
	- Selector construction, data analysis, and data storage
		- listArrayCurveLocationsAnalyzed
			- bridgesSimple
			- groupAlphaCurves
			- groupZuluCurves
			- bridgesAlignedAtEven
		- dictionaryCurveLocationsAnalyzed
			- bridgesGroupAlphaPairedToOdd
			- bridgesGroupZuluPairedToOdd
	- aggregate listArrayCurveLocationsAnalyzed and dictionaryCurveLocationsAnalyzed
		- create a new _dictionary_
		- assigned to dictionaryCurveLocationsStarting
- "sum" dictionaryCurveLocationsStarting.values(), but the dictionary only has one item.
"""

indexDistinctCrossings: int = 0
indexGroupAlpha: int = 1
indexGroupZulu: int = 2

def convert(listArrayCurveLocationsAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]], dictionaryCurveLocationsAnalyzed: dict[int, int]) -> dict[int, int]:
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
		dictionaryCurveLocationsAnalyzed[CONVERTkeyCurveLocation] = dictionaryCurveLocationsAnalyzed.get(CONVERTkeyCurveLocation, 0) + CONVERTvalueDistinctCrossings

	return dictionaryCurveLocationsAnalyzed

def convertStartingCurveLocationsToArray(dictionaryCurveLocationsStarting: dict[int, int]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	"""Convert dictionaryCurveLocationsStarting dict to ndarray for curve location analysis."""
	groupAlphaLocator: numpy.uint64 = numpy.uint64(0x5555555555555555)
	groupZuluLocator: numpy.uint64 = numpy.uint64(0xaaaaaaaaaaaaaaaa)

	arrayKeys: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.fromiter(dictionaryCurveLocationsStarting.keys(), dtype=numpy.uint64)

	return numpy.column_stack((
		numpy.fromiter(dictionaryCurveLocationsStarting.values(), dtype=numpy.uint64)
		, arrayKeys & groupAlphaLocator
		, (arrayKeys & groupZuluLocator) >> numpy.uint64(1)
	))

def count(bridges: int, dictionaryCurveLocationsStarting: dict[int, int]) -> int:

	while bridges > 0:
		# while ------------------------------------------------------------------------------------------------
		bridges -= 1
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)
		listArrayCurveLocationsAnalyzed: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]] = []
		arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = convertStartingCurveLocationsToArray(dictionaryCurveLocationsStarting)

		# Selectors -------------------------------------------------------------------------------------------
		selectGroupAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexGroupAlpha] > numpy.uint64(1)
		selectGroupZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexGroupZulu] > numpy.uint64(1)
		selectGroupAlphaAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexGroupAlpha] & numpy.uint64(1)) == numpy.uint64(0)
		selectGroupZuluAtEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexGroupZulu] & numpy.uint64(1)) == numpy.uint64(0)

		# bridgesSimple ----------------------------------------------------------------------------------------------
		curveLocation_bridgesSimple: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[:, indexGroupAlpha] | (arrayCurveLocations[:, indexGroupZulu] << numpy.uint64(1))) << numpy.uint64(2)) | numpy.uint64(3))
		listArrayCurveLocationsAnalyzed.append(numpy.column_stack((
			arrayCurveLocations[:, indexDistinctCrossings][curveLocation_bridgesSimple < curveLocationsMAXIMUM]
			, curveLocation_bridgesSimple[curveLocation_bridgesSimple < curveLocationsMAXIMUM]
		)))

		# groupAlphaCurves -----------------------------------------------------------------------------------
		selector = selectGroupAlphaCurves
		curveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			(arrayCurveLocations[selector, indexGroupAlpha] >> numpy.uint64(2))
			| (arrayCurveLocations[selector, indexGroupZulu] << numpy.uint64(3))
			| ((numpy.uint64(1) - (arrayCurveLocations[selector, indexGroupAlpha] & numpy.uint64(1))) << numpy.uint64(1))
		)

		# Can this be combined with the step above or the step below?
		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)

		# After zeroes are added to `curveLocations`, it is effectively a boolean selector.
		selectNonZero: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(curveLocations)[0]

		listArrayCurveLocationsAnalyzed.append(numpy.column_stack((
				arrayCurveLocations[selector, indexDistinctCrossings][selectNonZero]
				, curveLocations[selectNonZero]
		)))

		# groupZuluCurves -----------------------------------------------------------------------------------
		selector = selectGroupZuluCurves
		curveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			arrayCurveLocations[selector, indexGroupZulu] >> numpy.uint64(1)
			| arrayCurveLocations[selector, indexGroupAlpha] << numpy.uint64(2)
			| (numpy.uint64(1) - (arrayCurveLocations[selector, indexGroupZulu] & numpy.uint64(1)))
		)

		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(curveLocations)[0]

		listArrayCurveLocationsAnalyzed.append(numpy.column_stack((
				arrayCurveLocations[selector, indexDistinctCrossings][selectNonZero]
				, curveLocations[selectNonZero]
		)))

		# bridgesAlignedAtEven -----------------------------------------------------------------------------------
		selector: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((selectGroupAlphaCurves, selectGroupZuluCurves, selectGroupAlphaAtEven, selectGroupZuluAtEven))
		curveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[selector, indexGroupZulu] >> numpy.uint64(2)) << numpy.uint64(1))
			| (arrayCurveLocations[selector, indexGroupAlpha] >> numpy.uint64(2))
		)

		curveLocations[curveLocations >= curveLocationsMAXIMUM] = numpy.uint64(0)
		selectNonZero: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]] = numpy.nonzero(curveLocations)[0]

		listArrayCurveLocationsAnalyzed.append(numpy.column_stack((
				arrayCurveLocations[selector, indexDistinctCrossings][selectNonZero]
				, curveLocations[selectNonZero]
		)))

		# REFACTOR TARGET start -----------------------------------------------------------------------------------
		# selectGroupAlphaPairedToOdd and selectGroupZuluPairedToOdd -----------------------------------------------------
		dictionaryCurveLocationsAnalyzed: dict[int, int] = {}
		for LOOPindex in numpy.nonzero(selectGroupAlphaCurves & selectGroupZuluCurves & (selectGroupAlphaAtEven ^ selectGroupZuluAtEven))[0].tolist():
			XOrHere2makePair = 0b1
			findUnpaired_0b1 = 0

			if (arrayCurveLocations[LOOPindex, indexGroupAlpha] & 1) == 0:
				indexer = indexGroupAlpha
			else:
				indexer = indexGroupZulu

			while findUnpaired_0b1 >= 0:
				XOrHere2makePair <<= 2
				findUnpaired_0b1 += 1 if (arrayCurveLocations[LOOPindex, indexer] & XOrHere2makePair) == 0 else -1
			arrayCurveLocations[LOOPindex, indexer] ^= XOrHere2makePair

			curveLocationsAnalyzed = ((arrayCurveLocations[LOOPindex, indexGroupZulu] >> 2) << 1) | (arrayCurveLocations[LOOPindex, indexGroupAlpha] >> 2)
			if curveLocationsAnalyzed < curveLocationsMAXIMUM:
				dictionaryCurveLocationsAnalyzed[curveLocationsAnalyzed] = dictionaryCurveLocationsAnalyzed.get(curveLocationsAnalyzed, 0) + arrayCurveLocations[LOOPindex, indexDistinctCrossings]
		# REFACTOR TARGET end -----------------------------------------------------------------------------------

		dictionaryCurveLocationsStarting = convert(listArrayCurveLocationsAnalyzed, dictionaryCurveLocationsAnalyzed)

	return sum(dictionaryCurveLocationsStarting.values())

def initializeA000682(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)

	curveSeed: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveSeed << 1) | curveSeed]

	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveSeed = (curveSeed << 4) | 0b101
		listCurveLocations.append((curveSeed << 1) | curveSeed)

	return dict.fromkeys(listCurveLocations, 1)

def A000682(n: int) -> int:
	return count(n - 1, initializeA000682(n - 1))
