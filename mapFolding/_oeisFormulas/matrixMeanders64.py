import numpy

distinctCrossings = int(0)
bifurcationAlpha = int(1)
bifurcationZulu = int(2)

"""TODO temporary notes:

Flow:
- mask construction
	- Z0Z_simpleBridges
	- bifurcationAlphaCurves
	- bifurcationZuluCurves
	- Z0Z_alignedBridges
- aggregate curveLocationAnalysis and distinctCrossings
	- Details are unclear
	- each mask generates curveLocationAnalysis, distinctCrossings pairs or indices for the distinctCrossings
	- the curveLocationAnalysis, distinctCrossings data is aggregated
	- distinctCrossings are summed for identical curveLocationsAnalysis
- replace `arrayCurveLocations` with new ndarray

"""

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:
	dictionaryCurveLocations: dict[int, int] = {}

	while bridges > 0:
		bridges -= 1

		curveLocationsMAXIMUM = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)
		bifurcationAlphaLocator = (
			(1 << 2 * ((int(curveLocationsMAXIMUM).bit_length() + 1) // 2)
			) - 1
		) // 3
		bifurcationZuluLocator  = bifurcationAlphaLocator << numpy.uint64(1)

		arrayKeys = numpy.fromiter(startingCurveLocations.keys(), dtype=numpy.uint64)
		arrayCurveLocations = numpy.column_stack((
			numpy.fromiter(startingCurveLocations.values(), dtype=numpy.uint64),
			arrayKeys & bifurcationAlphaLocator,
			(arrayKeys & bifurcationZuluLocator) >> numpy.uint64(1),
		))

		for index in range(len(arrayCurveLocations)):
			bifurcationAlphaHasCurves = arrayCurveLocations[index, bifurcationAlpha] != 1
			bifurcationZuluHasCurves = arrayCurveLocations[index, bifurcationZulu] != 1

			# Z0Z_simpleBridges
			curveLocationAnalysis = ((arrayCurveLocations[index, bifurcationAlpha] | (arrayCurveLocations[index, bifurcationZulu] << 1)) << 2) | 3
			if curveLocationAnalysis < curveLocationsMAXIMUM:
				dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + arrayCurveLocations[index, distinctCrossings]

			# bifurcationAlphaCurves
			if bifurcationAlphaHasCurves:
				curveLocationAnalysis = (bifurcationAlphaShiftRight2 := arrayCurveLocations[index, bifurcationAlpha] >> 2) | (arrayCurveLocations[index, bifurcationZulu] << 3) | ((bifurcationAlphaIsEven := 1 - (arrayCurveLocations[index, bifurcationAlpha] & 0b1)) << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + arrayCurveLocations[index, distinctCrossings]

			# bifurcationZuluCurves
			if bifurcationZuluHasCurves:
				curveLocationAnalysis = (arrayCurveLocations[index, bifurcationZulu] >> 1) | (arrayCurveLocations[index, bifurcationAlpha] << 2) | (bifurcationZuluIsEven := not (arrayCurveLocations[index, bifurcationZulu] & 1))
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + arrayCurveLocations[index, distinctCrossings]

			# Z0Z_alignedBridges
			if bifurcationAlphaHasCurves and bifurcationZuluHasCurves:
				# One Truth-check to select a code path
				bifurcationsCanBePairedTogether = (bifurcationZuluIsEven << 1) | bifurcationAlphaIsEven # pyright: ignore[reportPossiblyUnboundVariable]

				if bifurcationsCanBePairedTogether != 0:
					XOrHere2makePair = 0b1
					findUnpaired_0b1 = 0

					if bifurcationsCanBePairedTogether == 1:
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (arrayCurveLocations[index, bifurcationAlpha] & XOrHere2makePair) == 0 else -1
						bifurcationAlphaShiftRight2 = (arrayCurveLocations[index, bifurcationAlpha] ^ XOrHere2makePair) >> 2
					elif bifurcationsCanBePairedTogether == 2:
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (arrayCurveLocations[index, bifurcationZulu] & XOrHere2makePair) == 0 else -1
						arrayCurveLocations[index, bifurcationZulu] ^= XOrHere2makePair

					curveLocationAnalysis = ((arrayCurveLocations[index, bifurcationZulu] >> 2) << 1) | bifurcationAlphaShiftRight2 # pyright: ignore[reportPossiblyUnboundVariable]
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + arrayCurveLocations[index, distinctCrossings]

		startingCurveLocations.clear()
		startingCurveLocations, dictionaryCurveLocations = dictionaryCurveLocations, startingCurveLocations

	return sum(startingCurveLocations.values())
