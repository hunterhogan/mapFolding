import numpy

"""TODO temporary notes:

Flow:
- startingCurveLocations to ndarray, `arrayCurveLocations`
	- curveLocations
	- distinctCrossings
	- bifurcationAlpha
	- bifurcationZulu
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
		curveLocationsMAXIMUM = 1 << (2 * bridges + 4)
		bifurcationAlphaLocator = int('01' * ((curveLocationsMAXIMUM.bit_length() + 1) // 2), 2)
		bifurcationZuluLocator = bifurcationAlphaLocator << 1

		for curveLocations, distinctCrossings in startingCurveLocations.items():
			bifurcationAlpha = (curveLocations & bifurcationAlphaLocator)
			bifurcationZulu = (curveLocations & bifurcationZuluLocator) >> 1

			bifurcationAlphaHasCurves = bifurcationAlpha != 1
			bifurcationZuluHasCurves = bifurcationZulu != 1

			# Z0Z_simpleBridges
			curveLocationAnalysis = ((bifurcationAlpha | (bifurcationZulu << 1)) << 2) | 3
			if curveLocationAnalysis < curveLocationsMAXIMUM:
				dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			# bifurcationAlphaCurves
			if bifurcationAlphaHasCurves:
				curveLocationAnalysis = (bifurcationAlphaShiftRight2 := bifurcationAlpha >> 2) | (bifurcationZulu << 3) | ((bifurcationAlphaIsEven := 1 - (bifurcationAlpha & 0b1)) << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			# bifurcationZuluCurves
			if bifurcationZuluHasCurves:
				curveLocationAnalysis = (bifurcationZulu >> 1) | (bifurcationAlpha << 2) | (bifurcationZuluIsEven := not (bifurcationZulu & 1))
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

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
							findUnpaired_0b1 += 1 if (bifurcationAlpha & XOrHere2makePair) == 0 else -1
						bifurcationAlphaShiftRight2 = (bifurcationAlpha ^ XOrHere2makePair) >> 2
					elif bifurcationsCanBePairedTogether == 2:
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (bifurcationZulu & XOrHere2makePair) == 0 else -1
						bifurcationZulu ^= XOrHere2makePair

					curveLocationAnalysis = ((bifurcationZulu >> 2) << 1) | bifurcationAlphaShiftRight2 # pyright: ignore[reportPossiblyUnboundVariable]
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

		startingCurveLocations.clear()
		startingCurveLocations, dictionaryCurveLocations = dictionaryCurveLocations, startingCurveLocations

	return sum(startingCurveLocations.values())
