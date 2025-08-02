from mapFolding._oeisFormulas.matrixMeandersAnnex import curveMaximum as curveMaximum

dictionaryCurveLocations: dict[int, int] = {}

def getCurveLocations() -> dict[int, int]:
	global dictionaryCurveLocations  # noqa: PLW0603
	sherpa = dictionaryCurveLocations.copy()
	dictionaryCurveLocations = {}
	return sherpa

def recordAnalysis(curveLocationAnalysis: int, curveLocationsMAXIMUM: int, distinctCrossings: int) -> None:
	if curveLocationAnalysis < curveLocationsMAXIMUM:
		dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

def initializeCurveLocations(startingCurveLocations: dict[int, int]) -> None:
	global dictionaryCurveLocations  # noqa: PLW0603
	dictionaryCurveLocations = startingCurveLocations.copy()

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:
	initializeCurveLocations(startingCurveLocations)

	while bridges > 0:
		bridges -= 1
		curveLocationsMAXIMUM, bifurcationEvenLocator, bifurcationOddLocator = curveMaximum[bridges]

		for curveLocations, distinctCrossings in getCurveLocations().items():
			bifurcationEven = (curveLocations & bifurcationEvenLocator) >> 1
			bifurcationOdd = curveLocations & bifurcationOddLocator

			bifurcationEvenFinalZero = (bifurcationEven & 0b1) == 0
			bifurcationEvenHasCurves = bifurcationEven != 1
			bifurcationOddFinalZero = (bifurcationOdd & 0b1) == 0
			bifurcationOddHasCurves = bifurcationOdd != 1

			if bifurcationEvenHasCurves:
				curveLocationAnalysis = (bifurcationEven >> 1) | (bifurcationOdd << 2) | bifurcationEvenFinalZero
				recordAnalysis(curveLocationAnalysis, curveLocationsMAXIMUM, distinctCrossings)

			if bifurcationOddHasCurves:
				curveLocationAnalysis = (bifurcationOdd >> 2) | (bifurcationEven << 3) | (bifurcationOddFinalZero << 1)
				recordAnalysis(curveLocationAnalysis, curveLocationsMAXIMUM, distinctCrossings)

			curveLocationAnalysis = ((bifurcationOdd | (bifurcationEven << 1)) << 2) | 3
			recordAnalysis(curveLocationAnalysis, curveLocationsMAXIMUM, distinctCrossings)

			if bifurcationEvenHasCurves and bifurcationOddHasCurves and (bifurcationEvenFinalZero or bifurcationOddFinalZero):
				XOrHere2makePair = 0b1
				findUnpairedBinary1 = 0

				if bifurcationEvenFinalZero and not bifurcationOddFinalZero:
					while findUnpairedBinary1 >= 0:
						XOrHere2makePair <<= 2
						findUnpairedBinary1 += 1 if (bifurcationEven & XOrHere2makePair) == 0 else -1
					bifurcationEven ^= XOrHere2makePair

				elif bifurcationOddFinalZero and not bifurcationEvenFinalZero:
					while findUnpairedBinary1 >= 0:
						XOrHere2makePair <<= 2
						findUnpairedBinary1 += 1 if (bifurcationOdd & XOrHere2makePair) == 0 else -1
					bifurcationOdd ^= XOrHere2makePair

				curveLocationAnalysis = ((bifurcationEven >> 2) << 1) | (bifurcationOdd >> 2)
				recordAnalysis(curveLocationAnalysis, curveLocationsMAXIMUM, distinctCrossings)

	return sum(getCurveLocations().values())
