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
		curveLocationsMAXIMUM, bifurcationZuluLocator, bifurcationAlphaLocator = curveMaximum[bridges]

		for curveLocations, distinctCrossings in getCurveLocations().items():
			bifurcationZulu = (curveLocations & bifurcationZuluLocator) >> 1
			bifurcationAlpha = curveLocations & bifurcationAlphaLocator

			bifurcationZuluFinalZero = (bifurcationZulu & 0b1) == 0
			bifurcationZuluHasCurves = bifurcationZulu != 1
			bifurcationAlphaFinalZero = (bifurcationAlpha & 0b1) == 0
			bifurcationAlphaHasCurves = bifurcationAlpha != 1

			if bifurcationZuluHasCurves:
				curveLocationAnalysis = (bifurcationZulu >> 1) | (bifurcationAlpha << 2) | bifurcationZuluFinalZero
				recordAnalysis(curveLocationAnalysis, curveLocationsMAXIMUM, distinctCrossings)

			if bifurcationAlphaHasCurves:
				curveLocationAnalysis = (bifurcationAlpha >> 2) | (bifurcationZulu << 3) | (bifurcationAlphaFinalZero << 1)
				recordAnalysis(curveLocationAnalysis, curveLocationsMAXIMUM, distinctCrossings)

			curveLocationAnalysis = ((bifurcationAlpha | (bifurcationZulu << 1)) << 2) | 3
			recordAnalysis(curveLocationAnalysis, curveLocationsMAXIMUM, distinctCrossings)

			if bifurcationZuluHasCurves and bifurcationAlphaHasCurves and (bifurcationZuluFinalZero or bifurcationAlphaFinalZero):
				XOrHere2makePair = 0b1
				findUnpairedBinary1 = 0

				if bifurcationZuluFinalZero and not bifurcationAlphaFinalZero:
					while findUnpairedBinary1 >= 0:
						XOrHere2makePair <<= 2
						findUnpairedBinary1 += 1 if (bifurcationZulu & XOrHere2makePair) == 0 else -1
					bifurcationZulu ^= XOrHere2makePair

				elif bifurcationAlphaFinalZero and not bifurcationZuluFinalZero:
					while findUnpairedBinary1 >= 0:
						XOrHere2makePair <<= 2
						findUnpairedBinary1 += 1 if (bifurcationAlpha & XOrHere2makePair) == 0 else -1
					bifurcationAlpha ^= XOrHere2makePair

				curveLocationAnalysis = ((bifurcationZulu >> 2) << 1) | (bifurcationAlpha >> 2)
				recordAnalysis(curveLocationAnalysis, curveLocationsMAXIMUM, distinctCrossings)

	return sum(getCurveLocations().values())
