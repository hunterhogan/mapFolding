from mapFolding._oeisFormulas.matrixMeandersAnnex import curveMaximum as curveMaximum
from typing import NamedTuple

class BifurcatedCurves(NamedTuple):
    bifurcationEven: int
    bifurcationOdd: int
    distinctCrossings: int
    curveLocationsMAXIMUM: int

dictionaryCurveLocations: dict[int, list[int]] = {}

def getCurveLocations(bridges: int) -> list[BifurcatedCurves]:
	global dictionaryCurveLocations  # noqa: PLW0603
	curveLocationsMAXIMUM, bifurcationEvenLocator, bifurcationOddLocator = curveMaximum[bridges]
	listBifurcatedCurves: list[BifurcatedCurves] = []
	# TODO This is ready for concurrency and/or vectorization.
	for curveLocations, listDistinctCrossings in dictionaryCurveLocations.items():
		bifurcationEven = (curveLocations & bifurcationEvenLocator) >> 1
		bifurcationOdd = (curveLocations & bifurcationOddLocator)
		distinctCrossings = sum(listDistinctCrossings)
		listBifurcatedCurves.append(BifurcatedCurves(bifurcationEven, bifurcationOdd, distinctCrossings, curveLocationsMAXIMUM))
	dictionaryCurveLocations = {}
	return listBifurcatedCurves

def recordAnalysis(curveLocationAnalysis: int, curveLocationsMAXIMUM: int, distinctCrossings: int) -> None:
	if curveLocationAnalysis < curveLocationsMAXIMUM:
		dictionaryCurveLocations.setdefault(curveLocationAnalysis, []).append(distinctCrossings)

def analyzeCurve(bifurcationEven: int, bifurcationOdd: int, distinctCrossings: int, curveLocationsMAXIMUM: int) -> None:
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

def initializeCurveLocations(startingCurveLocations: dict[int, int]) -> None:
	global dictionaryCurveLocations  # noqa: PLW0603
	dictionaryCurveLocations = {curve: [distinctCrossings] for curve, distinctCrossings in startingCurveLocations.items()}

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:
	initializeCurveLocations(startingCurveLocations)

	while bridges > 0:
		bridges -= 1

		# TODO This could be parallelized when `recordAnalysis` is thread-safe
		for bifurcatedCurve in getCurveLocations(bridges):
			analyzeCurve(*bifurcatedCurve)

	return getCurveLocations(bridges)[0].distinctCrossings
