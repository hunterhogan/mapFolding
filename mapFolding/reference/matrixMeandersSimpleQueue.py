from mapFolding._oeisFormulas.matrixMeandersAnnex import curveMaximum as curveMaximum
from queue import Empty, SimpleQueue
from typing import NamedTuple
import contextlib

class BifurcatedCurves(NamedTuple):
    bifurcationZulu: int
    bifurcationAlpha: int
    distinctCrossings: int
    curveLocationsMAXIMUM: int

dictionaryCurveLocations: dict[int, int] = {}
simpleQueueCurveLocations: SimpleQueue[tuple[int, int]] = SimpleQueue()

def unpackQueue() -> dict[int, int]:
	with contextlib.suppress(Empty):
		while True:
			curveLocations, distinctCrossings = simpleQueueCurveLocations.get_nowait()
			dictionaryCurveLocations[curveLocations] = dictionaryCurveLocations.get(curveLocations, 0) + distinctCrossings

	return dictionaryCurveLocations

def getCurveLocations(bridges: int) -> list[BifurcatedCurves]:
	global dictionaryCurveLocations  # noqa: PLW0603
	dictionaryCurveLocations = unpackQueue()
	curveLocationsMAXIMUM, bifurcationZuluLocator, bifurcationAlphaLocator = curveMaximum[bridges]
	listBifurcatedCurves: list[BifurcatedCurves] = []
	# TODO This is ready for concurrency and/or vectorization.
	for curveLocations, distinctCrossings in dictionaryCurveLocations.items():
		bifurcationZulu = (curveLocations & bifurcationZuluLocator) >> 1
		bifurcationAlpha = (curveLocations & bifurcationAlphaLocator)
		listBifurcatedCurves.append(BifurcatedCurves(bifurcationZulu, bifurcationAlpha, distinctCrossings, curveLocationsMAXIMUM))
	dictionaryCurveLocations = {}
	return listBifurcatedCurves

def recordAnalysis(curveLocationAnalysis: int, curveLocationsMAXIMUM: int, distinctCrossings: int) -> None:
	if curveLocationAnalysis < curveLocationsMAXIMUM:
		simpleQueueCurveLocations.put((curveLocationAnalysis, distinctCrossings))

def analyzeCurve(bifurcationZulu: int, bifurcationAlpha: int, distinctCrossings: int, curveLocationsMAXIMUM: int) -> None:
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

def initializeCurveLocations(startingCurveLocations: dict[int, int]) -> None:
	global dictionaryCurveLocations  # noqa: PLW0603
	dictionaryCurveLocations = startingCurveLocations.copy()

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:
	initializeCurveLocations(startingCurveLocations)

	while bridges > 0:
		bridges -= 1

		# TODO This could be parallelized when `recordAnalysis` is thread-safe
		for bifurcatedCurve in getCurveLocations(bridges):
			analyzeCurve(*bifurcatedCurve)

	return getCurveLocations(bridges)[0].distinctCrossings
