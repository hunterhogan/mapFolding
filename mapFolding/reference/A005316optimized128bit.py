def count(bridges: int, dictionaryCurveLocationsKnown: dict[int, int]) -> int:
	while bridges > 0:
		bridges -= 1
		curveLocationsMAXIMUM = 1 << (2 * bridges + 4)
		dictionaryCurveLocationsDiscovered: dict[int, int] = {}

		for curveLocations, distinctCrossings in dictionaryCurveLocationsKnown.items():
			bifurcationAlpha = curveLocations & 0x5555555555555555555555555555555555555555555555555555555555555555
			bifurcationZulu = (curveLocations ^ bifurcationAlpha) >> 1

			bifurcationAlphaHasCurves = bifurcationAlpha != 1
			bifurcationZuluHasCurves = bifurcationZulu != 1
			bifurcationAlphaFinalZero = not bifurcationAlpha & 1
			bifurcationZuluFinalZero = not bifurcationZulu & 1

			if bifurcationAlphaHasCurves:
				curveLocationAnalysis = (bifurcationAlpha >> 2) | (bifurcationZulu << 3) | (bifurcationAlphaFinalZero << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			if bifurcationZuluHasCurves:
				curveLocationAnalysis = (bifurcationZulu >> 1) | (bifurcationAlpha << 2) | bifurcationZuluFinalZero
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			curveLocationAnalysis = ((bifurcationAlpha | (bifurcationZulu << 1)) << 2) | 3
			if curveLocationAnalysis < curveLocationsMAXIMUM:
				dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			if bifurcationAlphaHasCurves and bifurcationZuluHasCurves and (bifurcationAlphaFinalZero or bifurcationZuluFinalZero):
				XOrHere2makePair = 0b1
				findUnpairedBinary1 = 0
				if bifurcationAlphaFinalZero and not bifurcationZuluFinalZero:
					while findUnpairedBinary1 >= 0:
						XOrHere2makePair <<= 2
						findUnpairedBinary1 += 1 if (bifurcationAlpha & XOrHere2makePair) == 0 else -1
					bifurcationAlpha ^= XOrHere2makePair

				elif bifurcationZuluFinalZero and not bifurcationAlphaFinalZero:
					while findUnpairedBinary1 >= 0:
						XOrHere2makePair <<= 2
						findUnpairedBinary1 += 1 if (bifurcationZulu & XOrHere2makePair) == 0 else -1
					bifurcationZulu ^= XOrHere2makePair

				curveLocationAnalysis = (bifurcationAlpha >> 2) | ((bifurcationZulu >> 2) << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

		dictionaryCurveLocationsKnown = dictionaryCurveLocationsDiscovered

	return sum(dictionaryCurveLocationsKnown.values())

def initializeA005316(n: int) -> dict[int, int]:
	if n & 1:
		return {22: 1}
	else:
		return {15: 1}

def initializeA000682(n: int) -> dict[int, int]:
	stateToCount: dict[int, int] = {}

	curveLocationsMAXIMUM = 1 << (2 * n + 4)

	bitPattern = 5 - (n & 1) * 4

	packedState = bitPattern | (bitPattern << 1)
	while packedState < curveLocationsMAXIMUM:
		stateToCount[packedState] = 1
		bitPattern = ((bitPattern << 4) | 0b0101)
		packedState = bitPattern | (bitPattern << 1)

	return stateToCount

def A005316(n: int) -> int:
	return count(n, initializeA005316(n))

def A000682(n: int) -> int:
	return count(n - 1, initializeA000682(n - 1))

