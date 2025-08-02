def count(bridges: int, dictionaryCurveLocationsKnown: dict[int, int]) -> int:
	while bridges > 0:
		bridges -= 1
		curveLocationsMAXIMUM = 1 << (2 * bridges + 4)
		dictionaryCurveLocationsDiscovered: dict[int, int] = {}

		for curveLocations, distinctCrossings in dictionaryCurveLocationsKnown.items():
			bifurcationOdd = curveLocations & 0x5555555555555555555555555555555555555555555555555555555555555555
			bifurcationEven = (curveLocations ^ bifurcationOdd) >> 1

			bifurcationOddHasCurves = bifurcationOdd != 1
			bifurcationEvenHasCurves = bifurcationEven != 1
			bifurcationOddFinalZero = not bifurcationOdd & 1
			bifurcationEvenFinalZero = not bifurcationEven & 1

			if bifurcationOddHasCurves:
				curveLocationAnalysis = (bifurcationOdd >> 2) | (bifurcationEven << 3) | (bifurcationOddFinalZero << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			if bifurcationEvenHasCurves:
				curveLocationAnalysis = (bifurcationEven >> 1) | (bifurcationOdd << 2) | bifurcationEvenFinalZero
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			curveLocationAnalysis = ((bifurcationOdd | (bifurcationEven << 1)) << 2) | 3
			if curveLocationAnalysis < curveLocationsMAXIMUM:
				dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			if bifurcationOddHasCurves and bifurcationEvenHasCurves and (bifurcationOddFinalZero or bifurcationEvenFinalZero):
				XOrHere2makePair = 0b1
				findUnpairedBinary1 = 0
				if bifurcationOddFinalZero and not bifurcationEvenFinalZero:
					while findUnpairedBinary1 >= 0:
						XOrHere2makePair <<= 2
						findUnpairedBinary1 += 1 if (bifurcationOdd & XOrHere2makePair) == 0 else -1
					bifurcationOdd ^= XOrHere2makePair

				elif bifurcationEvenFinalZero and not bifurcationOddFinalZero:
					while findUnpairedBinary1 >= 0:
						XOrHere2makePair <<= 2
						findUnpairedBinary1 += 1 if (bifurcationEven & XOrHere2makePair) == 0 else -1
					bifurcationEven ^= XOrHere2makePair

				curveLocationAnalysis = (bifurcationOdd >> 2) | ((bifurcationEven >> 2) << 1)
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

