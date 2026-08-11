bifurcationAlphaLocator = 0x55555555555555555555555555555555
bitWidth = 1 << (bifurcationAlphaLocator.bit_length() - 1).bit_length()

def count(bridges: int, dictionaryCurveLocationsKnown: dict[int, int]) -> int:
	while bridges > 0:
		bridges -= 1
		curveLocationsMAXIMUM = 1 << (2 + (2 * (bridges + 1)))

		dictionaryCurveLocationsDiscovered: dict[int, int] = {}

		for curveLocations, distinctCrossings in dictionaryCurveLocationsKnown.items():
			global bifurcationAlphaLocator, bitWidth  # noqa: PLW0603

			if curveLocations > bifurcationAlphaLocator:
				while curveLocations > bifurcationAlphaLocator:
					bifurcationAlphaLocator |= bifurcationAlphaLocator << bitWidth
					bitWidth <<= 1

			bifurcationAlpha = curveLocations & bifurcationAlphaLocator
			bifurcationZulu = (curveLocations ^ bifurcationAlpha) >> 1

			bifurcationAlphaHasCurves = bifurcationAlpha != 1
			bifurcationZuluHasCurves = bifurcationZulu != 1
			bifurcationAlphaFinalZero = (bifurcationAlpha & 1) == 0
			bifurcationZuluFinalZero = (bifurcationZulu & 1) == 0

			if bifurcationAlphaHasCurves:
				curveLocationAnalysis = (bifurcationAlpha >> 2) | (bifurcationZulu << 3) | (bifurcationAlphaFinalZero << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			if bifurcationZuluHasCurves:
				curveLocationAnalysis = (bifurcationZulu >> 1) | ((bifurcationAlpha << 2) | bifurcationZuluFinalZero)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			curveLocationAnalysis = ((bifurcationAlpha | (bifurcationZulu << 1)) << 2) | 3
			if curveLocationAnalysis < curveLocationsMAXIMUM:
				dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			if bifurcationAlphaHasCurves and bifurcationZuluHasCurves and (bifurcationAlphaFinalZero or bifurcationZuluFinalZero):
				if bifurcationAlphaFinalZero and not bifurcationZuluFinalZero:
					Z0Z_idk = 0
					Z0Z_indexIDK = 1
					while Z0Z_idk >= 0:
						Z0Z_indexIDK <<= 2
						Z0Z_idk += 1 if (bifurcationAlpha & Z0Z_indexIDK) == 0 else -1
					bifurcationAlpha ^= Z0Z_indexIDK

				if bifurcationZuluFinalZero and not bifurcationAlphaFinalZero:
					Z0Z_idk = 0
					Z0Z_indexIDK = 1
					while Z0Z_idk >= 0:
						Z0Z_indexIDK <<= 2
						Z0Z_idk += 1 if (bifurcationZulu & Z0Z_indexIDK) == 0 else -1
					bifurcationZulu ^= Z0Z_indexIDK

				curveLocationAnalysis = (bifurcationAlpha >> 2) | ((bifurcationZulu >> 2) << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

		dictionaryCurveLocationsKnown = dictionaryCurveLocationsDiscovered

	return sum(dictionaryCurveLocationsKnown.values())

def initializeA005316(n: int) -> dict[int, int]:
	bridgesTotalIsOdd = (n & 1) == 1
	if bridgesTotalIsOdd:
		arrayBitPattern = (1 << 2) | 1
		arrayBitPattern <<= 2
		initialState = arrayBitPattern | 1 << 1
		return {initialState: 1}
	else:
		arrayBitPattern = (1 << 2) | 1
		initialState = arrayBitPattern | arrayBitPattern << 1
		return {initialState: 1}

def initializeA000682(n: int) -> dict[int, int]:
	bridgesTotalIsOdd = (n & 1) == 1
	archStateLimit = 1 << (2 + (2 * (n + 1)))

	dictionaryStateToTotal: dict[int, int] = {}
	arrayBitPattern = 1 if bridgesTotalIsOdd else ((1 << 2) | 1)

	arrayPackedState = arrayBitPattern | arrayBitPattern << 1
	while arrayPackedState < archStateLimit:
		dictionaryStateToTotal[arrayPackedState] = 1
		arrayBitPattern = ((arrayBitPattern << 2) | 1) << 2 | 1
		arrayPackedState = arrayBitPattern | arrayBitPattern << 1

	return dictionaryStateToTotal

def A005316(n: int) -> int:
	return count(n, initializeA005316(n))

def A000682(n: int) -> int:
	return count(n - 1, initializeA000682(n - 1))
