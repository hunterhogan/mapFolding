bifurcationOddLocator = 0x55555555555555555555555555555555
bitWidth = 1 << (bifurcationOddLocator.bit_length() - 1).bit_length()

def count(bridges: int, dictionaryCurveLocationsKnown: dict[int, int]) -> int:
	while bridges > 0:
		bridges -= 1
		curveLocationsMAXIMUM = 1 << (2 + (2 * (bridges + 1)))

		dictionaryCurveLocationsDiscovered: dict[int, int] = {}

		for curveLocations, distinctCrossings in dictionaryCurveLocationsKnown.items():
			global bifurcationOddLocator, bitWidth  # noqa: PLW0603

			if curveLocations > bifurcationOddLocator:
				while curveLocations > bifurcationOddLocator:
					bifurcationOddLocator |= bifurcationOddLocator << bitWidth
					bitWidth <<= 1

			bifurcationOdd = curveLocations & bifurcationOddLocator
			bifurcationEven = (curveLocations ^ bifurcationOdd) >> 1

			bifurcationOddHasCurves = bifurcationOdd != 1
			bifurcationEvenHasCurves = bifurcationEven != 1
			bifurcationOddFinalZero = (bifurcationOdd & 1) == 0
			bifurcationEvenFinalZero = (bifurcationEven & 1) == 0

			if bifurcationOddHasCurves:
				curveLocationAnalysis = (bifurcationOdd >> 2) | (bifurcationEven << 3) | (bifurcationOddFinalZero << 1)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			if bifurcationEvenHasCurves:
				curveLocationAnalysis = (bifurcationEven >> 1) | ((bifurcationOdd << 2) | bifurcationEvenFinalZero)
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			curveLocationAnalysis = ((bifurcationOdd | (bifurcationEven << 1)) << 2) | 3
			if curveLocationAnalysis < curveLocationsMAXIMUM:
				dictionaryCurveLocationsDiscovered[curveLocationAnalysis] = dictionaryCurveLocationsDiscovered.get(curveLocationAnalysis, 0) + distinctCrossings

			if bifurcationOddHasCurves and bifurcationEvenHasCurves and (bifurcationOddFinalZero or bifurcationEvenFinalZero):
				if bifurcationOddFinalZero and not bifurcationEvenFinalZero:
					Z0Z_idk = 0
					Z0Z_indexIDK = 1
					while Z0Z_idk >= 0:
						Z0Z_indexIDK <<= 2
						Z0Z_idk += 1 if (bifurcationOdd & Z0Z_indexIDK) == 0 else -1
					bifurcationOdd ^= Z0Z_indexIDK

				if bifurcationEvenFinalZero and not bifurcationOddFinalZero:
					Z0Z_idk = 0
					Z0Z_indexIDK = 1
					while Z0Z_idk >= 0:
						Z0Z_indexIDK <<= 2
						Z0Z_idk += 1 if (bifurcationEven & Z0Z_indexIDK) == 0 else -1
					bifurcationEven ^= Z0Z_indexIDK

				curveLocationAnalysis = (bifurcationOdd >> 2) | ((bifurcationEven >> 2) << 1)
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
