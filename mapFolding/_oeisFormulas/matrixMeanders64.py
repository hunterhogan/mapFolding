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

listCurveMaximums will be disassembled
- bifurcationAlphaLocator: computing ndarray
- bifurcationZuluLocator: computing ndarray
- curveLocationsMAXIMUM: mask component

"""

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:
	listCurveMaximums: list[tuple[int, int, int]] = [
		(0x15, 0x2a, 0x10), # `bridges = 0`
		(0x55, 0xaa, 0x40),
		(0x155, 0x2aa, 0x100),
		(0x555, 0xaaa, 0x400),
		(0x1555, 0x2aaa, 0x1000),
		(0x5555, 0xaaaa, 0x4000),
		(0x15555, 0x2aaaa, 0x10000),
		(0x55555, 0xaaaaa, 0x40000),
		(0x155555, 0x2aaaaa, 0x100000),
		(0x555555, 0xaaaaaa, 0x400000),
		(0x1555555, 0x2aaaaaa, 0x1000000),
		(0x5555555, 0xaaaaaaa, 0x4000000),
		(0x15555555, 0x2aaaaaaa, 0x10000000),
		(0x55555555, 0xaaaaaaaa, 0x40000000),
		(0x155555555, 0x2aaaaaaaa, 0x100000000),
		(0x555555555, 0xaaaaaaaaa, 0x400000000),
		(0x1555555555, 0x2aaaaaaaaa, 0x1000000000),
		(0x5555555555, 0xaaaaaaaaaa, 0x4000000000),
		(0x15555555555, 0x2aaaaaaaaaa, 0x10000000000),
		(0x55555555555, 0xaaaaaaaaaaa, 0x40000000000),
		(0x155555555555, 0x2aaaaaaaaaaa, 0x100000000000),
		(0x555555555555, 0xaaaaaaaaaaaa, 0x400000000000),
		(0x1555555555555, 0x2aaaaaaaaaaaa, 0x1000000000000),
		(0x5555555555555, 0xaaaaaaaaaaaaa, 0x4000000000000),
		(0x15555555555555, 0x2aaaaaaaaaaaaa, 0x10000000000000),
		(0x55555555555555, 0xaaaaaaaaaaaaaa, 0x40000000000000),
		(0x155555555555555, 0x2aaaaaaaaaaaaaa, 0x100000000000000),
		(0x555555555555555, 0xaaaaaaaaaaaaaaa, 0x400000000000000),
		(0x1555555555555555, 0x2aaaaaaaaaaaaaaa, 0x1000000000000000),
		(0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x4000000000000000),
	]

	listCurveMaximums = listCurveMaximums[0:bridges]

	dictionaryCurveLocations: dict[int, int] = {}

	# This might stay a loop.
	while bridges > 0:
		bridges -= 1

		bifurcationAlphaLocator, bifurcationZuluLocator, curveLocationsMAXIMUM = listCurveMaximums[bridges]

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
