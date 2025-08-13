def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:
	curveMaximum: dict[int, tuple[int, int, int]] = {
28: (0x1555555555555555, 0x2aaaaaaaaaaaaaaa, 0x1000000000000000),
29: (0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x4000000000000000),
30: (0x15555555555555555, 0x2aaaaaaaaaaaaaaaa, 0x10000000000000000),
31: (0x55555555555555555, 0xaaaaaaaaaaaaaaaaa, 0x40000000000000000),
32: (0x155555555555555555, 0x2aaaaaaaaaaaaaaaaa, 0x100000000000000000),
33: (0x555555555555555555, 0xaaaaaaaaaaaaaaaaaa, 0x400000000000000000),
34: (0x1555555555555555555, 0x2aaaaaaaaaaaaaaaaaa, 0x1000000000000000000),
35: (0x5555555555555555555, 0xaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000),
36: (0x15555555555555555555, 0x2aaaaaaaaaaaaaaaaaaa, 0x10000000000000000000),
37: (0x55555555555555555555, 0xaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000),
38: (0x155555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000),
39: (0x555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000),
40: (0x1555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000),
41: (0x5555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000),
42: (0x15555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000),
43: (0x55555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000),
44: (0x155555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000),
45: (0x555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000),
46: (0x1555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000),
47: (0x5555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000),
48: (0x15555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000),
49: (0x55555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000),
50: (0x155555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000000),
51: (0x555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000000),
52: (0x1555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000000),
53: (0x5555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000000),
54: (0x15555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000000),
55: (0x55555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000000),
56: (0x155555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000000000),
57: (0x555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000000000),
58: (0x1555555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000000000),
59: (0x5555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000000000),
60: (0x15555555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000000000),
61: (0x55555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000000000),
}
	"""`bridges = 29`
	0x5000000000000000.bit_length() = 63;
	0xaaaaaaaaaaaaaaaa.bit_length() = 64;
	0x5555555555555555.bit_length() = 63"""

	dictionaryCurveLocations: dict[int, int] = {}
	while bridges > 28:
		bridges -= 1

		bifurcationAlphaLocator, bifurcationZuluLocator, curveLocationsMAXIMUM = curveMaximum[bridges]

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
			if bifurcationZuluHasCurves and bifurcationAlphaHasCurves:
				# One Truth-check to select a code path
				bifurcationsCanBePairedTogether = (bifurcationZuluIsEven << 1) | bifurcationAlphaIsEven # pyright: ignore[reportPossiblyUnboundVariable]

				if bifurcationsCanBePairedTogether != 0:  # Case 0 (False, False)
					XOrHere2makePair = 0b1
					findUnpaired_0b1 = 0

					if bifurcationsCanBePairedTogether == 1:  # Case 1: (False, True)
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (bifurcationAlpha & XOrHere2makePair) == 0 else -1
						bifurcationAlphaShiftRight2 = (bifurcationAlpha ^ XOrHere2makePair) >> 2
					elif bifurcationsCanBePairedTogether == 2:  # Case 2: (True, False)
						while findUnpaired_0b1 >= 0:
							XOrHere2makePair <<= 2
							findUnpaired_0b1 += 1 if (bifurcationZulu & XOrHere2makePair) == 0 else -1
						bifurcationZulu ^= XOrHere2makePair

					# Cases 1, 2, and 3 all compute curveLocationAnalysis
# TODO https://github.com/hunterhogan/mapFolding/issues/19
					curveLocationAnalysis = ((bifurcationZulu >> 2) << 1) | bifurcationAlphaShiftRight2 # pyright: ignore[reportPossiblyUnboundVariable]
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

		startingCurveLocations.clear()
		startingCurveLocations, dictionaryCurveLocations = dictionaryCurveLocations, startingCurveLocations

	from mapFolding._oeisFormulas.matrixMeanders64 import count  # noqa: PLC0415
	return count(bridges, startingCurveLocations)

