def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:
	listCurveMaximums: list[tuple[int, int, int]] = [
(0x15, 0x2a, 0x10),
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
(0x55555555, 0xaaaaaaaa, 0x40000000), # `bridges = 13`, 0xaaaaaaaa.bit_length() = 32
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
(0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x4000000000000000), # 0x5000000000000000.bit_length() = 63; 0xaaaaaaaaaaaaaaaa.bit_length() = 64; 0x5555555555555555.bit_length() = 63
(0x15555555555555555, 0x2aaaaaaaaaaaaaaaa, 0x10000000000000000),
(0x55555555555555555, 0xaaaaaaaaaaaaaaaaa, 0x40000000000000000),
(0x155555555555555555, 0x2aaaaaaaaaaaaaaaaa, 0x100000000000000000),
(0x555555555555555555, 0xaaaaaaaaaaaaaaaaaa, 0x400000000000000000),
(0x1555555555555555555, 0x2aaaaaaaaaaaaaaaaaa, 0x1000000000000000000),
(0x5555555555555555555, 0xaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000),
(0x15555555555555555555, 0x2aaaaaaaaaaaaaaaaaaa, 0x10000000000000000000),
(0x55555555555555555555, 0xaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000),
(0x155555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000),
(0x555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000),
(0x1555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000),
(0x5555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000),
(0x15555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000),
(0x55555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000),
(0x155555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000),
(0x555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000),
(0x1555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000),
(0x5555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000),
(0x15555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000),
(0x55555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000),
(0x155555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000000),
(0x555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000000),
(0x1555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000000),
(0x5555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000000),
(0x15555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000000),
(0x55555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000000),
(0x155555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x100000000000000000000000000000),
(0x555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x400000000000000000000000000000),
(0x1555555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x1000000000000000000000000000000),
(0x5555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x4000000000000000000000000000000),
(0x15555555555555555555555555555555, 0x2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x10000000000000000000000000000000),
(0x55555555555555555555555555555555, 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0x40000000000000000000000000000000),
]
	"""`bridges = 29`
	0x5000000000000000.bit_length() = 63;
	0xaaaaaaaaaaaaaaaa.bit_length() = 64;
	0x5555555555555555.bit_length() = 63"""

	listCurveMaximums = listCurveMaximums[0:bridges]

	dictionaryCurveLocations: dict[int, int] = {}
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
				curveLocationAnalysis = (bifurcationZulu >> 1) | (bifurcationAlpha << 2) | (bifurcationZuluIsEven := 1 - (bifurcationZulu & 1))
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

	return sum(startingCurveLocations.values())

