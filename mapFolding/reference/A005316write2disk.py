from pathlib import Path
import shutil
import threading

pathRoot = Path.cwd() / 'curves'

curveMaximum: dict[int, int] = {0: 16,
1: 64,
2: 256,
3: 1024,
4: 4096,
5: 16384,
6: 65536,
7: 262144,
8: 1048576,
9: 4194304,
10: 16777216,
11: 67108864,
12: 268435456,
13: 1073741824,
14: 4294967296,
15: 17179869184,
16: 68719476736,
17: 274877906944,
18: 1099511627776,
19: 4398046511104,
20: 17592186044416,
21: 70368744177664,
22: 281474976710656,
23: 1125899906842624,
24: 4503599627370496,
25: 18014398509481984,
26: 72057594037927936,
27: 288230376151711744,
28: 1152921504606846976,
29: 4611686018427387904,
30: 18446744073709551616,
31: 73786976294838206464,
32: 295147905179352825856,
33: 1180591620717411303424,
34: 4722366482869645213696,
35: 18889465931478580854784,
36: 75557863725914323419136,
37: 302231454903657293676544,
38: 1208925819614629174706176,
39: 4835703278458516698824704,
40: 19342813113834066795298816,
41: 77371252455336267181195264,
42: 309485009821345068724781056,
43: 1237940039285380274899124224,
44: 4951760157141521099596496896,
45: 19807040628566084398385987584,
46: 79228162514264337593543950336,
47: 316912650057057350374175801344,
48: 1267650600228229401496703205376,
49: 5070602400912917605986812821504,
50: 20282409603651670423947251286016,
51: 81129638414606681695789005144064,
52: 324518553658426726783156020576256,
53: 1298074214633706907132624082305024,
54: 5192296858534827628530496329220096,
55: 20769187434139310514121985316880384,
56: 83076749736557242056487941267521536,
57: 332306998946228968225951765070086144,
58: 1329227995784915872903807060280344576,
59: 5316911983139663491615228241121378304,
60: 21267647932558653966460912964485513216,
61: 85070591730234615865843651857942052864}

def buildDictionaryCurveLocations(bridges: int) -> dict[int, int]:
	"""Wait for the writing to finish."""
	dictionaryCurveLocations: dict[int, int] = {}
	for curveLocation in Path(pathRoot, str(bridges)).glob('*'):
		dictionaryCurveLocations[int(curveLocation.stem)] = eval(curveLocation.read_text().strip().rstrip('+'))  # noqa: S307
	return dictionaryCurveLocations

def recordAnalysis(bridges: int, curveLocationAnalysis: int, distinctCrossings: int) -> None:
	"""Record and building system should be fluid: it will write to disk when needed."""
	def _record() -> None:
		if curveLocationAnalysis < curveMaximum[bridges]:
			with Path(pathRoot, str(bridges), str(curveLocationAnalysis)).open('a') as appendStream:
				appendStream.write(f"{distinctCrossings}+")

	bookkeeper = threading.Thread(target=_record)
	bookkeeper.daemon = True
	bookkeeper.start()

def analyzeCurves(bridges: int, curveLocations: int, distinctCrossings: int) -> None:
	bifurcationOdd = curveLocations & 0x5555555555555555555555555555555555555555555555555555555555555555
	bifurcationEven = (curveLocations ^ bifurcationOdd) >> 1

	bifurcationOddHasCurves = bifurcationOdd != 1
	bifurcationEvenHasCurves = bifurcationEven != 1
	bifurcationOddFinalZero = not bifurcationOdd & 1
	bifurcationEvenFinalZero = not bifurcationEven & 1

	if bifurcationOddHasCurves:
		curveLocationAnalysis = (bifurcationOdd >> 2) | (bifurcationEven << 3) | (bifurcationOddFinalZero << 1)
		recordAnalysis(bridges, curveLocationAnalysis, distinctCrossings)

	if bifurcationEvenHasCurves:
		curveLocationAnalysis = (bifurcationEven >> 1) | (bifurcationOdd << 2) | bifurcationEvenFinalZero
		recordAnalysis(bridges, curveLocationAnalysis, distinctCrossings)

	curveLocationAnalysis = ((bifurcationOdd | (bifurcationEven << 1)) << 2) | 3
	recordAnalysis(bridges, curveLocationAnalysis, distinctCrossings)

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
		recordAnalysis(bridges, curveLocationAnalysis, distinctCrossings)

def count(bridges: int, dictionaryCurveLocations: dict[int, int]) -> int:
	shutil.rmtree(pathRoot, ignore_errors=True)
	pathRoot.mkdir(exist_ok=True)
	for n in range(bridges):
		Path(pathRoot, str(n)).mkdir(exist_ok=True)

	while bridges > 0:
		bridges -= 1

		for curveLocations, distinctCrossings in dictionaryCurveLocations.items():
			"""This is now ready for concurrency."""
			analyzeCurves(bridges, curveLocations, distinctCrossings)

		dictionaryCurveLocations = buildDictionaryCurveLocations(bridges)

	return sum(dictionaryCurveLocations.values())

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

