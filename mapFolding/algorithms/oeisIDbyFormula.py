from functools import cache
from mapFolding import countFolds
from mapFolding.algorithms.matrixMeanders import doTheNeedful

@cache
def A000136(n: int) -> int:
	return n * A000682(n)

def A000560(n: int) -> int:
	return A000682(n + 1) // 2

def A000682getCurveLocations(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)

	curveStart: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveStart << 1) | curveStart]

	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveStart = (curveStart << 4) | 0b101
		listCurveLocations.append((curveStart << 1) | curveStart)

	return dict.fromkeys(listCurveLocations, 1)

@cache
def A000682(n: int) -> int:
	return doTheNeedful(n - 1, A000682getCurveLocations(n - 1))

def A001010(n: int) -> int:
	"""Formulas.

	a(2n-1) = 2*A007822(n)
	OddQ[n], 2*A007822[[(n - 1)/2 + 1]]]

	a(2n) = 2*A000682(n+1)
	EvenQ[n], 2*A000682[[n/2 + 1]]
	"""
	if n == 1:
		foldsTotal = 1
	elif n & 0b1:
		foldsTotal = 2 * countFolds(oeisID='A007822', oeis_n=(n - 1)//2 + 1, flow='theorem2Numba')
	else:
		foldsTotal = 2 * A000682(n // 2 + 1)
	return foldsTotal

def A001011(n: int) -> int:
	if n == 0:
		foldsTotal = 1
	else:
		foldsTotal = (A001010(n) + A000136(n)) // 4
	return foldsTotal

def A005316getCurveLocations(n: int) -> dict[int, int]:
	if n & 0b1:
		return {22: 1}
	else:
		return {15: 1}

@cache
def A005316(n: int) -> int:
	return doTheNeedful(n-1, A005316getCurveLocations(n-1))

@cache
def A005315(n: int) -> int:
	if n == 1:
		foldsTotal = 1
	else:
		foldsTotal = A005316(2 * n - 1)
	return foldsTotal

def A060206(n: int) -> int:
	return A000682(2 * n + 1)

def A077460(n: int) -> int:
	"""Formulas.

	a[0] = a[1] = 1;
	a[n_] := If[OddQ[n], (A005316[[n + 1]] + A005316[[2n]] + A000682[[n]])/4
	a(2n+1) = (A005315(2n+1) + A005316(2n+1) + A060206(n)) / 4.

	a(2n) = (A005315(2n) + 2 * A005316(2n)) / 4.
	(A005316[[2n]] + 2 A005316[[n + 1]])/4];

	"""
	if n in {0, 1}:
		foldsTotal = 1
	elif n & 0b1:
		foldsTotal = (A005315(n) + A005316(n) + A060206((n - 1) // 2)) // 4
	else:
		foldsTotal = (A005315(n) + 2 * A005316(n)) // 4

	return foldsTotal

def A078591(n: int) -> int:
	return A005315(n) // 2

def A178961(n: int) -> int:
	from mapFolding.oeis import dictionaryOEISMeanders  # noqa: PLC0415
	A001010valuesKnown: dict[int, int] = dictionaryOEISMeanders['A001010']['valuesKnown']
	foldsTotal: int = 0
	for n下i in range(1, n+1):
		foldsTotal += A001010valuesKnown[n下i]
	return foldsTotal

def A223094(n: int) -> int:
	return A000136(n) - A000682(n + 1)
# TODO A223094 For n >= 3: a(n) = n! - Sum_{k=3..n-1} (a(k)*n!/k!) - A000682(n+1). - _Roger Ford_, Aug 24 2024

def A259702(n: int) -> int:
	return A000682(n) // 2 - A000682(n - 1)

def A301620(n: int) -> int:
	return A000682(n + 2) - 2 * A000682(n + 1)
# TODO A301620 a(n) = Sum_{k=3..floor((n+3)/2)} (A259689(n+1,k)*(k-2)). - _Roger Ford_, Dec 10 2018
