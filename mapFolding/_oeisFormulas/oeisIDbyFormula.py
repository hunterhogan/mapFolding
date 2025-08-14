from mapFolding import countFolds
from mapFolding._oeisFormulas.A000682 import A000682
from mapFolding._oeisFormulas.A005316 import A005316

def A000136(n: int) -> int:
	return n * A000682(n)

def A000560(n: int) -> int:
	return A000682(n + 1) // 2

def A001010(n: int) -> int:
	"""Complicated.

	a(2n-1) = 2*A007822(n)
	OddQ[n], 2*A007822[[(n - 1)/2 + 1]]]

	a(2n) = 2*A000682(n+1)
	EvenQ[n], 2*A000682[[n/2 + 1]]
	"""
	if n & 0b1:
		foldsTotal = 2 * countFolds(oeisID='A007822', oeis_n=(n - 1)//2 + 1, flow='theorem2Numba')
	else:
		foldsTotal = 2 * A000682(n // 2 + 1)

	return foldsTotal

def A001011(n: int) -> int:
	return (A001010(n) + A000136(n)) // 4

def A005315(n: int) -> int:
	return A005316(2 * n - 1)

def A223094(n: int) -> int:
	return A000136(n) - A000682(n + 1)
# TODO A223094 For n >= 3: a(n) = n! - Sum_{k=3..n-1} (a(k)*n!/k!) - A000682(n+1). - _Roger Ford_, Aug 24 2024

def A259702(n: int) -> int:
	return A000682(n) // 2 - A000682(n - 1)

def A301620(n: int) -> int:
	return A000682(n + 2) - 2 * A000682(n + 1)
# TODO A301620 a(n) = Sum_{k=3..floor((n+3)/2)} (A259689(n+1,k)*(k-2)). - _Roger Ford_, Dec 10 2018

