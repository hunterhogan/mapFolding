from mapFolding import countFolds
from mapFolding._oeisFormulas.A000682 import A000682

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

