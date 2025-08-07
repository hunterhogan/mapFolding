from mapFolding._oeisFormulas.A000682 import A000682
from mapFolding.oeis import oeisIDfor_n

def A001010(Z0Z_n: int) -> tuple[int, int]:
	"""Complicated.

	a(2n+1) = 2*A007822(n)
	a(2n) = 2*A000682(n+2)
	"""
	if Z0Z_n & 0b1:
		foldsTotal = 2 * oeisIDfor_n('A007822', Z0Z_n)
		A001010n = 2 * Z0Z_n + 1
	else:
		foldsTotal = 2 * A000682(Z0Z_n + 2)
		A001010n = 2 * Z0Z_n

	return (A001010n, foldsTotal)

