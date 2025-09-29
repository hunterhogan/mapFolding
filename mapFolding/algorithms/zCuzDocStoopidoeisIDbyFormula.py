"""Compute a(n) for an OEIS ID by computing other OEIS IDs.

TODO Implement A178961 for unknown values of A001010
TODO A223094 For n >= 3: a(n) = n! - Sum_{k=3..n-1} (a(k)*n!/k!) - A000682(n+1). - _Roger Ford_, Aug 24 2024
TODO A301620 a(n) = Sum_{k=3..floor((n+3)/2)} (A259689(n+1,k)*(k-2)). - _Roger Ford_, Dec 10 2018
"""
from functools import cache
from mapFolding import dictionaryOEIS
from mapFolding.basecamp import NOTcountingFolds

# ruff: noqa: D400
@cache
def A000136(n: int) -> int:
	"""A000682"""
	return n * _A000682(n)

def A000560(n: int) -> int:
	"""A000682"""
	return _A000682(n + 1) // 2

def A001010(n: int) -> int:
	"""A000682 or A007822"""
	if n == 1:
		countTotal = 1
	elif n & 0b1:
		countTotal = 2 * _A007822((n - 1)//2 + 1)
	else:
		countTotal = 2 * _A000682(n // 2 + 1)
	return countTotal

def A001011(n: int) -> int:
	"""A000136 and A001010"""
	if n == 1:
		countTotal = 1
	else:
		countTotal = (A001010(n) + A000136(n)) // 4
	return countTotal

@cache
def A005315(n: int) -> int:
	"""A005316"""
	if n in {0, 1}:
		countTotal = 1
	else:
		countTotal = _A005316(2 * n - 1)
	return countTotal

def A060206(n: int) -> int:
	"""A000682"""
	return _A000682(2 * n + 1)

def A077460(n: int) -> int:
	"""A005315, A005316, and A060206"""
	if n in {0, 1}:
		countTotal = 1
	elif n & 0b1:
		countTotal = (A005315(n) + _A005316(n) + A060206((n - 1) // 2)) // 4
	else:
		countTotal = (A005315(n) + 2 * _A005316(n)) // 4

	return countTotal

def A078591(n: int) -> int:
	"""A005315"""
	if n in {0, 1}:
		countTotal = 1
	else:
		countTotal = A005315(n) // 2
	return countTotal

def A178961(n: int) -> int:
	"""A001010"""
	A001010valuesKnown: dict[int, int] = dictionaryOEIS['A001010']['valuesKnown']
	countTotal: int = 0
	for n下i in range(1, n+1):
		countTotal += A001010valuesKnown[n下i]
	return countTotal

def A223094(n: int) -> int:
	"""A000136 and A000682"""
	return A000136(n) - _A000682(n + 1)

def A259702(n: int) -> int:
	"""A000682"""
	if n == 2:
		countTotal = 0
	else:
		countTotal = _A000682(n) // 2 - _A000682(n - 1)
	return countTotal

def A301620(n: int) -> int:
	"""A000682"""
	return _A000682(n + 2) - 2 * _A000682(n + 1)

# ================= Not formulas ==========================

@cache
def _A000682(n: int) -> int:
	return NOTcountingFolds('A000682', n)

def _A007822(n: int) -> int:
	return NOTcountingFolds('A007822', n)

@cache
def _A005316(n: int) -> int:
	return NOTcountingFolds('A005316', n)
