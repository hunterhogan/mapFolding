# ruff: file-ignore[collapsible-else-if]
# ruff: file-ignore[commented-out-code]
"""Compute a(n) for an OEIS ID by computing other OEIS IDs."""
from __future__ import annotations

from functools import cache
from hunterMakesPy import inclusive
from itertools import chain
from mapFolding.oeis import countingMeanders
from mapFolding.oeis._metadata import dictionaryOEIS
from math import factorial, isqrt
from typing import Literal

@cache
def A000136(n: int, f: Literal['A000682', 'A000560'] = 'A000682') -> int:
	"""A000682 or A000560"""
	if n in {1, 2}:
		countTotal: int = n * _A000682(n)
	elif f == 'A000560':
		countTotal = 2 * n * A000560(n - 1)
	else:
		countTotal = n * _A000682(n)
	return countTotal

def A000560(n: int) -> int:
	"""A000682"""
	return _A000682(n + 1) // 2

@cache
def A000682(n: int, f: Literal['A000560', 'A301620', 'A259689', 'A000136', 'A223094'] = 'A000560') -> int:
	"""A000560 or A301620 or A259689 or A000136 or A223094"""
	if n in {1, 2}:
		countTotal: int = 1
	elif f == 'A301620':
		countTotal = 2 ** (n - 2) + sum(2 ** (n - n下x - 2) * A301620(n下x) for n下x in range(3, n - 1))
	elif f == 'A259689':
		countTotal = 2 ** (n - 2) + sum(2 ** (n - 1 - n下j) * sum(A259689(n下j, n下k) * (n下k - 2) for n下k in range(3, (n下j + 2) // 2 + inclusive)) for n下j in range(4, n))
	elif f == 'A000136':
		countTotal = A000136(n) // n
	elif f == 'A223094':
		nMinus1Factorial: int = factorial(n - 1)
		countTotal = nMinus1Factorial - sum(A223094(n下k) * (nMinus1Factorial // factorial(n下k)) for n下k in range(3, n))
	else:
		countTotal = 2 * A000560(n - 1)
	return countTotal

def A001010(n: int, f: Literal['A000682 and A007822', 'A001011 and A000136'] = 'A000682 and A007822') -> int:
	"""A000682 and A007822 or A001011 and A000136"""
	if n == 1:
		countTotal: int = 1
	elif f == 'A001011 and A000136':
		countTotal = 4 * A001011(n) - A000136(n)
	# elif f == 'A000682 and A007822':
	else:
		if n & 0b1:
			countTotal = 2 * _A007822((n - 1) // 2 + 1)
		else:
			countTotal = 2 * _A000682(n // 2 + 1)
	return countTotal

def A001011(n: int) -> int:
	"""A000136 and A001010"""
	if n == 1:
		countTotal: int = 1
	else:
		countTotal = (A001010(n) + A000136(n)) // 4
	return countTotal

@cache
def A005315(n: int) -> int:
	"""A005316"""
	if n in {0, 1}:
		countTotal: int = 1
	else:
		countTotal = _A005316(2 * n - 1)
	return countTotal

def A007822(n: int) -> int:
	"""A001010"""
	if n == 1:
		countTotal: int = 1
	else:
		countTotal = A001010(2 * n - 1) // 2
	return countTotal

def A060206(n: int) -> int:
	"""A000682"""
	return _A000682(2 * n + 1)

def A077460(n: int) -> int:
	"""A005315, A005316, and A060206"""
	if n in {0, 1}:
		countTotal: int = 1
	elif n & 0b1:
		countTotal = (A005315(n) + _A005316(n) + A060206((n - 1) // 2)) // 4
	else:
		countTotal = (A005315(n) + 2 * _A005316(n)) // 4

	return countTotal

def A078591(n: int) -> int:
	"""A005315"""
	if n in {0, 1}:
		countTotal: int = 1
	else:
		countTotal = A005315(n) // 2
	return countTotal

def A178961(n: int) -> int:
	"""A001010"""
	A001010valuesKnown: dict[int, int] = dictionaryOEIS['A001010']['valuesKnown']
	countTotal: int = 0
	for n下i in range(1, n + inclusive):
		if n下i in A001010valuesKnown:
			countTotal += A001010valuesKnown[n下i]
		else:
			countTotal += A001010(n下i)
	return countTotal

def A223094(n: int, f: Literal['A000136 and A000682', 'A223094 and A000682', 'A000682'] = 'A000136 and A000682') -> int:
	"""A000136 and A000682 or A223094 and A000682 or A000682"""
	if n in {1, 2}:
		countTotal: int = A000136(n) - _A000682(n + 1)
	elif f == 'A223094 and A000682':
		nFactorial: int = factorial(n)
		countTotal = (nFactorial - sum(A223094(n下k) * (nFactorial // factorial(n下k)) for n下k in range(3, n)) - _A000682(n + 1))
	elif f == 'A000682':
		countTotal = n * _A000682(n) - _A000682(n + 1)
	else:
		countTotal = A000136(n) - _A000682(n + 1)
	return countTotal

@cache
def A259689(n: int, n下k: int | None = None) -> int:
	"""A000682"""
	nRow: int = n
	if n下k is None:
		nFlattenedZeroBased: int = n - 2
		rowLength: int = (isqrt(4 * nFlattenedZeroBased + 1) + 1) // 2
		indexInRowsPair: int = nFlattenedZeroBased - rowLength * (rowLength - 1)
		if indexInRowsPair < rowLength:
			nRow = 2 * rowLength
			n下k = indexInRowsPair + 2
		else:
			nRow = 2 * rowLength + 1
			n下k = indexInRowsPair - rowLength + 2

	if nRow >= 4 and n下k == nRow // 2:
		countTotal: int = 2 ** ((nRow - 1) // 2) * (nRow - 4) + 2
	elif nRow > 2 and n下k == (nRow + 2) // 2:
		countTotal = 2 ** ((nRow - 1) // 2)
	else:
		countTotal = (
			_A000682(nRow + 1)
			- sum(n下kOther * dictionaryOEIS['A259689']['valuesKnown'][((nRow - 1) ** 2) // 4 + n下kOther] for n下kOther in chain(range(2, n下k), range(n下k + 1, nRow // 2 + 2)))
		) // n下k
	return countTotal

def A259702(n: int) -> int:
	"""A000682"""
	if n == 2:
		countTotal: int = 0
	else:
		countTotal = _A000682(n) // 2 - _A000682(n - 1)
	return countTotal

def A301620(n: int, f: Literal['A000682', 'A259689', 'A259702'] = 'A000682') -> int:
	"""A000682 or A259689 or A259702"""
	if f == 'A259689':
		countTotal: int = sum(A259689(n + 1, n下k) * (n下k - 2) for n下k in range(3, (n + 3) // 2 + inclusive))
	elif f == 'A259702':
		countTotal = 2 * A259702(n + 2)
	else:
		countTotal = _A000682(n + 2) - 2 * _A000682(n + 1)
	return countTotal

#================== Not formulas ==========================

@cache
def _A000682(n: int) -> int:
	return countingMeanders('A000682', n)

def _A007822(n: int) -> int:
	return countingMeanders('A007822', n)

@cache
def _A005316(n: int) -> int:
	return countingMeanders('A005316', n)
