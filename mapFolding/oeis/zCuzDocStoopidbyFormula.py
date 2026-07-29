# ruff: file-ignore[collapsible-else-if]
# Run makeDocstrings.py
"""Compute a(n) for an OEIS ID by computing other OEIS IDs."""
from __future__ import annotations

from functools import cache
from mapFolding.basecamp import countFoldsSymmetric
from mapFolding.oeis import countMeanders
from math import factorial
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
def A000682(n: int, f: Literal['A000560', 'A301620', 'A000136', 'A223094'] = 'A000560') -> int:
	"""A000560 or A301620 or A000136 or A223094"""
	if n in {1, 2}:
		countTotal: int = 1
	elif f == 'A301620':
		countTotal = 2 ** (n - 2) + sum(2 ** (n - n下x - 2) * A301620(n下x) for n下x in range(3, n - 1))
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

def A301620(n: int) -> int:
	"""A000682"""
	return _A000682(n + 2) - 2 * _A000682(n + 1)

#================== Not formulas ==========================

@cache
def _A000682(n: int) -> int:
	return countMeanders('A000682', n)

def _A007822(n: int) -> int:
	return countFoldsSymmetric((1, 2 * n))

@cache
def _A005316(n: int) -> int:
	return countMeanders('A005316', n)
