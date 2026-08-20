#=Sin= ast source module.
#ruff: file-ignore[undocumented-public-function]
#=Sin= Don't remove `typing.Literal`.
#pyright: reportUnusedImport=false
#=Do= Run makeDocstrings.py
"""Compute a(n) for an OEIS ID by computing other OEIS IDs."""

from __future__ import annotations

from itertools import chain
from mapFolding.basecamp import countFolds, countFoldsSymmetric, countMeanders
from mapFolding.oeis import getValuesKnown, makeMapShape
from math import factorial, isqrt
from typing import Literal, LiteralString

# TODO Think of a non-arbitrary way to determine which `f` ought to be the default.
# - Prefer formulas with terms that are "not formulas".

# TODO Standardize the formatting of the formulas.
# - Use A000682 instead of A000136, except in A000682 and A000136.
# - Use modulo to modulate by parity instead of bitwise AND.
# - ONLY because these are Python expressions, use `//` for integer division instead of `/`.
# - Put positive terms to the left of negative terms.
# - Put "dominant" terms to the left of "subordinate" terms.

# TODO Learn if there are other algorithms to directly compute other sequences.
# Yes, Sawada and Li. 'A000136', 'A000560', 'A001011', 'A077055'.

# TODO For each sequence,
# 1. All formulas that exclusively use the terms `n` and the "Not formulas".
# 2. All formulas that use only `n` and one other sequence.

def A000136(n: int, f: LiteralString | None = None) -> int:
	if n in {1, 2}:
		countTotal: int = n * _A000682(n)
	else:
		match f:
			case 'A000560':
				countTotal = 2 * n * A000560(n - 1)
			case 'A001011 and A001010':
				countTotal = 4 * A001011(n) - A001010(n)
			case 'A223094 and A000682':
				countTotal = A223094(n) + _A000682(n + 1)
			case 'A223095, A077014, and A000682':
				countTotal = A223095(n) - A077014(n) + 2 * _A000682(n + 1)
			case 'A227167':
				countTotal = (2 - (n % 2)) * A227167(n)
			case 'A000682' | _:
				countTotal = n * _A000682(n)
	return countTotal

def A000560(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A000136':
			countTotal: int = A000136(n + 1) // (2 * n + 2)
		case 'A000682' | _:
			countTotal = _A000682(n + 1) // 2
	return countTotal

def A000682(n: int, f: LiteralString | None = None) -> int:
	if n in {1, 2}:
		countTotal: int = 1
	else:
		match f:
			case 'A000560':
				countTotal = 2 * A000560(n - 1)
			case 'A001010':
				countTotal = A001010((2 * n) - 2) // 2
			case 'A223094':
				nLess1Factorial: int = factorial(n - 1)
				countTotal = nLess1Factorial - sum(A223094(n下k) * (nLess1Factorial // factorial(n下k)) for n下k in range(3, n))
			case 'A259689':
				countTotal = 2 ** (n - 2) + sum(
					2 ** (n - 1 - n下j) * sum(A259689(((n下j - 1) ** 2) // 4 + n下k) * (n下k - 2)
					for n下k in range(3, (n下j + 2) // 2 + 1))
					for n下j in range(4, n))
			case 'A301620':
				countTotal = 2 ** (n - 2) + sum(2 ** (n - n下x - 2) * A301620(n下x) for n下x in range(3, n - 1))
			case 'A337581':
				countTotal = A337581(n + 2) // 4
			case 'A077460, A005316, and A223093':
				if n % 2:
					countTotal = (4 * A077460(n)) - _A005316((2 * n) - 1) - _A005316(n)
				else:
					countTotal = A223093(n - 1) + ((1 + (n % 2)) * _A005316(n - 1))
			case 'A223093 and A005316':
				countTotal = A223093(n - 1) + ((1 + (n % 2)) * _A005316(n - 1))
			case 'A223093 and A077014':
				countTotal = A223093(n - 1) + A077014(n - 1)
			case 'A000136, A077014, and A223095':
				countTotal = (A000136(n - 1) + A077014(n - 1) - A223095(n - 1)) // 2
			case 'A223094 and A000682':
				countTotal = ((n - 1) * _A000682(n - 1)) - A223094(n - 1)
			case 'A259702 and A000682':
				countTotal = 2 * (A259702(n) + _A000682(n - 1))
			case 'A333971 and A000682':
				countTotal = (A333971(n + 1) // 4) + _A000682(n - 1)
			case 'A334615 and A000682':
				if 4 <= n:
					countTotal = A334615(n) + (4 * _A000682(n - 1)) - (4 * _A000682(n - 2))
				else:
					countTotal = _A000682(n)
			case 'A060206 and A000560':
				if n % 2:
					countTotal = A060206((n - 1) // 2)
				else:
					countTotal = 2 * A000560(n - 1)
			case 'A000136' | _:
				countTotal = _A000136(n) // n
	return countTotal

def A001010(n: int, f: LiteralString | None = None) -> int:
	if n == 1:
		countTotal: int = 1
	else:
		match f:
			case 'A001011 and A000682':
				countTotal = (4 * A001011(n)) - (n * _A000682(n))
			case 'A007822 and A000682' | _:
				if n % 2:
					countTotal = 2 * _A007822((n - 1) // 2 + 1)
				else:
					countTotal = 2 * _A000682(n // 2 + 1)
	return countTotal

def A001011(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A001010 and A000682' | _:
			if n == 1:
				countTotal: int = 1
			else:
				countTotal = (A001010(n) + (n * _A000682(n))) // 4
	return countTotal

def A005315(n: int, f: LiteralString | None = None) -> int:
	if 0 <= n < 2:
		countTotal: int = 1
	else:
		match f:
			case 'A077460, A005316, and A060206':
				if n % 2:
					countTotal = (4 * A077460(n)) - _A005316(n) - A060206((n - 1) // 2)
				else:
					countTotal = (4 * A077460(n)) - _A005316(n) - _A005316(n)
			case 'A078591':
				countTotal = 2 * A078591(n)
			case 'A085973 and A077054':
				countTotal = A085973(n) - A077054(n)
			case 'A208357':
				countTotal = isqrt(A208357(n - 1))
			case 'A005316' | _:
				countTotal = _A005316((2 * n) - 1)
	return countTotal

def A005316(n: int, f: LiteralString | None = None) -> int:
	if 0 <= n < 2:
		countTotal: int = 1
	else:
		match f:
			case 'A077014':
				countTotal = A077014(n) // (2 - (n % 2))
			case 'A077054 and A005315':
				if n % 2:
					countTotal = A005315((n + 1) // 2)
				else:
					countTotal = A077054(n // 2)
			case 'A077460, A005315, and A060206':
				if n % 2:
					countTotal = (4 * A077460(n)) - A005315(n) - A060206((n - 1) // 2)
				else:
					countTotal = ((4 * A077460(n)) - A005315(n)) // 2
			case 'A078592 and A005315':
				if n % 2:
					countTotal = A005315((n + 1) // 2)
				else:
					countTotal = (2 * A078592(n // 2)) - _A005316(n // 2)
			case 'A227167, A217310, and A217318':
				countTotal = A227167(n) - A217310(n) - A217318(n)
			case 'A000682 and A223093' | _:
				countTotal = (_A000682(n + 1) - A223093(n)) // (2 - (n % 2))
	return countTotal

def A007822(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A001010' | _:
			if n == 1:
				countTotal: int = 1
			else:
				countTotal = A001010((2 * n) - 1) // 2
	return countTotal

def A060206(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A077460, A005315, and A005316':
			if 0 < n:
				countTotal: int = (4 * A077460((2 * n) + 1)) - A005315((2 * n) + 1) - _A005316((2 * n) + 1)
			else:
				countTotal = A077460((2 * n) + 1)
		case 'A000682' | _:
			countTotal = _A000682((2 * n) + 1)
	return countTotal

def A077014(n: int, f: LiteralString | None = None) -> int:
	if n == 0:
		countTotal: int = 2
	else:
		match f:
			case 'A000682 and A223093':
				if n == 0:
					countTotal = 2
				else:
					countTotal = _A000682(n + 1) - A223093(n)
			case 'A223095 and A000682':
				countTotal = A223095(n) - (n * _A000682(n)) + (2 * _A000682(n + 1))
			case 'A005316' | _:
				countTotal = (2 - (n % 2)) * _A005316(n)
	return countTotal

def A077054(n: int, f: LiteralString | None = None) -> int:
	if n == 0:
		countTotal: int = 1
	else:
		match f:
			case 'A085973 and A005315':
				countTotal = A085973(n) - A005315(n)
			case 'A005316' | _:
				countTotal = _A005316(2 * n)
	return countTotal

# A223096: need a formula or algorithm.
# ruff: ignore[commented-out-code]
# def A077055(n: int, f: LiteralString | None = None) -> int:
# 	match f:
# 		case 'A005316' | _:
# 			if n % 2:
# 				countTotal: int = (_A005316(n) + 2 * A223096(n // 2)) // 4
# 			else:
# 				countTotal = (_A005316(n) + _A005316(n // 2)) // 2
# 	return countTotal

def A077460(n: int, f: LiteralString | None = None) -> int:
	if 0 <= n < 2:
		countTotal: int = 1
	else:
		match f:
			case 'A005315, A005316, and A060206':
				if n % 2:
					countTotal = (A005315(n) + _A005316(n) + A060206((n - 1) // 2)) // 4
				else:
					countTotal = (A005315(n) + (2 * _A005316(n))) // 4
			case 'A000682 and A005316' | _:
				if n % 2:
					countTotal = (_A000682(n) + _A005316((2 * n) - 1) + _A005316(n)) // 4
				else:
					countTotal = (_A005316(n) + _A005316((2 * n) - 1) + _A005316(n)) // 4
	return countTotal

def A078591(n: int, f: LiteralString | None = None) -> int:
	if 0 <= n < 2:
		countTotal: int = 1
	else:
		match f:
			case 'A005316':
				countTotal = _A005316((2 * n) - 1) // 2
			case 'A005315' | _:
				countTotal = A005315(n) // 2
	return countTotal

def A078592(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A005316' | _:
			if n == 0:
				countTotal: int = 1
			else:
				countTotal = (_A005316(2 * n) + _A005316(n)) // 2
	return countTotal

def A085973(n: int, f: LiteralString | None = None) -> int:
	if n == 0:
		countTotal: int = 3
	else:
		match f:
			case 'A077054 and A005315':
				countTotal = A077054(n) + A005315(n)
			# ruff: ignore[commented-out-code]
			# case 'A005315 and A005316':
				# FIXME This formula system is not commutative for `n`, so `n + 1` gets boosted too
				# much in A005315. I think I need a totally different paradigm.
				# countTotal = A005315(n + 1) + _A005316((2 * n) + 1)
			case 'A005316' | _:
				countTotal = _A005316(2 * n) + _A005316((2 * n) - 1)
	return countTotal

def A208357(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A005315':
			countTotal: int = A005315(n + 1) ** 2
		case 'A005316' | _:
			countTotal = _A005316((2 * n) + 1) ** 2
	return countTotal

def A217310(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A223093':
			countTotal: int = (1 + (n % 2)) * A223093(n)
		case 'A227167, A217318, and A005316':
			countTotal = A227167(n) - A217318(n) - _A005316(n)
		case 'A000682 and A005316' | _:
			countTotal = ((1 + (n % 2)) * _A000682(n + 1)) - (2 * _A005316(n))
	return countTotal

def A217318(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A223095':
			countTotal: int = (1 + (n % 2)) * A223095(n) // 2
		case 'A227167, A217310, and A005316':
			countTotal = A227167(n) - A217310(n) - _A005316(n)
		case 'A005316 and A000682' | _:
			countTotal = _A005316(n) + (n * _A000682(n)) - _A000682(n + 1) - (((1 - (n % 2)) * ((n * _A000682(n)) // 2)) + ((n % 2) * _A000682(n + 1)))
	return countTotal

def A223093(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A217310':
			countTotal: int = A217310(n) // (1 + (n % 2))
		case 'A223094 and A223095':
			countTotal = A223094(n) - A223095(n)
		case 'A000682 and A077014':
			countTotal = _A000682(n + 1) - A077014(n)
		case 'A000682 and A005316' | _:
			countTotal = _A000682(n + 1) - ((2 - (n % 2)) * _A005316(n))
	return countTotal

def A223094(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A223094 and A000682':
			if n in {1, 2}:
				countTotal: int = 0
			else:
				nFactorial: int = factorial(n)
				countTotal = nFactorial - _A000682(n + 1) - sum(A223094(n下k) * (nFactorial // factorial(n下k)) for n下k in range(3, n))
		case 'A223093 and A223095':
			countTotal = A223093(n) + A223095(n)
		case 'A000682' | _:
			countTotal = (n * _A000682(n)) - _A000682(n + 1)
	return countTotal

def A223095(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A077014, and A000682':
			countTotal: int = A077014(n) + (n * _A000682(n)) - (2 * _A000682(n + 1))
		case 'A223094 and A223093':
			countTotal = A223094(n) - A223093(n)
		case 'A217318':
			countTotal = (2 - (n % 2)) * A217318(n)
		case 'A000682 and A005316' | _:
			countTotal = ((2 - (n % 2)) * _A005316(n)) + (n * _A000682(n)) - (2 * _A000682(n + 1))
	return countTotal

def A227167(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A217310, A217318, and A005316':
			countTotal: int = A217310(n) + A217318(n) + _A005316(n)
		case 'A000136' | _:
			countTotal = (n * _A000682(n)) // (2 - (n % 2))
	return countTotal

def A259689(n: int, f: LiteralString | None = None) -> int:
	nFlattenedZeroBased: int = n - 2
	rowLength: int = (isqrt(4 * nFlattenedZeroBased + 1) + 1) // 2
	次InRowsPair: int = nFlattenedZeroBased - rowLength * (rowLength - 1)
	if 次InRowsPair < rowLength:
		nRow: int = 2 * rowLength
		n下k: int = 次InRowsPair + 2
	else:
		nRow = 2 * rowLength + 1
		n下k = 次InRowsPair - rowLength + 2

	match f:
		case 'A000682' | _:
			if (4 <= nRow) and (n下k == (nRow // 2)):
				countTotal: int = (2 ** ((nRow - 1) // 2)) * (nRow - 4) + 2
			elif (2 < nRow) and (n下k == ((nRow + 2) // 2)):
				countTotal = 2 ** ((nRow - 1) // 2)
			else:
				countTotal = (_A000682(nRow + 1)
					- sum(n下i * getValuesKnown('A259689')[((nRow - 1) ** 2) // 4 + n下i]
							for n下i in chain(range(2, n下k), range(n下k + 1, nRow // 2 + 2))
					)
				) // n下k
	return countTotal

def A259702(n: int, f: LiteralString | None = None) -> int:
	if n == 2:
		countTotal: int = 0
	else:
		match f:
			case 'A301620':
				countTotal = A301620(n - 2) // 2
			case 'A000682' | _:
				countTotal = (_A000682(n) // 2) - _A000682(n - 1)
	return countTotal

def A301620(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A334615, A301620, and A000682':
			if 2 <= n:
				countTotal: int = A334615(n + 2) + (2 * A301620(n - 1))
			else:
				countTotal = _A000682(n + 2) - (2 * _A000682(n + 1))
		case 'A259689':
			countTotal = sum(A259689(((n**2) // 4) + n下k) * (n下k - 2) for n下k in range(3, ((n + 3) // 2) + 1))
		case 'A259702':
			countTotal = 2 * A259702(n + 2)
		case 'A000682' | _:
			countTotal = _A000682(n + 2) - (2 * _A000682(n + 1))
	return countTotal

def A333971(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A000682' | _:
			if n in {2, 3}:
				countTotal: int = n - 1
			else:
				countTotal = 4 * (_A000682(n - 1) - _A000682(n - 2))
	return countTotal

def A334615(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A000560':
			if 2 <= n < 5:
				countTotal: int = 0
			else:
				countTotal = 2 * (A000560(n - 1) - (4 * A000560(n - 2)) + (4 * A000560(n - 3)))
		case 'A001010':
			countTotal = (A001010((2 * n) - 2) - (4 * A001010((2 * n) - 4)) + (4 * A001010((2 * n) - 6))) // 2
		case 'A259702':
			countTotal = 2 * (A259702(n) - (2 * A259702(n - 1)))
		case 'A337581':
			countTotal = (A337581(n + 2) - (4 * A337581(n + 1)) + (4 * A337581(n))) // 4
		case 'A227167':
			if 2 <= n < 4:
				countTotal = 0
			else:
				countTotal = (((2 - (n % 2)) * A227167(n - 0) // (n - 0))
						- (4 * (1 + (n % 2)) * A227167(n - 1) // (n - 1))
						+ (4 * (2 - (n % 2)) * A227167(n - 2) // (n - 2))
					)
		case 'A223094':
			if 2 <= n < 4:
				countTotal = 0
			else:
				offset: int = 0
				nLess1factorial: int = factorial(n - 1 + offset)
				countTotal = nLess1factorial - sum(A223094(n下k) * (nLess1factorial // factorial(n下k)) for n下k in range(3, n + offset))

				offset -= 1
				nLess1factorial = factorial(n - 1 + offset)
				countTotal = countTotal - (4 * (nLess1factorial  # ruff: ignore[non-augmented-assignment]
								- sum(A223094(n下k) * (nLess1factorial // factorial(n下k)) for n下k in range(3, n + offset))))

				offset -= 1
				nLess1factorial = factorial(n - 1 + offset)
				countTotal = countTotal + (4 * (nLess1factorial  # ruff: ignore[non-augmented-assignment]
								- sum(A223094(n下k) * (nLess1factorial // factorial(n下k)) for n下k in range(3, n + offset))))
		case 'A301620':
			if 2 <= n < 4:
				countTotal = 0
			else:
				countTotal = A301620(n - 2) - (2 * A301620(n - 3))
		case 'A005316 and A223093':
			countTotal = ((A223093(n - 1) + ((1 + (n % 2)) * _A005316(n - 1)))
					- (4 * (A223093(n - 2) + ((2 - (n % 2)) * _A005316(n - 2))))
					+ (4 * (A223093(n - 3) + ((1 + (n % 2)) * _A005316(n - 3))))
				)
		case 'A005316 and A217310':
			countTotal = (((A217310(n - 1) + (2 * _A005316(n - 1))) // (2 - (n % 2)))
					- ((4 * (A217310(n - 2) + (2 * _A005316(n - 2)))) // (1 + (n % 2)))
					+ ((4 * (A217310(n - 3) + (2 * _A005316(n - 3)))) // (2 - (n % 2)))
				)
		case 'A000682' | _:
			if 2 <= n < 4:
				countTotal = 0
			else:
				countTotal = _A000682(n) - (4 * _A000682(n - 1)) + (4 * _A000682(n - 2))
	return countTotal

def A337581(n: int, f: LiteralString | None = None) -> int:
	match f:
		case 'A000682' | _:
			if n in {2, 3}:
				countTotal: int = n - 1
			else:
				countTotal = 4 * _A000682(n - 2)
	return countTotal

#================== Not formulas ==========================

def _A000136(n: int) -> int:
	return countFolds(makeMapShape('A000136', n))

def _A000682(n: int) -> int:
	return countMeanders('semi', n)

def _A007822(n: int) -> int:
	return countFoldsSymmetric(makeMapShape('A007822', n))

def _A005316(n: int) -> int:
	return countMeanders('meanders', n)
