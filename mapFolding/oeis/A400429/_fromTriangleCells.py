from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import 形Triangle

def _getCell(triangle: 形Triangle, row: int, column: int) -> int:
	return triangle[row][column - 1]

def A000136(n: int, *, triangle: 形Triangle) -> int:
	return n * sum(triangle[n])

def A000682(n: int, *, triangle: 形Triangle) -> int:
	return sum(triangle[n])

def A005315(n: int, *, triangle: 形Triangle) -> int:
	if countTotal := _getCell(triangle, 2 * n, 1):
		return countTotal
	return _getCell(triangle, 2 * n - 1, 1)

def A005316(n: int, *, triangle: 形Triangle) -> Fraction:
	return _getCell(triangle, n, 1) + Fraction((1 - n % 2) * (_getCell(triangle, n, 2) - _getCell(triangle, n - 1, 2)), 2)

def A076876(n: int, *, triangle: 形Triangle) -> Fraction:
	return Fraction(
		_getCell(triangle, n, 3) - _getCell(triangle, n, 2) + _getCell(triangle, n + 1, 1)
		- 2 * _getCell(triangle, n + 1, 2) + _getCell(triangle, n + 2, 2), 2)

def A006661(n: int, *, triangle: 形Triangle) -> Fraction:
	return Fraction(
		_getCell(triangle, n - 4, 3) - 2 * _getCell(triangle, n - 4, 2) + 2 * _getCell(triangle, n - 3, 1)
		- 2 * _getCell(triangle, n - 3, 2) + _getCell(triangle, n - 2, 2), 2)

def A077054(n: int, *, triangle: 形Triangle) -> Fraction:
	return Fraction(_getCell(triangle, 2 * n + 1, 1) - _getCell(triangle, 2 * n, 2), 2)

def A077460(n: int, *, triangle: 形Triangle) -> Fraction:
	if n % 2:
		return Fraction(_getCell(triangle, 2 * n, 1) + _getCell(triangle, n, 1) + sum(triangle[n]), 4)
	return Fraction(_getCell(triangle, 2 * n, 1) + _getCell(triangle, n + 1, 1) - _getCell(triangle, n, 2), 4)

def calculateDiagonal2(n: int) -> Fraction:
	return Fraction(n**2 + 2 * n + n % 2 - 20, 2)

def calculateDiagonal3(n: int) -> Fraction:
	return Fraction(
		13 * n**4 + 24 * n**3 - 284 * n**2 - 4560 * n + 17920
		+ (n % 2) * (18 * n**2 - 72 * n + 189) - 64 * (n % 3 == 0), 288)
