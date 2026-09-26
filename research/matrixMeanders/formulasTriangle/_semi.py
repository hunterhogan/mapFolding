from __future__ import annotations

from research.matrixMeanders.formulasTriangle._A005315 import (
	A005315of1, A005315of2, A005315of3, A005315of4, A005315of5, A005315of6, A005315of7, A005315of8, A005315of9, A005315of10)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

def 工diagonal1(n: int) -> int:
	return A005315of1(n)

def 工diagonal2(n: int) -> int:
	return A005315of2(n)

def 工diagonal3(n: int) -> int:
	return A005315of3(n)

def 工diagonal4(n: int) -> int:
	return A005315of4(n)

def 工diagonal5(n: int) -> int:
	return A005315of5(n)

def 工diagonal6(n: int) -> int:
	return A005315of6(n)

def 工diagonal7(n: int) -> int:
	return A005315of7(n)

def 工diagonal8(n: int) -> int:
	return A005315of8(n)

def 工diagonal9(n: int) -> int:
	return A005315of9(n)

def 工diagonal10(n: int) -> int:
	return A005315of10(n)

boxOfDiagonals: list[Callable[[int], int]] = [
	工diagonal1, 工diagonal2, 工diagonal3, 工diagonal4, 工diagonal5
	, 工diagonal6, 工diagonal7, 工diagonal8, 工diagonal9, 工diagonal10
]
