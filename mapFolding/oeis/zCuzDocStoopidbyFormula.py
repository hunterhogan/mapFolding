# ruff: file-ignore[unused-function-argument]
# ruff: file-ignore[collapsible-else-if]
#=SIN= Ruff suppression: the redundant Literal unions expose valid formula selectors while str accepts oeisIDfor_n dispatch values.
# ruff: file-ignore[redundant-literal-union]
# Run makeDocstrings.py
"""Compute a(n) for an OEIS ID by computing other OEIS IDs."""

from __future__ import annotations

from functools import cache
from mapFolding.basecamp import countFoldsSymmetric
from mapFolding.oeis import getMapShape
from mapFolding.oeis._meanders import countMeanders
from math import factorial, isqrt
from typing import Literal

@cache
def A000136(
    n: int
    , f: str
    | Literal['A000682', 'A000560', 'A001011 and A001010', 'A223094 and A000682', 'A223095, A077014, and A000682', 'A227167'] = 'A000682'
) -> int:
    """A000682 or A000560 or A001011 and A001010 or A223094 and A000682 or A223095, A077014, and A000682 or A227167"""
    if n in {1, 2}:
        countTotal: int = n * _A000682(n)
    elif f == 'A000560':
        countTotal = 2 * n * A000560(n - 1)
    elif f == 'A001011 and A001010':
        countTotal = 4 * A001011(n) - A001010(n)
    elif f == 'A223094 and A000682':
        countTotal = A223094(n) + _A000682(n + 1)
    elif f == 'A223095, A077014, and A000682':
        countTotal = A223095(n) - A077014(n) + 2 * _A000682(n + 1)
    elif f == 'A227167':
        countTotal = (2 - (n & 0b1)) * A227167(n)
    else:
        countTotal = n * _A000682(n)
    return countTotal

def A000560(n: int, f: str | Literal['A000682', 'A000136'] = 'A000682') -> int:
    """A000682 or A000136"""
    if f == 'A000136':
        countTotal: int = A000136(n + 1) // (2 * n + 2)
    else:
        countTotal = _A000682(n + 1) // 2
    return countTotal

@cache
def A000682(
    n: int
    , f: str
    | Literal[
        'A000560'
        , 'A301620'
        , 'A000136'
        , 'A223094'
        , 'A001010'
        , 'A060206'
        , 'A223093 and A077014'
        , 'A000136 and A223094'
        , 'A223094 and A000682'
        , 'A000136, A077014, and A223095'
        , 'A259702 and A000682'
        , 'A330269'
        , 'A333971 and A000682'
        , 'A334615 and A000682'
        , 'A337581'
    ] = 'A000560'
) -> int:
    """A000560 or A301620 or A000136 or A223094 or A001010 or A060206 or A223093 and A077014 or A000136 and A223094 or A223094 and A000682 or A000136, A077014, and A223095 or A259702 and A000682 or A330269 or A333971 and A000682 or A334615 and A000682 or A337581"""
    if n in {1, 2}:
        countTotal: int = 1
    elif f == 'A301620':
        countTotal = 2 ** (n - 2) + sum(2 ** (n - n下x - 2) * A301620(n下x) for n下x in range(3, n - 1))
    elif f == 'A000136':
        countTotal = A000136(n) // n
    elif f == 'A223094':
        nMinus1Factorial: int = factorial(n - 1)
        countTotal = nMinus1Factorial - sum(A223094(n下k) * (nMinus1Factorial // factorial(n下k)) for n下k in range(3, n))
    elif f == 'A001010':
        countTotal = A001010(2 * n - 2) // 2
    elif f == 'A060206' and n & 0b1:
        countTotal = A060206((n - 1) // 2)
    elif f == 'A223093 and A077014':
        countTotal = A223093(n - 1) + A077014(n - 1)
    elif f == 'A000136 and A223094':
        countTotal = A000136(n - 1) - A223094(n - 1)
    elif f == 'A223094 and A000682':
        countTotal = (n - 1) * _A000682(n - 1) - A223094(n - 1)
    elif f == 'A000136, A077014, and A223095':
        countTotal = (A000136(n - 1) + A077014(n - 1) - A223095(n - 1)) // 2
    elif f == 'A259702 and A000682':
        countTotal = 2 * (A259702(n) + _A000682(n - 1))
    elif f == 'A330269':
        countTotal = A330269(n + 1) - A330269(n)
    elif f == 'A333971 and A000682':
        countTotal = A333971(n + 1) // 4 + _A000682(n - 1)
    elif f == 'A334615 and A000682' and n >= 4:
        countTotal = A334615(n) + 4 * _A000682(n - 1) - 4 * _A000682(n - 2)
    elif f == 'A337581':
        countTotal = A337581(n + 2) // 4
    else:
        countTotal = 2 * A000560(n - 1)
    return countTotal

def A001010(n: int, f: str | Literal['A000682 and A007822', 'A001011 and A000136'] = 'A000682 and A007822') -> int:
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

def A001011(n: int, f: str | Literal['A000136 and A001010'] = 'A000136 and A001010') -> int:
    """A000136 and A001010"""
    if n == 1:
        countTotal: int = 1
    else:
        countTotal = (A001010(n) + A000136(n)) // 4
    return countTotal

@cache
def A005315(
    n: int, f: str | Literal['A005316', 'A077460, A005316, and A060206', 'A078591', 'A085973 and A077054', 'A208357'] = 'A005316'
) -> int:
    """A005316 or A077460, A005316, and A060206 or A078591 or A085973 and A077054 or A208357"""
    if n in {0, 1}:
        countTotal: int = 1
    elif f == 'A077460, A005316, and A060206':
        if n & 0b1:
            countTotal = 4 * A077460(n) - _A005316(n) - A060206((n - 1) // 2)
        else:
            countTotal = 4 * A077460(n) - 2 * _A005316(n)
    elif f == 'A078591':
        countTotal = 2 * A078591(n)
    elif f == 'A085973 and A077054':
        countTotal = A085973(n) - A077054(n)
    elif f == 'A208357':
        countTotal = isqrt(A208357(n - 1))
    else:
        countTotal = _A005316(2 * n - 1)
    return countTotal

@cache
def A005316(
    n: int
    , f: str
    | Literal[
        'A005315', 'A077014', 'A077054', 'A077460, A005315, and A060206', 'A078592 and A005316', 'A227167, A217310, and A217318'
    ] = 'A005315'
) -> int:
    """A005315 or A077014 or A077054 or A077460, A005315, and A060206 or A078592 and A005316 or A227167, A217310, and A217318"""
    if n in {0, 1}:
        countTotal: int = 1
    elif f == 'A077014':
        countTotal = A077014(n) // (2 - (n & 0b1))
    elif f == 'A077054':
        if n & 0b1:
            countTotal = _A005316(n)
        else:
            countTotal = A077054(n // 2)
    elif f == 'A077460, A005315, and A060206':
        if n & 0b1:
            countTotal = 4 * A077460(n) - A005315(n) - A060206((n - 1) // 2)
        else:
            countTotal = (4 * A077460(n) - A005315(n)) // 2
    elif f == 'A078592 and A005316':
        if n & 0b1:
            countTotal = _A005316(n)
        else:
            countTotal = 2 * A078592(n // 2) - _A005316(n // 2)
    elif f == 'A227167, A217310, and A217318':
        countTotal = A227167(n) - A217310(n) - A217318(n)
    elif n & 0b1:
        countTotal = A005315((n + 1) // 2)
    else:
        countTotal = _A005316(n)
    return countTotal

def A007822(n: int, f: str | Literal['A001010'] = 'A001010') -> int:
    """A001010"""
    if n == 1:
        countTotal: int = 1
    else:
        countTotal = A001010(2 * n - 1) // 2
    return countTotal

def A060206(n: int, f: str | Literal['A000682', 'A077460, A005315, and A005316'] = 'A000682') -> int:
    """A000682 or A077460, A005315, and A005316"""
    if f == 'A077460, A005315, and A005316' and n > 0:
        countTotal: int = 4 * A077460(2 * n + 1) - A005315(2 * n + 1) - _A005316(2 * n + 1)
    else:
        countTotal = _A000682(2 * n + 1)
    return countTotal

def A077014(n: int, f: str | Literal['A005316', 'A000682 and A223093', 'A223095, A000136, and A000682'] = 'A005316') -> int:
    """A005316 or A000682 and A223093 or A223095, A000136, and A000682"""
    if n == 0:
        countTotal: int = 2
    elif f == 'A000682 and A223093':
        countTotal = _A000682(n + 1) - A223093(n)
    elif f == 'A223095, A000136, and A000682':
        countTotal = A223095(n) - A000136(n) + 2 * _A000682(n + 1)
    elif n & 0b1:
        countTotal = _A005316(n)
    else:
        countTotal = 2 * _A005316(n)
    return countTotal

def A077054(n: int, f: str | Literal['A005316', 'A085973 and A005315'] = 'A005316') -> int:
    """A005316 or A085973 and A005315"""
    if n == 0:
        countTotal: int = 1
    elif f == 'A085973 and A005315':
        countTotal = A085973(n) - A005315(n)
    else:
        countTotal = _A005316(2 * n)
    return countTotal

def A077460(n: int, f: str | Literal['A005315, A005316, and A060206'] = 'A005315, A005316, and A060206') -> int:
    """A005315, A005316, and A060206"""
    if n in {0, 1}:
        countTotal: int = 1
    elif n & 0b1:
        countTotal = (A005315(n) + _A005316(n) + A060206((n - 1) // 2)) // 4
    else:
        countTotal = (A005315(n) + 2 * _A005316(n)) // 4

    return countTotal

def A078591(n: int, f: str | Literal['A005315'] = 'A005315') -> int:
    """A005315"""
    if n in {0, 1}:
        countTotal: int = 1
    else:
        countTotal = A005315(n) // 2
    return countTotal

def A078592(n: int, f: str | Literal['A005316'] = 'A005316') -> int:
    """A005316"""
    if n == 0:
        countTotal: int = 1
    else:
        countTotal = (_A005316(2 * n) + _A005316(n)) // 2
    return countTotal

def A085973(n: int, f: str | Literal['A077054 and A005315'] = 'A077054 and A005315') -> int:
    """A077054 and A005315"""
    if n == 0:
        countTotal: int = 3
    else:
        countTotal = A077054(n) + A005315(n)
    return countTotal

def A208357(n: int, f: str | Literal['A005315'] = 'A005315') -> int:
    """A005315"""
    return A005315(n + 1) ** 2

# TODO typo on 39? on https://oeis.org/A217310
def A217310(n: int, f: str | Literal['A223093', 'A227167, A217318, and A005316'] = 'A223093') -> int:
    """A223093 or A227167, A217318, and A005316"""
    if f == 'A227167, A217318, and A005316':
        countTotal: int = A227167(n) - A217318(n) - _A005316(n)
    else:
        countTotal = A223093(n) * (1 + (n % 2))
    return countTotal

def A217318(n: int, f: str | Literal['A223095 and A000034', 'A227167, A217310, and A005316'] = 'A223095 and A000034') -> int:
    """A223095 and A000034 or A227167, A217310, and A005316"""
    if f == 'A227167, A217310, and A005316':
        countTotal: int = A227167(n) - A217310(n) - _A005316(n)
    else:
        countTotal = A223095(n) * (1 + (n & 0b1)) // 2
    return countTotal

def A223093(n: int, f: str | Literal['A000682 and A077014', 'A217310', 'A223094 and A223095'] = 'A000682 and A077014') -> int:
    """A000682 and A077014 or A217310 or A223094 and A223095"""
    if f == 'A217310':
        countTotal: int = A217310(n) // (1 + (n % 2))
    elif f == 'A223094 and A223095':
        countTotal = A223094(n) - A223095(n)
    else:
        countTotal = _A000682(n + 1) - A077014(n)
    return countTotal

def A223094(
    n: int, f: str | Literal['A000136 and A000682', 'A223094 and A000682', 'A000682', 'A223095 and A223093'] = 'A000136 and A000682'
) -> int:
    """A000136 and A000682 or A223094 and A000682 or A000682 or A223095 and A223093"""
    if n in {1, 2}:
        countTotal: int = A000136(n) - _A000682(n + 1)
    elif f == 'A223094 and A000682':
        nFactorial: int = factorial(n)
        countTotal = nFactorial - sum(A223094(n下k) * (nFactorial // factorial(n下k)) for n下k in range(3, n)) - _A000682(n + 1)
    elif f == 'A000682':
        countTotal = n * _A000682(n) - _A000682(n + 1)
    elif f == 'A223095 and A223093':
        countTotal = A223095(n) + A223093(n)
    else:
        countTotal = A000136(n) - _A000682(n + 1)
    return countTotal

def A223095(n: int, f: str | Literal['A223094 and A223093', 'A000136, A077014, and A000682', 'A217318'] = 'A223094 and A223093') -> int:
    """A223094 and A223093 or A000136, A077014, and A000682 or A217318"""
    if f == 'A000136, A077014, and A000682':
        countTotal: int = A000136(n) + A077014(n) - 2 * _A000682(n + 1)
    elif f == 'A217318':
        countTotal = (2 - (n & 0b1)) * A217318(n)
    else:
        countTotal = A223094(n) - A223093(n)
    return countTotal

def A227167(n: int, f: str | Literal['A000136', 'A217310, A217318, and A005316'] = 'A000136') -> int:
    """A000136 or A217310, A217318, and A005316"""
    if f == 'A217310, A217318, and A005316':
        countTotal: int = A217310(n) + A217318(n) + _A005316(n)
    elif n & 0b1:
        countTotal = A000136(n)
    else:
        countTotal = A000136(n) // 2
    return countTotal

def A259702(n: int, f: str | Literal['A000682'] = 'A000682') -> int:
    """A000682"""
    if n <= 2:
        countTotal: int = 0
    else:
        countTotal = _A000682(n) // 2 - _A000682(n - 1)
    return countTotal

@cache
def A301620(n: int, f: str | Literal['A000682', 'A334615 and A301620'] = 'A000682') -> int:
    """A000682 or A334615 and A301620"""
    if f == 'A334615 and A301620' and n >= 2:
        countTotal: int = A334615(n + 2) + 2 * A301620(n - 1)
    else:
        countTotal = _A000682(n + 2) - 2 * _A000682(n + 1)
    return countTotal

def A330269(n: int, f: str | Literal['A000682'] = 'A000682') -> int:
    """A000682"""
    if n == 1:
        countTotal: int = 1
    else:
        countTotal = sum(map(_A000682, range(1, n)))
    return countTotal

def A333971(n: int, f: str | Literal['A000682'] = 'A000682') -> int:
    """A000682"""
    if n in {2, 3}:
        countTotal: int = n - 1
    else:
        countTotal = 4 * (_A000682(n - 1) - _A000682(n - 2))
    return countTotal

# TODO error on https://oeis.org/A334615. submitted. reviewed.
def A334615(n: int, f: str | Literal['A000682', 'A301620'] = 'A000682') -> int:
    """A000682 or A301620"""
    if n in {2, 3}:
        countTotal: int = 0
    elif f == 'A301620':
        countTotal = A301620(n - 2) - 2 * A301620(n - 3)
    else:
        countTotal = _A000682(n) - 4 * _A000682(n - 1) + 4 * _A000682(n - 2)
    return countTotal

def A337581(n: int, f: str | Literal['A000682'] = 'A000682') -> int:
    """A000682"""
    if n in {2, 3}:
        countTotal: int = n - 1
    else:
        countTotal = 4 * _A000682(n - 2)
    return countTotal

#================== Not formulas ==========================

@cache
def _A000682(n: int) -> int:
    return countMeanders('A000682', n)

def _A007822(n: int) -> int:
    return countFoldsSymmetric(getMapShape('A007822', n))

@cache
def _A005316(n: int) -> int:
    return countMeanders('A005316', n)
