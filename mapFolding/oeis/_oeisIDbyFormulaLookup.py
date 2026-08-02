from __future__ import annotations

from hunterMakesPy import errorL33T
from itertools import chain
from mapFolding.oeis._metadata import dictionaryOEIS
from math import factorial, isqrt
from typing import Literal

def A000136(n: int, f: str | Literal['A000682', 'A000560', 'A001011 and A001010', 'A223094 and A000682', 'A223095, A077014, and A000682', 'A227167']='A000682') -> int:
    if n in {1, 2}:
        countTotal: int = n * _A000682(n)
    elif f == 'A000560':
        countTotal = 2 * n * _A000560(n - 1)
    elif f == 'A001011 and A001010':
        countTotal = 4 * _A001011(n) - _A001010(n)
    elif f == 'A223094 and A000682':
        countTotal = _A223094(n) + _A000682(n + 1)
    elif f == 'A223095, A077014, and A000682':
        countTotal = _A223095(n) - _A077014(n) + 2 * _A000682(n + 1)
    elif f == 'A227167':
        countTotal = (2 - (n & 1)) * _A227167(n)
    elif f == 'A000682':
        countTotal = n * _A000682(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A000560(n: int, f: str | Literal['A000682', 'A000136']='A000682') -> int:
    if f == 'A000136':
        countTotal: int = _A000136(n + 1) // (2 * n + 2)
    elif f == 'A000682':
        countTotal = _A000682(n + 1) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A000682(n: int, f: str | Literal['A000560', 'A301620', 'A259689', 'A000136', 'A223094', 'A001010', 'A060206 and A000560', 'A077460, A005316, and A000560', 'A223093 and A077014', 'A223093 and A005316', 'A000136 and A223094', 'A223094 and A000682', 'A000136, A077014, and A223095', 'A259702 and A000682', 'A333971 and A000682', 'A334615, A000682, and A000560', 'A337581']='A000560') -> int:
    if n in {1, 2}:
        countTotal: int = 1
    elif f == 'A301620':
        countTotal = 2 ** (n - 2) + sum((2 ** (n - n下x - 2) * _A301620(n下x) for n下x in range(3, n - 1)))
    elif f == 'A259689':
        countTotal = 2 ** (n - 2) + sum((2 ** (n - 1 - n下j) * sum((_A259689((n下j - 1) ** 2 // 4 + n下k) * (n下k - 2) for n下k in range(3, (n下j + 2) // 2 + 1))) for n下j in range(4, n)))
    elif f == 'A000136':
        countTotal = _A000136(n) // n
    elif f == 'A223094':
        nMinus1Factorial: int = factorial(n - 1)
        countTotal = nMinus1Factorial - sum((_A223094(n下k) * (nMinus1Factorial // factorial(n下k)) for n下k in range(3, n)))
    elif f == 'A001010':
        countTotal = _A001010(2 * n - 2) // 2
    elif f == 'A060206 and A000560':
        if n & 1:
            countTotal = _A060206((n - 1) // 2)
        else:
            countTotal = 2 * _A000560(n - 1)
    elif f == 'A077460, A005316, and A000560':
        if n & 1:
            countTotal = 4 * _A077460(n) - _A005316(2 * n - 1) - _A005316(n)
        else:
            countTotal = 2 * _A000560(n - 1)
    elif f == 'A223093 and A077014':
        countTotal = _A223093(n - 1) + _A077014(n - 1)
    elif f == 'A223093 and A005316':
        countTotal = _A223093(n - 1) + (1 + (n & 1)) * _A005316(n - 1)
    elif f == 'A000136 and A223094':
        countTotal = _A000136(n - 1) - _A223094(n - 1)
    elif f == 'A223094 and A000682':
        countTotal = (n - 1) * _A000682(n - 1) - _A223094(n - 1)
    elif f == 'A000136, A077014, and A223095':
        countTotal = (_A000136(n - 1) + _A077014(n - 1) - _A223095(n - 1)) // 2
    elif f == 'A259702 and A000682':
        countTotal = 2 * (_A259702(n) + _A000682(n - 1))
    elif f == 'A333971 and A000682':
        countTotal = _A333971(n + 1) // 4 + _A000682(n - 1)
    elif f == 'A334615, A000682, and A000560':
        if 4 <= n:
            countTotal = _A334615(n) + 4 * _A000682(n - 1) - 4 * _A000682(n - 2)
        else:
            countTotal = 2 * _A000560(n - 1)
    elif f == 'A337581':
        countTotal = _A337581(n + 2) // 4
    elif f == 'A000560':
        countTotal = 2 * _A000560(n - 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A001010(n: int, f: str | Literal['A000682 and A007822', 'A001011 and A000136']='A000682 and A007822') -> int:
    if n == 1:
        countTotal: int = 1
    elif f == 'A001011 and A000136':
        countTotal = 4 * _A001011(n) - _A000136(n)
    elif f == 'A000682 and A007822':
        if n & 1:
            countTotal = 2 * _A007822((n - 1) // 2 + 1)
        else:
            countTotal = 2 * _A000682(n // 2 + 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A001011(n: int, f: str | Literal['A000136 and A001010']='A000136 and A001010') -> int:
    if n == 1:
        countTotal: int = 1
    elif f == 'A000136 and A001010':
        countTotal = (_A001010(n) + _A000136(n)) // 4
    else:
        countTotal = -errorL33T
    return countTotal

def A005315(n: int, f: str | Literal['A005316', 'A077460, A005316, and A060206', 'A078591', 'A085973 and A077054', 'A208357']='A005316') -> int:
    if n in {0, 1}:
        countTotal: int = 1
    elif f == 'A077460, A005316, and A060206':
        if n & 1:
            countTotal = 4 * _A077460(n) - _A005316(n) - _A060206((n - 1) // 2)
        else:
            countTotal = 4 * _A077460(n) - 2 * _A005316(n)
    elif f == 'A078591':
        countTotal = 2 * _A078591(n)
    elif f == 'A085973 and A077054':
        countTotal = _A085973(n) - _A077054(n)
    elif f == 'A208357':
        countTotal = isqrt(_A208357(n - 1))
    elif f == 'A005316':
        countTotal = _A005316(2 * n - 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A005316(n: int, f: str | Literal['A005315 and A005316', 'A000682 and A223093', 'A077014', 'A077054 and A005316', 'A077460, A005315, and A060206', 'A078592 and A005316', 'A227167, A217310, and A217318']='A005315 and A005316') -> int:
    if n in {0, 1}:
        countTotal: int = 1
    elif f == 'A000682 and A223093':
        countTotal = (_A000682(n + 1) - _A223093(n)) // (2 - (n & 1))
    elif f == 'A077014':
        countTotal = _A077014(n) // (2 - (n & 1))
    elif f == 'A077054 and A005316':
        if n & 1:
            countTotal = _A005316(n)
        else:
            countTotal = _A077054(n // 2)
    elif f == 'A077460, A005315, and A060206':
        if n & 1:
            countTotal = 4 * _A077460(n) - _A005315(n) - _A060206((n - 1) // 2)
        else:
            countTotal = (4 * _A077460(n) - _A005315(n)) // 2
    elif f == 'A078592 and A005316':
        if n & 1:
            countTotal = _A005316(n)
        else:
            countTotal = 2 * _A078592(n // 2) - _A005316(n // 2)
    elif f == 'A227167, A217310, and A217318':
        countTotal = _A227167(n) - _A217310(n) - _A217318(n)
    elif f == 'A005315 and A005316':
        if n & 1:
            countTotal = _A005315((n + 1) // 2)
        else:
            countTotal = _A005316(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A007822(n: int, f: str | Literal['A001010']='A001010') -> int:
    if n == 1:
        countTotal: int = 1
    elif f == 'A001010':
        countTotal = _A001010(2 * n - 1) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A060206(n: int, f: str | Literal['A000682', 'A077460, A005315, A005316, and A000682']='A000682') -> int:
    if f == 'A077460, A005315, A005316, and A000682':
        if 0 < n:
            countTotal: int = 4 * _A077460(2 * n + 1) - _A005315(2 * n + 1) - _A005316(2 * n + 1)
        else:
            countTotal = _A000682(2 * n + 1)
    elif f == 'A000682':
        countTotal = _A000682(2 * n + 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A077014(n: int, f: str | Literal['A005316', 'A000682 and A223093', 'A223095, A000136, and A000682']='A005316') -> int:
    if n == 0:
        countTotal: int = 2
    elif f == 'A000682 and A223093':
        countTotal = _A000682(n + 1) - _A223093(n)
    elif f == 'A223095, A000136, and A000682':
        countTotal = _A223095(n) - _A000136(n) + 2 * _A000682(n + 1)
    elif f == 'A005316':
        countTotal = (2 - (n & 1)) * _A005316(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A077054(n: int, f: str | Literal['A005316', 'A085973 and A005315']='A005316') -> int:
    if n == 0:
        countTotal: int = 1
    elif f == 'A085973 and A005315':
        countTotal = _A085973(n) - _A005315(n)
    elif f == 'A005316':
        countTotal = _A005316(2 * n)
    else:
        countTotal = -errorL33T
    return countTotal

def A077460(n: int, f: str | Literal['A005315, A005316, and A060206', 'A005316, A005315, and A060206', 'A000682 and A005316']='A005315, A005316, and A060206') -> int:
    if n in {0, 1}:
        countTotal: int = 1
    elif f == 'A005316, A005315, and A060206':
        if n & 1:
            countTotal = (_A005315(n) + _A005316(n) + _A060206((n - 1) // 2)) // 4
        else:
            countTotal = (_A005316(2 * n - 1) + 2 * _A005316(n)) // 4
    elif f == 'A000682 and A005316':
        if n & 1:
            countTotal = (_A000682(n) + _A005316(2 * n - 1) + _A005316(n)) // 4
        else:
            countTotal = (_A005316(2 * n - 1) + 2 * _A005316(n)) // 4
    elif f == 'A005315, A005316, and A060206':
        if n & 1:
            countTotal = (_A005315(n) + _A005316(n) + _A060206((n - 1) // 2)) // 4
        else:
            countTotal = (_A005315(n) + 2 * _A005316(n)) // 4
    else:
        countTotal = -errorL33T
    return countTotal

def A078591(n: int, f: str | Literal['A005315', 'A005316']='A005315') -> int:
    if n in {0, 1}:
        countTotal: int = 1
    elif f == 'A005316':
        countTotal = _A005316(2 * n - 1) // 2
    elif f == 'A005315':
        countTotal = _A005315(n) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A078592(n: int, f: str | Literal['A005316']='A005316') -> int:
    if n == 0:
        countTotal: int = 1
    elif f == 'A005316':
        countTotal = (_A005316(2 * n) + _A005316(n)) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A085973(n: int, f: str | Literal['A077054 and A005315', 'A005316']='A077054 and A005315') -> int:
    if n == 0:
        countTotal: int = 3
    elif f == 'A005316':
        countTotal = _A005316(2 * n) + _A005316(2 * n - 1)
    elif f == 'A077054 and A005315':
        countTotal = _A077054(n) + _A005315(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A208357(n: int, f: str | Literal['A005315', 'A005316']='A005315') -> int:
    if f == 'A005316':
        countTotal: int = _A005316(2 * n + 1) ** 2
    elif f == 'A005315':
        countTotal = _A005315(n + 1) ** 2
    else:
        countTotal = -errorL33T
    return countTotal

def A217310(n: int, f: str | Literal['A223093', 'A227167, A217318, and A005316']='A223093') -> int:
    if f == 'A227167, A217318, and A005316':
        countTotal: int = _A227167(n) - _A217318(n) - _A005316(n)
    elif f == 'A223093':
        countTotal = _A223093(n) * (1 + n % 2)
    else:
        countTotal = -errorL33T
    return countTotal

def A217318(n: int, f: str | Literal['A223095 and A000034', 'A227167, A217310, and A005316']='A223095 and A000034') -> int:
    if f == 'A227167, A217310, and A005316':
        countTotal: int = _A227167(n) - _A217310(n) - _A005316(n)
    elif f == 'A223095 and A000034':
        countTotal = _A223095(n) * (1 + (n & 1)) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A223093(n: int, f: str | Literal['A000682 and A077014', 'A217310', 'A223094 and A223095']='A000682 and A077014') -> int:
    if f == 'A217310':
        countTotal: int = _A217310(n) // (1 + n % 2)
    elif f == 'A223094 and A223095':
        countTotal = _A223094(n) - _A223095(n)
    elif f == 'A000682 and A077014':
        countTotal = _A000682(n + 1) - _A077014(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A223094(n: int, f: str | Literal['A000136 and A000682', 'A223094 and A000682', 'A000682', 'A223095 and A223093']='A000136 and A000682') -> int:
    if n in {1, 2}:
        countTotal: int = _A000136(n) - _A000682(n + 1)
    elif f == 'A223094 and A000682':
        nFactorial: int = factorial(n)
        countTotal = nFactorial - sum((_A223094(n下k) * (nFactorial // factorial(n下k)) for n下k in range(3, n))) - _A000682(n + 1)
    elif f == 'A000682':
        countTotal = n * _A000682(n) - _A000682(n + 1)
    elif f == 'A223095 and A223093':
        countTotal = _A223095(n) + _A223093(n)
    elif f == 'A000136 and A000682':
        countTotal = _A000136(n) - _A000682(n + 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A223095(n: int, f: str | Literal['A223094 and A223093', 'A000136, A077014, and A000682', 'A217318']='A223094 and A223093') -> int:
    if f == 'A000136, A077014, and A000682':
        countTotal: int = _A000136(n) + _A077014(n) - 2 * _A000682(n + 1)
    elif f == 'A217318':
        countTotal = (2 - (n & 1)) * _A217318(n)
    elif f == 'A223094 and A223093':
        countTotal = _A223094(n) - _A223093(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A227167(n: int, f: str | Literal['A000136', 'A217310, A217318, and A005316']='A000136') -> int:
    if f == 'A217310, A217318, and A005316':
        countTotal: int = _A217310(n) + _A217318(n) + _A005316(n)
    elif f == 'A000136':
        if n & 1:
            countTotal = _A000136(n)
        else:
            countTotal = _A000136(n) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A259689(n: int, f: str | Literal['A000682']='A000682') -> int:
    nFlattenedZeroBased: int = n - 2
    rowLength: int = (isqrt(4 * nFlattenedZeroBased + 1) + 1) // 2
    indexInRowsPair: int = nFlattenedZeroBased - rowLength * (rowLength - 1)
    if indexInRowsPair < rowLength:
        nRow: int = 2 * rowLength
        n下k: int = indexInRowsPair + 2
    else:
        nRow = 2 * rowLength + 1
        n下k = indexInRowsPair - rowLength + 2
    if f == 'A000682':
        if nRow >= 4 and n下k == nRow // 2:
            countTotal: int = 2 ** ((nRow - 1) // 2) * (nRow - 4) + 2
        elif nRow > 2 and n下k == (nRow + 2) // 2:
            countTotal = 2 ** ((nRow - 1) // 2)
        else:
            countTotal = (_A000682(nRow + 1) - sum((n下kOther * dictionaryOEIS['A259689']['valuesKnown'][(nRow - 1) ** 2 // 4 + n下kOther] for n下kOther in chain(range(2, n下k), range(n下k + 1, nRow // 2 + 2))))) // n下k
    else:
        countTotal = -errorL33T
    return countTotal

def A259702(n: int, f: str | Literal['A000682', 'A301620']='A000682') -> int:
    if n <= 2:
        countTotal: int = 0
    elif f == 'A301620':
        countTotal = _A301620(n - 2) // 2
    elif f == 'A000682':
        countTotal = _A000682(n) // 2 - _A000682(n - 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A301620(n: int, f: str | Literal['A000682', 'A259689', 'A259702', 'A334615, A301620, and A000682']='A000682') -> int:
    if f == 'A334615, A301620, and A000682':
        if 2 <= n:
            countTotal: int = _A334615(n + 2) + 2 * _A301620(n - 1)
        else:
            countTotal = _A000682(n + 2) - 2 * _A000682(n + 1)
    elif f == 'A259689':
        countTotal = sum((_A259689(n ** 2 // 4 + n下k) * (n下k - 2) for n下k in range(3, (n + 3) // 2 + 1)))
    elif f == 'A259702':
        countTotal = 2 * _A259702(n + 2)
    elif f == 'A000682':
        countTotal = _A000682(n + 2) - 2 * _A000682(n + 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A333971(n: int, f: str | Literal['A000682']='A000682') -> int:
    if n in {2, 3}:
        countTotal: int = n - 1
    elif f == 'A000682':
        countTotal = 4 * (_A000682(n - 1) - _A000682(n - 2))
    else:
        countTotal = -errorL33T
    return countTotal

def A334615(n: int, f: str | Literal['A000682', 'A301620']='A000682') -> int:
    if n in {2, 3}:
        countTotal: int = 0
    elif f == 'A301620':
        countTotal = _A301620(n - 2) - 2 * _A301620(n - 3)
    elif f == 'A000682':
        countTotal = _A000682(n) - 4 * _A000682(n - 1) + 4 * _A000682(n - 2)
    else:
        countTotal = -errorL33T
    return countTotal

def A337581(n: int, f: str | Literal['A000682']='A000682') -> int:
    if n in {2, 3}:
        countTotal: int = n - 1
    elif f == 'A000682':
        countTotal = 4 * _A000682(n - 2)
    else:
        countTotal = -errorL33T
    return countTotal

def _A000136(n: int) -> int:
    return dictionaryOEIS['A000136']['valuesKnown'][n]

def _A000560(n: int) -> int:
    return dictionaryOEIS['A000560']['valuesKnown'][n]

def _A000682(n: int) -> int:
    return dictionaryOEIS['A000682']['valuesKnown'][n]

def _A001010(n: int) -> int:
    return dictionaryOEIS['A001010']['valuesKnown'][n]

def _A001011(n: int) -> int:
    return dictionaryOEIS['A001011']['valuesKnown'][n]

def _A005315(n: int) -> int:
    return dictionaryOEIS['A005315']['valuesKnown'][n]

def _A005316(n: int) -> int:
    return dictionaryOEIS['A005316']['valuesKnown'][n]

def _A007822(n: int) -> int:
    return dictionaryOEIS['A007822']['valuesKnown'][n]

def _A060206(n: int) -> int:
    return dictionaryOEIS['A060206']['valuesKnown'][n]

def _A077014(n: int) -> int:
    return dictionaryOEIS['A077014']['valuesKnown'][n]

def _A077054(n: int) -> int:
    return dictionaryOEIS['A077054']['valuesKnown'][n]

def _A077460(n: int) -> int:
    return dictionaryOEIS['A077460']['valuesKnown'][n]

def _A078591(n: int) -> int:
    return dictionaryOEIS['A078591']['valuesKnown'][n]

def _A078592(n: int) -> int:
    return dictionaryOEIS['A078592']['valuesKnown'][n]

def _A085973(n: int) -> int:
    return dictionaryOEIS['A085973']['valuesKnown'][n]

def _A208357(n: int) -> int:
    return dictionaryOEIS['A208357']['valuesKnown'][n]

def _A217310(n: int) -> int:
    return dictionaryOEIS['A217310']['valuesKnown'][n]

def _A217318(n: int) -> int:
    return dictionaryOEIS['A217318']['valuesKnown'][n]

def _A223093(n: int) -> int:
    return dictionaryOEIS['A223093']['valuesKnown'][n]

def _A223094(n: int) -> int:
    return dictionaryOEIS['A223094']['valuesKnown'][n]

def _A223095(n: int) -> int:
    return dictionaryOEIS['A223095']['valuesKnown'][n]

def _A227167(n: int) -> int:
    return dictionaryOEIS['A227167']['valuesKnown'][n]

def _A259689(n: int) -> int:
    return dictionaryOEIS['A259689']['valuesKnown'][n]

def _A259702(n: int) -> int:
    return dictionaryOEIS['A259702']['valuesKnown'][n]

def _A301620(n: int) -> int:
    return dictionaryOEIS['A301620']['valuesKnown'][n]

def _A333971(n: int) -> int:
    return dictionaryOEIS['A333971']['valuesKnown'][n]

def _A334615(n: int) -> int:
    return dictionaryOEIS['A334615']['valuesKnown'][n]

def _A337581(n: int) -> int:
    return dictionaryOEIS['A337581']['valuesKnown'][n]
