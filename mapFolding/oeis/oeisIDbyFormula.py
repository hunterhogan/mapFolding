"""Compute a(n) for an OEIS ID by computing other OEIS IDs.

This is a generated file; edit the source file.
"""
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
    """
    Compute A000136(n) as a function of A000682 or A000560.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A000136 is: "Number of ways of folding a strip of n labeled stamps."

    The domain of A000136 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 46.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of ways of folding a strip of n labeled stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A000136
    """
    if n in {1, 2}:
        countTotal: int = n * _A000682(n)
    elif f == 'A000560':
        countTotal = 2 * n * A000560(n - 1)
    else:
        countTotal = n * _A000682(n)
    return countTotal

def A000560(n: int) -> int:
    """
    Compute A000560(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A000560 is: "Number of symmetric ways of folding a strip of n labeled stamps."

    The domain of A000560 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 45.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of symmetric ways of folding a strip of n labeled stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A000560
    """
    return _A000682(n + 1) // 2

@cache
def A000682(n: int, f: Literal['A000560', 'A301620', 'A259689', 'A000136', 'A223094'] = 'A000560') -> int:
    """
    Compute A000682(n) as a function of A000560 or A301620 or A259689 or A000136 or A223094.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A000682 is: "Semi-meanders: number of ways a semi-infinite directed curve can cross a straight line n times."

    The domain of A000682 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 46.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Semi-meanders: number of ways a semi-infinite directed curve can cross a straight line n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A000682
    """
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
    """
    Compute A001010(n) as a function of A000682 and A007822 or A001011 and A000136.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A001010 is: "Number of symmetric foldings of a strip of n blank stamps."

    The domain of A001010 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 53.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of symmetric foldings of a strip of n blank stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A001010
    """
    if n == 1:
        countTotal: int = 1
    elif f == 'A001011 and A000136':
        countTotal = 4 * A001011(n) - A000136(n)
    elif n & 1:
        countTotal = 2 * _A007822((n - 1) // 2 + 1)
    else:
        countTotal = 2 * _A000682(n // 2 + 1)
    return countTotal

def A001011(n: int) -> int:
    """
    Compute A001011(n) as a function of A000136 and A001010.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A001011 is: "Number of ways to fold a strip of n blank stamps."

    The domain of A001011 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 46.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of ways to fold a strip of n blank stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A001011
    """
    if n == 1:
        countTotal: int = 1
    else:
        countTotal = (A001010(n) + A000136(n)) // 4
    return countTotal

@cache
def A005315(n: int) -> int:
    """
    Compute A005315(n) as a function of A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A005315 is: "Closed meandric numbers (or meanders): number of ways a loop can cross a road 2n times."

    The domain of A005315 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 29.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Closed meandric numbers (or meanders): number of ways a loop can cross a road 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A005315
    """
    if n in {0, 1}:
        countTotal: int = 1
    else:
        countTotal = _A005316(2 * n - 1)
    return countTotal

def A007822(n: int) -> int:
    """
    Compute A007822(n) as a function of A001010.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A007822 is: "Number of symmetric foldings of 2n+1 stamps."

    The domain of A007822 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 27.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of symmetric foldings of 2n+1 stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A007822
    """
    if n == 1:
        countTotal: int = 1
    else:
        countTotal = A001010(2 * n - 1) // 2
    return countTotal

def A060206(n: int) -> int:
    """
    Compute A060206(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A060206 is: "Number of rotationally symmetric closed meanders of length 4n+2."

    The domain of A060206 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 21.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of rotationally symmetric closed meanders of length 4n+2.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A060206
    """
    return _A000682(2 * n + 1)

def A077460(n: int) -> int:
    """
    Compute A077460(n) as a function of A005315, A005316, and A060206.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077460 is: "Number of nonisomorphic ways a loop can cross a road (running East-West) 2n times."

    The domain of A077460 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 21.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of nonisomorphic ways a loop can cross a road (running East-West) 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A077460
    """
    if n in {0, 1}:
        countTotal: int = 1
    elif n & 1:
        countTotal = (A005315(n) + _A005316(n) + A060206((n - 1) // 2)) // 4
    else:
        countTotal = (A005315(n) + 2 * _A005316(n)) // 4
    return countTotal

def A078591(n: int) -> int:
    """
    Compute A078591(n) as a function of A005315.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A078591 is: "Number of nonisomorphic ways a loop can cross a road (running East-West) 2n times."

    The domain of A078591 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 29.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of nonisomorphic ways a loop can cross a road (running East-West) 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A078591
    """
    if n in {0, 1}:
        countTotal: int = 1
    else:
        countTotal = A005315(n) // 2
    return countTotal

def A178961(n: int) -> int:
    """
    Compute A178961(n) as a function of A001010.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A178961 is: "Partial sums of A001010."

    The domain of A178961 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 53.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Partial sums of A001010.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A178961
    """
    A001010valuesKnown: dict[int, int] = dictionaryOEIS['A001010']['valuesKnown']
    countTotal: int = 0
    for n下i in range(1, n + inclusive):
        if n下i in A001010valuesKnown:
            countTotal += A001010valuesKnown[n下i]
        else:
            countTotal += A001010(n下i)
    return countTotal

def A223094(n: int, f: Literal['A000136 and A000682', 'A223094 and A000682', 'A000682'] = 'A000136 and A000682') -> int:
    """
    Compute A223094(n) as a function of A000136 and A000682 or A223094 and A000682 or A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A223094 is: "Number of foldings of n labeled stamps in which leaf n is inwards."

    The domain of A223094 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 44.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of foldings of n labeled stamps in which leaf n is inwards.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A223094
    """
    if n in {1, 2}:
        countTotal: int = A000136(n) - _A000682(n + 1)
    elif f == 'A223094 and A000682':
        nFactorial: int = factorial(n)
        countTotal = nFactorial - sum(A223094(n下k) * (nFactorial // factorial(n下k)) for n下k in range(3, n)) - _A000682(n + 1)
    elif f == 'A000682':
        countTotal = n * _A000682(n) - _A000682(n + 1)
    else:
        countTotal = A000136(n) - _A000682(n + 1)
    return countTotal

@cache
def A259689(n: int, n下k: int | None = None) -> int:
    """
    Compute A259689(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A259689 is: "Irregular triangle read by rows: T(n,k) is the number of degree-n permutations without overlaps which furnish k new permutations without overlaps upon the addition of an (n+1)st element, 2 <= k <= 1 + floor(n/2)."

    The domain of A259689 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 171.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Irregular triangle read by rows: T(n,k) is the number of degree-n permutations without overlaps which furnish k new permutations without overlaps upon the addition of an (n+1)st element, 2 <= k <= 1 + floor(n/2).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A259689
    """
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
        countTotal = (_A000682(nRow + 1) - sum(n下kOther * dictionaryOEIS['A259689']['valuesKnown'][(nRow - 1) ** 2 // 4 + n下kOther] for n下kOther in chain(range(2, n下k), range(n下k + 1, nRow // 2 + 2)))) // n下k
    return countTotal

def A259702(n: int) -> int:
    """
    Compute A259702(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A259702 is: "Row sums of A259701 except first column."

    The domain of A259702 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 33.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Row sums of A259701 except first column.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A259702
    """
    if n == 2:
        countTotal: int = 0
    else:
        countTotal = _A000682(n) // 2 - _A000682(n - 1)
    return countTotal

def A301620(n: int, f: Literal['A000682', 'A259689', 'A259702'] = 'A000682') -> int:
    """
    Compute A301620(n) as a function of A000682 or A259689 or A259702.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A301620 is: "a(n) is the total number of top arches with exactly one covering arch for semi-meanders with n top arches."

    The domain of A301620 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 44.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        a(n) is the total number of top arches with exactly one covering arch for semi-meanders with n top arches.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A301620
    """
    if f == 'A259689':
        countTotal: int = sum(A259689(n + 1, n下k) * (n下k - 2) for n下k in range(3, (n + 3) // 2 + inclusive))
    elif f == 'A259702':
        countTotal = 2 * A259702(n + 2)
    else:
        countTotal = _A000682(n + 2) - 2 * _A000682(n + 1)
    return countTotal

@cache
def _A000682(n: int) -> int:
    return countingMeanders('A000682', n)

def _A007822(n: int) -> int:
    return countingMeanders('A007822', n)

@cache
def _A005316(n: int) -> int:
    return countingMeanders('A005316', n)
