"""Compute a(n) for an OEIS ID by computing other OEIS IDs.

This is a generated file; edit the source file.
"""
from __future__ import annotations
from functools import cache
from hunterMakesPy import errorL33T
from mapFolding.basecamp import countFoldsSymmetric, countMeanders
from mapFolding.oeis import getMapShape
from math import factorial, isqrt
from typing import Literal

@cache
def A000136(n: int, f: str | Literal['A000682', 'A000560', 'A001011 and A001010', 'A223094 and A000682', 'A223095, A077014, and A000682', 'A227167']='A000682') -> int:
    """
    Compute A000136(n) as a function of A000682 or A000560 or A001011 and A001010 or A223094 and A000682 or A223095, A077014, and A000682 or A227167.

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
    elif f == 'A001011 and A001010':
        countTotal = 4 * A001011(n) - A001010(n)
    elif f == 'A223094 and A000682':
        countTotal = A223094(n) + _A000682(n + 1)
    elif f == 'A223095, A077014, and A000682':
        countTotal = A223095(n) - A077014(n) + 2 * _A000682(n + 1)
    elif f == 'A227167':
        countTotal = (2 - (n & 1)) * A227167(n)
    elif f == 'A000682':
        countTotal = n * _A000682(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A000560(n: int, f: str | Literal['A000682', 'A000136']='A000682') -> int:
    """
    Compute A000560(n) as a function of A000682 or A000136.

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
    if f == 'A000136':
        countTotal: int = A000136(n + 1) // (2 * n + 2)
    elif f == 'A000682':
        countTotal = _A000682(n + 1) // 2
    else:
        countTotal = -errorL33T
    return countTotal

@cache
def A000682(n: int, f: str | Literal['A000560', 'A301620', 'A000136', 'A223094', 'A001010', 'A060206 and A000560', 'A077460, A005316, and A000560', 'A223093 and A077014', 'A223093 and A005316', 'A000136 and A223094', 'A223094 and A000682', 'A000136, A077014, and A223095', 'A259702 and A000682', 'A333971 and A000682', 'A334615, A000682, and A000560', 'A337581']='A000560') -> int:
    """
    Compute A000682(n) as a function of A000560 or A301620 or A000136 or A223094 or A001010 or A060206 and A000560 or A077460, A005316, and A000560 or A223093 and A077014 or A223093 and A005316 or A000136 and A223094 or A223094 and A000682 or A000136, A077014, and A223095 or A259702 and A000682 or A333971 and A000682 or A334615, A000682, and A000560 or A337581.

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
        countTotal = 2 ** (n - 2) + sum((2 ** (n - n下x - 2) * A301620(n下x) for n下x in range(3, n - 1)))
    elif f == 'A000136':
        countTotal = A000136(n) // n
    elif f == 'A223094':
        nMinus1Factorial: int = factorial(n - 1)
        countTotal = nMinus1Factorial - sum((A223094(n下k) * (nMinus1Factorial // factorial(n下k)) for n下k in range(3, n)))
    elif f == 'A001010':
        countTotal = A001010(2 * n - 2) // 2
    elif f == 'A060206 and A000560':
        if n & 1:
            countTotal = A060206((n - 1) // 2)
        else:
            countTotal = 2 * A000560(n - 1)
    elif f == 'A077460, A005316, and A000560':
        if n & 1:
            countTotal = 4 * A077460(n) - _A005316(2 * n - 1) - _A005316(n)
        else:
            countTotal = 2 * A000560(n - 1)
    elif f == 'A223093 and A077014':
        countTotal = A223093(n - 1) + A077014(n - 1)
    elif f == 'A223093 and A005316':
        countTotal = A223093(n - 1) + (1 + (n & 1)) * _A005316(n - 1)
    elif f == 'A000136 and A223094':
        countTotal = A000136(n - 1) - A223094(n - 1)
    elif f == 'A223094 and A000682':
        countTotal = (n - 1) * _A000682(n - 1) - A223094(n - 1)
    elif f == 'A000136, A077014, and A223095':
        countTotal = (A000136(n - 1) + A077014(n - 1) - A223095(n - 1)) // 2
    elif f == 'A259702 and A000682':
        countTotal = 2 * (A259702(n) + _A000682(n - 1))
    elif f == 'A333971 and A000682':
        countTotal = A333971(n + 1) // 4 + _A000682(n - 1)
    elif f == 'A334615, A000682, and A000560':
        if 4 <= n:
            countTotal = A334615(n) + 4 * _A000682(n - 1) - 4 * _A000682(n - 2)
        else:
            countTotal = 2 * A000560(n - 1)
    elif f == 'A337581':
        countTotal = A337581(n + 2) // 4
    elif f == 'A000560':
        countTotal = 2 * A000560(n - 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A001010(n: int, f: str | Literal['A000682 and A007822', 'A001011 and A000136']='A000682 and A007822') -> int:
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
    elif f == 'A000682 and A007822':
        if n & 1:
            countTotal = 2 * _A007822((n - 1) // 2 + 1)
        else:
            countTotal = 2 * _A000682(n // 2 + 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A001011(n: int, f: str | Literal['A000136 and A001010']='A000136 and A001010') -> int:
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
    elif f == 'A000136 and A001010':
        countTotal = (A001010(n) + A000136(n)) // 4
    else:
        countTotal = -errorL33T
    return countTotal

@cache
def A005315(n: int, f: str | Literal['A005316', 'A077460, A005316, and A060206', 'A078591', 'A085973 and A077054', 'A208357']='A005316') -> int:
    """
    Compute A005315(n) as a function of A005316 or A077460, A005316, and A060206 or A078591 or A085973 and A077054 or A208357.

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
    elif f == 'A077460, A005316, and A060206':
        if n & 1:
            countTotal = 4 * A077460(n) - _A005316(n) - A060206((n - 1) // 2)
        else:
            countTotal = 4 * A077460(n) - 2 * _A005316(n)
    elif f == 'A078591':
        countTotal = 2 * A078591(n)
    elif f == 'A085973 and A077054':
        countTotal = A085973(n) - A077054(n)
    elif f == 'A208357':
        countTotal = isqrt(A208357(n - 1))
    elif f == 'A005316':
        countTotal = _A005316(2 * n - 1)
    else:
        countTotal = -errorL33T
    return countTotal

@cache
def A005316(n: int, f: str | Literal['A005315 and A005316', 'A000682 and A223093', 'A077014', 'A077054 and A005316', 'A077460, A005315, and A060206', 'A078592 and A005316', 'A227167, A217310, and A217318']='A005315 and A005316') -> int:
    """
    Compute A005316(n) as a function of A005315 and A005316 or A000682 and A223093 or A077014 or A077054 and A005316 or A077460, A005315, and A060206 or A078592 and A005316 or A227167, A217310, and A217318.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A005316 is: "Meandric numbers: number of ways a river can cross a road n times."

    The domain of A005316 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 56.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Meandric numbers: number of ways a river can cross a road n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A005316
    """
    if n in {0, 1}:
        countTotal: int = 1
    elif f == 'A000682 and A223093':
        countTotal = (_A000682(n + 1) - A223093(n)) // (2 - (n & 1))
    elif f == 'A077014':
        countTotal = A077014(n) // (2 - (n & 1))
    elif f == 'A077054 and A005316':
        if n & 1:
            countTotal = _A005316(n)
        else:
            countTotal = A077054(n // 2)
    elif f == 'A077460, A005315, and A060206':
        if n & 1:
            countTotal = 4 * A077460(n) - A005315(n) - A060206((n - 1) // 2)
        else:
            countTotal = (4 * A077460(n) - A005315(n)) // 2
    elif f == 'A078592 and A005316':
        if n & 1:
            countTotal = _A005316(n)
        else:
            countTotal = 2 * A078592(n // 2) - _A005316(n // 2)
    elif f == 'A227167, A217310, and A217318':
        countTotal = A227167(n) - A217310(n) - A217318(n)
    elif f == 'A005315 and A005316':
        if n & 1:
            countTotal = A005315((n + 1) // 2)
        else:
            countTotal = _A005316(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A007822(n: int, f: str | Literal['A001010']='A001010') -> int:
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
    elif f == 'A001010':
        countTotal = A001010(2 * n - 1) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A060206(n: int, f: str | Literal['A000682', 'A077460, A005315, A005316, and A000682']='A000682') -> int:
    """
    Compute A060206(n) as a function of A000682 or A077460, A005315, A005316, and A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A060206 is: "Number of rotationally symmetric closed meanders of length 4n+2."

    The domain of A060206 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 23.

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
    if f == 'A077460, A005315, A005316, and A000682':
        if 0 < n:
            countTotal: int = 4 * A077460(2 * n + 1) - A005315(2 * n + 1) - _A005316(2 * n + 1)
        else:
            countTotal = _A000682(2 * n + 1)
    elif f == 'A000682':
        countTotal = _A000682(2 * n + 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A077014(n: int, f: str | Literal['A005316', 'A000682 and A223093', 'A223095, A000136, and A000682']='A005316') -> int:
    """
    Compute A077014(n) as a function of A005316 or A000682 and A223093 or A223095, A000136, and A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077014 is: "Number of ways that a directed line (or river) that starts in the south can cross an east-west road n times."

    The domain of A077014 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 56.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of ways that a directed line (or river) that starts in the south can cross an east-west road n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A077014
    """
    if n == 0:
        countTotal: int = 2
    elif f == 'A000682 and A223093':
        countTotal = _A000682(n + 1) - A223093(n)
    elif f == 'A223095, A000136, and A000682':
        countTotal = A223095(n) - A000136(n) + 2 * _A000682(n + 1)
    elif f == 'A005316':
        countTotal = (2 - (n & 1)) * _A005316(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A077054(n: int, f: str | Literal['A005316', 'A085973 and A005315']='A005316') -> int:
    """
    Compute A077054(n) as a function of A005316 or A085973 and A005315.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077054 is: "Number of ways a river can cross a road 2n times."

    The domain of A077054 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 28.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of ways a river can cross a road 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A077054
    """
    if n == 0:
        countTotal: int = 1
    elif f == 'A085973 and A005315':
        countTotal = A085973(n) - A005315(n)
    elif f == 'A005316':
        countTotal = _A005316(2 * n)
    else:
        countTotal = -errorL33T
    return countTotal

def A077460(n: int, f: str | Literal['A005315, A005316, and A060206', 'A005316, A005315, and A060206', 'A000682 and A005316']='A005315, A005316, and A060206') -> int:
    """
    Compute A077460(n) as a function of A005315, A005316, and A060206 or A005316, A005315, and A060206 or A000682 and A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077460 is: "Number of nonisomorphic ways a loop can cross a road (running East-West) 2n times."

    The domain of A077460 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
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
        https://oeis.org/A077460
    """
    if n in {0, 1}:
        countTotal: int = 1
    elif f == 'A005316, A005315, and A060206':
        if n & 1:
            countTotal = (A005315(n) + _A005316(n) + A060206((n - 1) // 2)) // 4
        else:
            countTotal = (_A005316(2 * n - 1) + 2 * _A005316(n)) // 4
    elif f == 'A000682 and A005316':
        if n & 1:
            countTotal = (_A000682(n) + _A005316(2 * n - 1) + _A005316(n)) // 4
        else:
            countTotal = (_A005316(2 * n - 1) + 2 * _A005316(n)) // 4
    elif f == 'A005315, A005316, and A060206':
        if n & 1:
            countTotal = (A005315(n) + _A005316(n) + A060206((n - 1) // 2)) // 4
        else:
            countTotal = (A005315(n) + 2 * _A005316(n)) // 4
    else:
        countTotal = -errorL33T
    return countTotal

def A078591(n: int, f: str | Literal['A005315', 'A005316']='A005315') -> int:
    """
    Compute A078591(n) as a function of A005315 or A005316.

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
    elif f == 'A005316':
        countTotal = _A005316(2 * n - 1) // 2
    elif f == 'A005315':
        countTotal = A005315(n) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A078592(n: int, f: str | Literal['A005316']='A005316') -> int:
    """
    Compute A078592(n) as a function of A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A078592 is: "Call two meanders from A005316 with 2n crossings equivalent if they differ by reflections in the X or Y axes. Sequence gives number of inequivalent meanders."

    The domain of A078592 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 28.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Call two meanders from A005316 with 2n crossings equivalent if they differ by reflections in the X or Y axes. Sequence gives number of inequivalent meanders.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A078592
    """
    if n == 0:
        countTotal: int = 1
    elif f == 'A005316':
        countTotal = (_A005316(2 * n) + _A005316(n)) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A085973(n: int, f: str | Literal['A077054 and A005315', 'A005316']='A077054 and A005315') -> int:
    """
    Compute A085973(n) as a function of A077054 and A005315 or A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A085973 is: "Number of ways a loop can cross two parallel roads 2n times."

    The domain of A085973 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 28.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of ways a loop can cross two parallel roads 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A085973
    """
    if n == 0:
        countTotal: int = 3
    elif f == 'A005316':
        countTotal = _A005316(2 * n) + _A005316(2 * n - 1)
    elif f == 'A077054 and A005315':
        countTotal = A077054(n) + A005315(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A208357(n: int, f: str | Literal['A005315', 'A005316']='A005315') -> int:
    """
    Compute A208357(n) as a function of A005315 or A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A208357 is: "Number of meanders of order 2n+1 (4*n+2 crossings of the infinite line) with central 1-1 cut."

    The domain of A208357 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 28.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of meanders of order 2n+1 (4*n+2 crossings of the infinite line) with central 1-1 cut.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A208357
    """
    if f == 'A005316':
        countTotal: int = _A005316(2 * n + 1) ** 2
    elif f == 'A005315':
        countTotal = A005315(n + 1) ** 2
    else:
        countTotal = -errorL33T
    return countTotal

def A217310(n: int, f: str | Literal['A223093', 'A227167, A217318, and A005316']='A223093') -> int:
    """
    Compute A217310(n) as a function of A223093 or A227167, A217318, and A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A217310 is: "The number of meandering curves of order n, with only one extremity covered by its arcs."

    The domain of A217310 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 45.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        The number of meandering curves of order n, with only one extremity covered by its arcs.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A217310
    """
    if f == 'A227167, A217318, and A005316':
        countTotal: int = A227167(n) - A217318(n) - _A005316(n)
    elif f == 'A223093':
        countTotal = A223093(n) * (1 + n % 2)
    else:
        countTotal = -errorL33T
    return countTotal

def A217318(n: int, f: str | Literal['A223095 and A000034', 'A227167, A217310, and A005316']='A223095 and A000034') -> int:
    """
    Compute A217318(n) as a function of A223095 and A000034 or A227167, A217310, and A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A217318 is: "The number of meandering curves of order n, with both extremity covered by its arcs."

    The domain of A217318 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 45.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        The number of meandering curves of order n, with both extremity covered by its arcs.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A217318
    """
    if f == 'A227167, A217310, and A005316':
        countTotal: int = A227167(n) - A217310(n) - _A005316(n)
    elif f == 'A223095 and A000034':
        countTotal = A223095(n) * (1 + (n & 1)) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A223093(n: int, f: str | Literal['A000682 and A077014', 'A217310', 'A223094 and A223095']='A000682 and A077014') -> int:
    """
    Compute A223093(n) as a function of A000682 and A077014 or A217310 or A223094 and A223095.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A223093 is: "Number of foldings of n labeled stamps in which leaf 1 is inwards and leaf n outwards (or leaf 1 outwards and leaf n inwards)."

    The domain of A223093 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 45.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of foldings of n labeled stamps in which leaf 1 is inwards and leaf n outwards (or leaf 1 outwards and leaf n inwards).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A223093
    """
    if f == 'A217310':
        countTotal: int = A217310(n) // (1 + n % 2)
    elif f == 'A223094 and A223095':
        countTotal = A223094(n) - A223095(n)
    elif f == 'A000682 and A077014':
        countTotal = _A000682(n + 1) - A077014(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A223094(n: int, f: str | Literal['A000136 and A000682', 'A223094 and A000682', 'A000682', 'A223095 and A223093']='A000136 and A000682') -> int:
    """
    Compute A223094(n) as a function of A000136 and A000682 or A223094 and A000682 or A000682 or A223095 and A223093.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A223094 is: "Number of foldings of n labeled stamps in which leaf n is inwards."

    The domain of A223094 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 45.

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
        countTotal = nFactorial - sum((A223094(n下k) * (nFactorial // factorial(n下k)) for n下k in range(3, n))) - _A000682(n + 1)
    elif f == 'A000682':
        countTotal = n * _A000682(n) - _A000682(n + 1)
    elif f == 'A223095 and A223093':
        countTotal = A223095(n) + A223093(n)
    elif f == 'A000136 and A000682':
        countTotal = A000136(n) - _A000682(n + 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A223095(n: int, f: str | Literal['A223094 and A223093', 'A000136, A077014, and A000682', 'A217318']='A223094 and A223093') -> int:
    """
    Compute A223095(n) as a function of A223094 and A223093 or A000136, A077014, and A000682 or A217318.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A223095 is: "Number of foldings of n labeled stamps in which both end leaves are inwards."

    The domain of A223095 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 45.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of foldings of n labeled stamps in which both end leaves are inwards.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A223095
    """
    if f == 'A000136, A077014, and A000682':
        countTotal: int = A000136(n) + A077014(n) - 2 * _A000682(n + 1)
    elif f == 'A217318':
        countTotal = (2 - (n & 1)) * A217318(n)
    elif f == 'A223094 and A223093':
        countTotal = A223094(n) - A223093(n)
    else:
        countTotal = -errorL33T
    return countTotal

def A227167(n: int, f: str | Literal['A000136', 'A217310, A217318, and A005316']='A000136') -> int:
    """
    Compute A227167(n) as a function of A000136 or A217310, A217318, and A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A227167 is: "The number of meandering curves of order n."

    The domain of A227167 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 46.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        The number of meandering curves of order n.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A227167
    """
    if f == 'A217310, A217318, and A005316':
        countTotal: int = A217310(n) + A217318(n) + _A005316(n)
    elif f == 'A000136':
        if n & 1:
            countTotal = A000136(n)
        else:
            countTotal = A000136(n) // 2
    else:
        countTotal = -errorL33T
    return countTotal

def A259702(n: int, f: str | Literal['A000682', 'A301620']='A000682') -> int:
    """
    Compute A259702(n) as a function of A000682 or A301620.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A259702 is: "Row sums of A259701 except first column."

    The domain of A259702 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 46.

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
    if n <= 2:
        countTotal: int = 0
    elif f == 'A301620':
        countTotal = A301620(n - 2) // 2
    elif f == 'A000682':
        countTotal = _A000682(n) // 2 - _A000682(n - 1)
    else:
        countTotal = -errorL33T
    return countTotal

@cache
def A301620(n: int, f: str | Literal['A000682', 'A259702', 'A334615, A301620, and A000682']='A000682') -> int:
    """
    Compute A301620(n) as a function of A000682 or A259702 or A334615, A301620, and A000682.

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
    if f == 'A334615, A301620, and A000682':
        if 2 <= n:
            countTotal: int = A334615(n + 2) + 2 * A301620(n - 1)
        else:
            countTotal = _A000682(n + 2) - 2 * _A000682(n + 1)
    elif f == 'A259702':
        countTotal = 2 * A259702(n + 2)
    elif f == 'A000682':
        countTotal = _A000682(n + 2) - 2 * _A000682(n + 1)
    else:
        countTotal = -errorL33T
    return countTotal

def A333971(n: int, f: str | Literal['A000682']='A000682') -> int:
    """
    Compute A333971(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A333971 is: "a(n) is the number of semi-meanders with n top arches that have at least one arch with length 1 adjacent to the center of the top arch configuration or at either end of the arch configuration."

    The domain of A333971 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 47.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        a(n) is the number of semi-meanders with n top arches that have at least one arch with length 1 adjacent to the center of the top arch configuration or at either end of the arch configuration.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A333971
    """
    if n in {2, 3}:
        countTotal: int = n - 1
    elif f == 'A000682':
        countTotal = 4 * (_A000682(n - 1) - _A000682(n - 2))
    else:
        countTotal = -errorL33T
    return countTotal

def A334615(n: int, f: str | Literal['A000682', 'A301620']='A000682') -> int:
    """
    Compute A334615(n) as a function of A000682 or A301620.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A334615 is: "a(n) is the number of semi-meanders with n top arches that has no arch of length 1 at the ends of the top arch configuration and no arch of length 1 adjacent to the center of the top arch configuration."

    The domain of A334615 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 46.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        a(n) is the number of semi-meanders with n top arches that has no arch of length 1 at the ends of the top arch configuration and no arch of length 1 adjacent to the center of the top arch configuration.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A334615
    """
    if n in {2, 3}:
        countTotal: int = 0
    elif f == 'A301620':
        countTotal = A301620(n - 2) - 2 * A301620(n - 3)
    elif f == 'A000682':
        countTotal = _A000682(n) - 4 * _A000682(n - 1) + 4 * _A000682(n - 2)
    else:
        countTotal = -errorL33T
    return countTotal

def A337581(n: int, f: str | Literal['A000682']='A000682') -> int:
    """
    Compute A337581(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A337581 is: "a(n) is the number of semi-meanders with n top arches that have both an arch of length 1 adjacent to the center of the top arch configuration and an arch of length 1 starting or ending the top arch configuration."

    The domain of A337581 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 48.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        a(n) is the number of semi-meanders with n top arches that have both an arch of length 1 adjacent to the center of the top arch configuration and an arch of length 1 starting or ending the top arch configuration.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A337581
    """
    if n in {2, 3}:
        countTotal: int = n - 1
    elif f == 'A000682':
        countTotal = 4 * _A000682(n - 2)
    else:
        countTotal = -errorL33T
    return countTotal

@cache
def _A000682(n: int) -> int:
    return countMeanders('semi', n)

def _A007822(n: int) -> int:
    return countFoldsSymmetric(getMapShape('A007822', n))

@cache
def _A005316(n: int) -> int:
    return countMeanders('meanders', n)
