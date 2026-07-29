"""Compute a(n) for an OEIS ID by computing other OEIS IDs.

This is a generated file; edit the source file.
"""
from __future__ import annotations
from functools import cache
from mapFolding.basecamp import countFoldsSymmetric
from mapFolding.oeis import countingMeanders
from math import factorial
from typing import Literal

@cache
def A000136(n: int, f: Literal['A000682', 'A000560']='A000682') -> int:
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
def A000682(n: int, f: Literal['A000560', 'A301620', 'A000136', 'A223094']='A000560') -> int:
    """
    Compute A000682(n) as a function of A000560 or A301620 or A000136 or A223094.

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
    else:
        countTotal = 2 * A000560(n - 1)
    return countTotal

def A001010(n: int, f: Literal['A000682 and A007822', 'A001011 and A000136']='A000682 and A007822') -> int:
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

def A223094(n: int, f: Literal['A000136 and A000682', 'A223094 and A000682', 'A000682']='A000136 and A000682') -> int:
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
        countTotal = nFactorial - sum((A223094(n下k) * (nFactorial // factorial(n下k)) for n下k in range(3, n))) - _A000682(n + 1)
    elif f == 'A000682':
        countTotal = n * _A000682(n) - _A000682(n + 1)
    else:
        countTotal = A000136(n) - _A000682(n + 1)
    return countTotal

def A301620(n: int) -> int:
    """
    Compute A301620(n) as a function of A000682.

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
    return _A000682(n + 2) - 2 * _A000682(n + 1)

@cache
def _A000682(n: int) -> int:
    return countingMeanders('A000682', n)

def _A007822(n: int) -> int:
    return countFoldsSymmetric((1, 2 * n))

@cache
def _A005316(n: int) -> int:
    return countingMeanders('A005316', n)
