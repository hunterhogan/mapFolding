"""Compute a(n) for an OEIS ID by computing other OEIS IDs.

This is a generated file; edit `mapFolding.oeis._symbolicRelations.boxOfRelations` and run `easyRun/synthesizeModules.py`.
"""
from __future__ import annotations

from itertools import repeat
from mapFolding.oeis import getTriangle, oeisIDfor_n
from math import factorial, isqrt
from operator import getitem, mul
from typing import Literal, LiteralString

def A000108(n: int, f: Literal['A000108', 'A001246', 'A001263', 'A053121', 'A014137', 'A014137 and A401693', 'A000108 and A401693', 'A001405', 'A287548', 'A331499', 'A008828', 'A000108 and A002057'] | LiteralString | None = None) -> int:
    """
    Compute A000108(n) as a function of A000108 or A001246 or A001263 or A053121 or A014137 or A014137 and A401693 or A000108 and A401693 or A001405 or A287548 or A331499 or A008828 or A000108 and A002057.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A000108
    is "Catalan numbers: C(n) = binomial(2n,n)/(n+1) = (2n)!/(n!(n+1)!).".

    The domain of A000108 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Catalan numbers: C(n) = binomial(2n,n)/(n+1) = (2n)!/(n!(n+1)!).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A000108
    """
    countTotal: int
    match f:
        case 'A000108':
            if 1 <= n:
                countTotal = 2 * (2 * n - 1) * A000108(n - 1) // (n + 1)
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A001246':
            countTotal = isqrt(A001246(n))
        case 'A001263':
            if 1 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A001263')[n], len(range(1, n + 1))), map(1 .__rsub__, range(1, n + 1)), strict=True))
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A053121':
            countTotal = getTriangle('A053121')[2 * n][0 - 0]
        case 'A053121':
            countTotal = sum(map(2 .__rpow__, map(getitem, repeat(getTriangle('A053121')[n], len(range(n + 1))), map(0 .__rsub__, range(n + 1)), strict=True)))
        case 'A014137':
            if 1 <= n:
                countTotal = A014137(n) - A014137(n - 1)
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A014137 and A401693':
            if 4 <= n and n % 2 == 0:
                countTotal = 2 * A014137(n) + A401693(n - 2) - 2 * A014137((n - 2) // 2) - A401693(n)
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A014137 and A401693':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = 2 * A014137(n) + A401693(n - 2) - 2 * A014137((n - 1) // 2) - A401693(n)
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A000108 and A401693':
            if 4 <= n and n % 2 == 0:
                countTotal = -((n + 3) * (2 * A401693(n) - 2 * A000108(n // 2) - A401693(n - 2) - A401693(n + 2))) // (25 * n + 15)
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A000108 and A401693':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = -((n + 3) * (2 * A401693(n) - 2 * A000108((n + 1) // 2) - A401693(n - 2) - A401693(n + 2))) // (25 * n + 15)
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A000108':
            if 1 <= n:
                countTotal = sum(map(mul, map(A000108, range(n), strict=True), map(A000108, map(1 .__rsub__, map(n.__sub__, range(n))), strict=True), strict=True))
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A001405':
            countTotal = A001405(2 * n) // (n + 1)
        case 'A287548':
            if 1 <= n:
                countTotal = getTriangle('A287548')[n][1 - 1]
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A331499':
            countTotal = getTriangle('A331499')[n + 1][n + 1 - 1]
        case 'A331499':
            countTotal = getTriangle('A331499')[n + 2][n + 1 - 1]
        case 'A331499':
            if 1 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A331499')[n], len(range(1, n + 1))), map(1 .__rsub__, range(1, n + 1)), strict=True))
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A008828':
            if 1 <= n:
                countTotal = getTriangle('A008828')[n][n - 1]
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case 'A000108 and A002057':
            if 3 <= n:
                countTotal = 2 * A000108(n - 1) + A002057(n - 3)
            elif True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
        case _:
            if True:
                countTotal = factorial(2 * n) // (factorial(n) * factorial(n + 1))
            else:
                countTotal = oeisIDfor_n('A000108', n)
    return countTotal

def A000136(n: int, f: Literal['A000682'] | LiteralString | None = None) -> int:
    """
    Compute A000136(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A000136
    is "Number of ways of folding a strip of n labeled stamps.".

    The domain of A000136 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 50.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of ways of folding a strip of n labeled stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A000136
    """
    countTotal: int
    match f:
        case 'A000682':
            countTotal = n * A000682(n)
        case _:
            countTotal = oeisIDfor_n('A000136', n)
    return countTotal

def A000560(n: int, f: Literal['A000682', 'A259701', 'A259703'] | LiteralString | None = None) -> int:
    # TODO Disambiguate the symmetry terminology in this generated OEIS description at its source.
    """
    Compute A000560(n) as a function of A000682 or A259701 or A259703.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A000560
    is "Number of symmetric ways of folding a strip of n labeled stamps.".

    The domain of A000560 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 49.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of symmetric ways of folding a strip of n labeled stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A000560
    """
    countTotal: int
    match f:
        case 'A000682':
            countTotal = A000682(n + 1) // 2
        case 'A259701':
            countTotal = getTriangle('A259701')[n + 1][1 - 1]
        case 'A259703':
            countTotal = getTriangle('A259703')[n + 1][1 - 1]
        case _:
            countTotal = oeisIDfor_n('A000560', n)
    return countTotal

def A000682(n: int, f: Literal['A060206', 'A000136', 'A000560', 'A001010', 'A001010 and A001011', 'A005316 and A223093', 'A000682 and A223094', 'A227167', 'A223094', 'A000682 and A301620', 'A000682 and A333971', 'A000682 and A334615', 'A337581', 'A301620', 'A339179', 'A287548', 'A331499', 'A000682 and A259702', 'A259689', 'A259698', 'A259703', 'A400429'] | LiteralString | None = None) -> int:
    """
    Compute A000682(n) as a function of A060206 or A000136 or A000560 or A001010 or A001010 and A001011 or A005316 and A223093 or A000682 and A223094 or A227167 or A223094 or A000682 and A301620 or A000682 and A333971 or A000682 and A334615 or A337581 or A301620 or A339179 or A287548 or A331499 or A000682 and A259702 or A259689 or A259698 or A259703 or A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A000682
    is "Semi-meanders: number of ways a semi-infinite directed curve can cross a straight line n times.".

    The domain of A000682 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 50.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Semi-meanders: number of ways a semi-infinite directed curve can cross a straight line n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A000682
    """
    countTotal: int
    match f:
        case 'A060206':
            if (n + 1) % 2 == 0:
                countTotal = A060206((n - 1) // 2)
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A000136':
            countTotal = A000136(n) // n
        case 'A000560':
            if 3 <= n:
                countTotal = 2 * A000560(n - 1)
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A001010':
            if 2 <= n:
                countTotal = A001010(2 * n - 2) // 2
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A001010 and A001011':
            if 2 <= n:
                countTotal = (4 * A001011(n) - A001010(n)) // n
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A005316 and A223093':
            if 2 <= n:
                countTotal = (2 - (n + 1) % 2) * A005316(n - 1) + A223093(n - 1)
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A000682 and A223094':
            if 2 <= n:
                countTotal = (n - 1) * A000682(n - 1) - A223094(n - 1)
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A227167':
            countTotal = (2 - n % 2) * A227167(n) // n
        case 'A223094':
            if 4 <= n:
                countTotal = factorial(n - 1) - sum(map(int.__floordiv__, map(factorial(n - 1).__rmul__, map(A223094, range(3, n), strict=True)), map(factorial, range(3, n), strict=True), strict=True))
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A000682 and A301620':
            if 3 <= n:
                countTotal = 2 * A000682(n - 1) + A301620(n - 2)
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A000682 and A333971':
            if 3 <= n:
                countTotal = (4 * A000682(n - 1) + A333971(n + 1)) // 4
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A000682 and A334615':
            if 4 <= n:
                countTotal = 4 * A000682(n - 1) + A334615(n) - 4 * A000682(n - 2)
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A337581':
            if 2 <= n:
                countTotal = A337581(n + 2) // 4
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A301620':
            if 3 <= n:
                countTotal = 2 ** (n - 2) + sum(map(mul, map(2 .__pow__, map(2 .__rsub__, map(n.__sub__, range(3, n - 1)))), map(A301620, range(3, n - 1), strict=True), strict=True))
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A339179':
            countTotal = sum(map(getitem, repeat(getTriangle('A339179')[n + 1], len(range(2, (n + 1) // 2 + 2))), map(2 .__rsub__, range(2, (n + 1) // 2 + 2)), strict=True))
        case 'A339179':
            if 2 <= n:
                countTotal = getTriangle('A339179')[n + 2][2 - 2]
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A339179':
            if 2 <= n:
                countTotal = getTriangle('A339179')[n + 2][3 - 2]
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A287548':
            countTotal = getTriangle('A287548')[n][n - 1]
        case 'A331499':
            countTotal = getTriangle('A331499')[n][1 - 1]
        case 'A000682 and A259702':
            if 3 <= n:
                countTotal = 2 * A000682(n - 1) + 2 * A259702(n)
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A259689':
            if 3 <= n:
                countTotal = sum(map(mul, range(2, (n - 1) // 2 + 2), map(getitem, repeat(getTriangle('A259689')[n - 1], len(range(2, (n - 1) // 2 + 2))), map(2 .__rsub__, range(2, (n - 1) // 2 + 2)), strict=True), strict=True))
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A259698':
            if 2 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A259698')[n], len(range(1, n))), map(1 .__rsub__, range(1, n)), strict=True))
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A259703':
            if 2 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A259703')[n], len(range(1, n))), map(1 .__rsub__, range(1, n)), strict=True))
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A259689':
            if 2 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A259689')[n], len(range(2, n // 2 + 2))), map(2 .__rsub__, range(2, n // 2 + 2)), strict=True))
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A400429':
            if 2 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A400429')[n], len(range(1, n // 2 + 1))), map(1 .__rsub__, range(1, n // 2 + 1)), strict=True))
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case _:
            if n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
    return countTotal

def A001010(n: int, f: Literal['A000682 and A007822', 'A000682 and A001011'] | LiteralString | None = None) -> int:
    # TODO Disambiguate the symmetry terminology in this generated OEIS description at its source.
    """
    Compute A001010(n) as a function of A000682 and A007822 or A000682 and A001011.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A001010
    is "Number of symmetric foldings of a strip of n labeled stamps.".

    The domain of A001010 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 53.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of symmetric foldings of a strip of n labeled stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A001010
    """
    countTotal: int
    match f:
        case 'A000682 and A007822':
            if 2 <= n:
                if n % 2 == 1:
                    countTotal = 2 * A007822((n + 1) // 2)
                else:
                    countTotal = 2 * A000682((n + 2) // 2)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A001010', n)
        case 'A000682 and A001011':
            if 2 <= n:
                countTotal = 4 * A001011(n) - n * A000682(n)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A001010', n)
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A001010', n)
    return countTotal

def A001011(n: int, f: Literal['A000682 and A001010'] | LiteralString | None = None) -> int:
    """
    Compute A001011(n) as a function of A000682 and A001010.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A001011
    is "Number of ways to fold a strip of n blank stamps.".

    The domain of A001011 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 50.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of ways to fold a strip of n blank stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A001011
    """
    countTotal: int
    match f:
        case 'A000682 and A001010':
            if 2 <= n:
                countTotal = (n * A000682(n) + A001010(n)) // 4
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A001011', n)
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A001011', n)
    return countTotal

def A001246(n: int, f: Literal['A000108', 'A001246', 'A008828'] | LiteralString | None = None) -> int:
    """
    Compute A001246(n) as a function of A000108 or A001246 or A008828.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A001246
    is "Squares of Catalan numbers.".

    The domain of A001246 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 101.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Squares of Catalan numbers.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A001246
    """
    countTotal: int
    match f:
        case 'A000108':
            countTotal = A000108(n) ** 2
        case 'A001246':
            if 1 <= n:
                countTotal = 4 * (2 * n - 1) ** 2 * A001246(n - 1) // (n + 1) ** 2
            else:
                countTotal = oeisIDfor_n('A001246', n)
        case 'A008828':
            if 1 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A008828')[n], len(range(1, n + 1))), map(1 .__rsub__, range(1, n + 1)), strict=True))
            else:
                countTotal = oeisIDfor_n('A001246', n)
        case _:
            countTotal = oeisIDfor_n('A001246', n)
    return countTotal

def A001405(n: int, f: Literal['A053121', 'A000108', 'A001405'] | LiteralString | None = None) -> int:
    """
    Compute A001405(n) as a function of A053121 or A000108 or A001405.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A001405
    is "a(n) = binomial(n, floor(n/2)).".

    The domain of A001405 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        a(n) = binomial(n, floor(n/2)).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A001405
    """
    countTotal: int
    match f:
        case 'A053121':
            countTotal = sum(map(getitem, repeat(getTriangle('A053121')[n], len(range(n + 1))), map(0 .__rsub__, range(n + 1)), strict=True))
        case 'A000108':
            if n % 2 == 0:
                countTotal = (n + 2) * A000108(n // 2) // 2
            elif True:
                countTotal = factorial(n) // (factorial(-(-n // 2)) * factorial(n // 2))
            else:
                countTotal = oeisIDfor_n('A001405', n)
        case 'A001405':
            if 2 <= n and n % 2 == 0:
                countTotal = 2 * A001405(n - 1)
            elif True:
                countTotal = factorial(n) // (factorial(-(-n // 2)) * factorial(n // 2))
            else:
                countTotal = oeisIDfor_n('A001405', n)
        case 'A001405':
            if 2 <= n:
                countTotal = (4 * (n - 1) * A001405(n - 2) + 2 * A001405(n - 1)) // (n + 1)
            elif True:
                countTotal = factorial(n) // (factorial(-(-n // 2)) * factorial(n // 2))
            else:
                countTotal = oeisIDfor_n('A001405', n)
        case _:
            if True:
                countTotal = factorial(n) // (factorial(-(-n // 2)) * factorial(n // 2))
            else:
                countTotal = oeisIDfor_n('A001405', n)
    return countTotal

def A002057(n: int, f: Literal['A000108', 'A006659'] | LiteralString | None = None) -> int:
    """
    Compute A002057(n) as a function of A000108 or A006659.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A002057
    is "Fourth convolution of Catalan numbers: a(n) = 4*binomial(2*n+3,n)/(n+4).".

    The domain of A002057 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Fourth convolution of Catalan numbers: a(n) = 4*binomial(2*n+3,n)/(n+4).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A002057
    """
    countTotal: int
    match f:
        case 'A000108':
            countTotal = A000108(n + 3) - 2 * A000108(n + 2)
        case 'A006659':
            countTotal = A006659(n + 1) // (n + 2)
        case _:
            if True:
                countTotal = 4 * factorial(2 * n + 3) // (factorial(n) * factorial(n + 4))
            else:
                countTotal = oeisIDfor_n('A002057', n)
    return countTotal

def A002415(n: int, f: Literal['A001263', 'A060972'] | LiteralString | None = None) -> int:
    """
    Compute A002415(n) as a function of A001263 or A060972.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A002415
    is "4-dimensional pyramidal numbers: a(n) = n^2*(n^2-1)/12.".

    The domain of A002415 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        4-dimensional pyramidal numbers: a(n) = n^2*(n^2-1)/12.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A002415
    """
    countTotal: int
    match f:
        case 'A001263':
            if 2 <= n:
                countTotal = getTriangle('A001263')[n + 1][3 - 1]
            elif True:
                countTotal = n ** 2 * (n ** 2 - 1) // 12
            else:
                countTotal = oeisIDfor_n('A002415', n)
        case 'A060972':
            if 2 <= n:
                countTotal = getTriangle('A060972')[n][2 - 1]
            elif True:
                countTotal = n ** 2 * (n ** 2 - 1) // 12
            else:
                countTotal = oeisIDfor_n('A002415', n)
        case 'A060972':
            if 2 <= n:
                countTotal = getTriangle('A060972')[n][n - 1 - 1]
            elif True:
                countTotal = n ** 2 * (n ** 2 - 1) // 12
            else:
                countTotal = oeisIDfor_n('A002415', n)
        case _:
            if True:
                countTotal = n ** 2 * (n ** 2 - 1) // 12
            else:
                countTotal = oeisIDfor_n('A002415', n)
    return countTotal

def A002694(n: int, f: Literal['A006659'] | LiteralString | None = None) -> int:
    """
    Compute A002694(n) as a function of A006659.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A002694
    is "Binomial coefficients C(2n, n-2).".

    The domain of A002694 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 201.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Binomial coefficients C(2n, n-2).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A002694
    """
    countTotal: int
    match f:
        case 'A006659':
            countTotal = A006659(n - 1) // 2
        case _:
            if True:
                countTotal = factorial(2 * n) // (factorial(n - 2) * factorial(n + 2))
            else:
                countTotal = oeisIDfor_n('A002694', n)
    return countTotal

def A004148(n: int, f: Literal['A004148', 'A001263'] | LiteralString | None = None) -> int:
    """
    Compute A004148(n) as a function of A004148 or A001263.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A004148
    is "Generalized Catalan numbers: a(n+1) = a(n) + Sum_{k=1..n-1} a(k)*a(n-1-k).".

    The domain of A004148 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 2405.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Generalized Catalan numbers: a(n+1) = a(n) + Sum_{k=1..n-1} a(k)*a(n-1-k).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A004148
    """
    countTotal: int
    match f:
        case 'A004148':
            if 2 <= n:
                countTotal = A004148(n - 1) + sum(map(mul, map(A004148, range(1, n - 1), strict=True), map(A004148, map(2 .__rsub__, map(n.__sub__, range(1, n - 1))), strict=True), strict=True))
            elif 1 <= n:
                countTotal = sum(map(int.__floordiv__, map(2 .__rpow__, map(factorial, range(-((-n - 1) // 2), n + 1), strict=True)), map(mul, map(mul, map(mul, map(mul, range(-((-n - 1) // 2), n + 1), map(factorial, map(n.__sub__, range(-((-n - 1) // 2), n + 1)), strict=True), strict=True), map(factorial, map(n.__rsub__, map(2 .__mul__, range(-((-n - 1) // 2), n + 1))), strict=True), strict=True), map(factorial, map((n + 1).__sub__, range(-((-n - 1) // 2), n + 1)), strict=True), strict=True), map(factorial, map(1 .__rsub__, map(n.__rsub__, map(2 .__mul__, range(-((-n - 1) // 2), n + 1)))), strict=True), strict=True), strict=True))
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A004148', n)
        case 'A001263':
            if 1 <= n:
                countTotal = sum(map(getitem, map(getitem, repeat(getTriangle('A001263'), len(range(-((-n - 1) // 2), n + 1))), range(-((-n - 1) // 2), n + 1), strict=True), map(1 .__rsub__, map((n + 1).__sub__, range(-((-n - 1) // 2), n + 1))), strict=True))
            elif 1 <= n:
                countTotal = sum(map(int.__floordiv__, map(2 .__rpow__, map(factorial, range(-((-n - 1) // 2), n + 1), strict=True)), map(mul, map(mul, map(mul, map(mul, range(-((-n - 1) // 2), n + 1), map(factorial, map(n.__sub__, range(-((-n - 1) // 2), n + 1)), strict=True), strict=True), map(factorial, map(n.__rsub__, map(2 .__mul__, range(-((-n - 1) // 2), n + 1))), strict=True), strict=True), map(factorial, map((n + 1).__sub__, range(-((-n - 1) // 2), n + 1)), strict=True), strict=True), map(factorial, map(1 .__rsub__, map(n.__rsub__, map(2 .__mul__, range(-((-n - 1) // 2), n + 1)))), strict=True), strict=True), strict=True))
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A004148', n)
        case 'A004148':
            if 4 <= n:
                countTotal = ((n - 1) * A004148(n - 2) + (2 * n - 5) * A004148(n - 3) + (2 * n + 1) * A004148(n - 1) - (n - 4) * A004148(n - 4)) // (n + 2)
            elif 1 <= n:
                countTotal = sum(map(int.__floordiv__, map(2 .__rpow__, map(factorial, range(-((-n - 1) // 2), n + 1), strict=True)), map(mul, map(mul, map(mul, map(mul, range(-((-n - 1) // 2), n + 1), map(factorial, map(n.__sub__, range(-((-n - 1) // 2), n + 1)), strict=True), strict=True), map(factorial, map(n.__rsub__, map(2 .__mul__, range(-((-n - 1) // 2), n + 1))), strict=True), strict=True), map(factorial, map((n + 1).__sub__, range(-((-n - 1) // 2), n + 1)), strict=True), strict=True), map(factorial, map(1 .__rsub__, map(n.__rsub__, map(2 .__mul__, range(-((-n - 1) // 2), n + 1)))), strict=True), strict=True), strict=True))
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A004148', n)
        case _:
            if 1 <= n:
                countTotal = sum(map(int.__floordiv__, map(2 .__rpow__, map(factorial, range(-((-n - 1) // 2), n + 1), strict=True)), map(mul, map(mul, map(mul, map(mul, range(-((-n - 1) // 2), n + 1), map(factorial, map(n.__sub__, range(-((-n - 1) // 2), n + 1)), strict=True), strict=True), map(factorial, map(n.__rsub__, map(2 .__mul__, range(-((-n - 1) // 2), n + 1))), strict=True), strict=True), map(factorial, map((n + 1).__sub__, range(-((-n - 1) // 2), n + 1)), strict=True), strict=True), map(factorial, map(1 .__rsub__, map(n.__rsub__, map(2 .__mul__, range(-((-n - 1) // 2), n + 1)))), strict=True), strict=True), strict=True))
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A004148', n)
    return countTotal

def A005315(n: int, f: Literal['A005316', 'A077054 and A085973', 'A005316, A060206, and A077460', 'A078591', 'A208357', 'A400429', 'A005316 and A400429', 'A008828', 'A060972'] | LiteralString | None = None) -> int:
    """
    Compute A005315(n) as a function of A005316 or A077054 and A085973 or A005316, A060206, and A077460 or A078591 or A208357 or A400429 or A005316 and A400429 or A008828 or A060972.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A005315
    is "Closed meandric numbers (or meanders): number of ways a loop can cross a road 2n times.".

    The domain of A005315 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 39.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Closed meandric numbers (or meanders): number of ways a loop can cross a road 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A005315
    """
    countTotal: int
    match f:
        case 'A005316':
            if 1 <= n:
                countTotal = A005316(2 * n - 1)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A077054 and A085973':
            if 1 <= n:
                countTotal = A085973(n) - A077054(n)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A005316, A060206, and A077460':
            if 2 <= n:
                if n % 2 == 1:
                    countTotal = 4 * A077460(n) - A005316(n) - A060206((n - 1) // 2)
                else:
                    countTotal = 4 * A077460(n) - 2 * A005316(n)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A078591':
            if 2 <= n:
                countTotal = 2 * A078591(n)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A208357':
            if 2 <= n:
                countTotal = isqrt(A208357(n - 1))
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A400429':
            if 1 <= n:
                countTotal = getTriangle('A400429')[2 * n][1 - 1]
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A400429':
            if 2 <= n:
                countTotal = getTriangle('A400429')[2 * n - 1][1 - 1]
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A005316 and A400429':
            if 3 <= n:
                countTotal = 2 * A005316(2 * n - 2) + getTriangle('A400429')[2 * n - 2][2 - 1]
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A008828':
            if 1 <= n:
                countTotal = getTriangle('A008828')[n][1 - 1]
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case 'A060972':
            if 1 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A060972')[n], len(range(1, n + 1))), map(1 .__rsub__, range(1, n + 1)), strict=True))
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
    return countTotal

def A005316(n: int, f: Literal['A005315', 'A077054', 'A077014', 'A005315, A060206, and A077460', 'A005316 and A078592', 'A005316 and A077055', 'A077055 and A223096', 'A000682 and A223093', 'A217310, A217318, and A227167', 'A005315 and A400429', 'A259974', 'A006661 and A076876', 'A380367'] | LiteralString | None = None) -> int:
    """
    Compute A005316(n) as a function of A005315 or A077054 or A077014 or A005315, A060206, and A077460 or A005316 and A078592 or A005316 and A077055 or A077055 and A223096 or A000682 and A223093 or A217310, A217318, and A227167 or A005315 and A400429 or A259974 or A006661 and A076876 or A380367.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A005316
    is "Meandric numbers: number of ways a river can cross a road n times.".

    The domain of A005316 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 60.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Meandric numbers: number of ways a river can cross a road n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A005316
    """
    countTotal: int
    match f:
        case 'A005315':
            if 1 <= n and (n + 1) % 2 == 0:
                countTotal = A005315((n + 1) // 2)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A077054':
            if n % 2 == 0:
                countTotal = A077054(n // 2)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A077014':
            countTotal = A077014(n) // (2 - n % 2)
        case 'A005315, A060206, and A077460':
            if 2 <= n:
                if n % 2 == 1:
                    countTotal = 4 * A077460(n) - A005315(n) - A060206((n - 1) // 2)
                else:
                    countTotal = (4 * A077460(n) - A005315(n)) // 2
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A005316 and A078592':
            if n % 2 == 0:
                countTotal = 2 * A078592(n // 2) - A005316(n // 2)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A005316 and A077055':
            if 2 <= n and n % 2 == 0:
                countTotal = 2 * A077055(n) - A005316(n // 2)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A077055 and A223096':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = 4 * A077055(n) - 2 * A223096((n - 1) // 2)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A000682 and A223093':
            if 1 <= n:
                countTotal = (A000682(n + 1) - A223093(n)) // (2 - n % 2)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A217310, A217318, and A227167':
            if 1 <= n:
                countTotal = A227167(n) - A217310(n) - A217318(n)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A005315 and A400429':
            if 4 <= n and n % 2 == 0:
                countTotal = (A005315((n + 2) // 2) - getTriangle('A400429')[n][2 - 1]) // 2
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A259974':
            countTotal = getTriangle('A259974')[n + 1][1 - 1]
        case 'A259974':
            if 1 <= n:
                countTotal = sum(map(getitem, repeat(getTriangle('A259974')[n], len(range(1, -(-n // 2) + 1))), map(1 .__rsub__, range(1, -(-n // 2) + 1)), strict=True))
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A006661 and A076876':
            if 2 <= n and n % 2 == 0:
                countTotal = A006661(n + 4) - A076876(n)
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
        case 'A380367':
            countTotal = getTriangle('A380367')[n][n - 0]
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
    return countTotal

def A006657(n: int, f: Literal['A008828'] | LiteralString | None = None) -> int:
    """
    Compute A006657(n) as a function of A008828.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A006657
    is "Number of closed meanders with 2 components and 2n bridges.".

    The domain of A006657 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 21.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of closed meanders with 2 components and 2n bridges.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A006657
    """
    countTotal: int
    match f:
        case 'A008828':
            countTotal = getTriangle('A008828')[n][2 - 1]
        case _:
            countTotal = oeisIDfor_n('A006657', n)
    return countTotal

def A006658(n: int, f: Literal['A008828'] | LiteralString | None = None) -> int:
    """
    Compute A006658(n) as a function of A008828.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A006658
    is "Closed meanders with 3 components and 2n bridges.".

    The domain of A006658 starts at 3, therefore for values of `n` < 3, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 21.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Closed meanders with 3 components and 2n bridges.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A006658
    """
    countTotal: int
    match f:
        case 'A008828':
            countTotal = getTriangle('A008828')[n][3 - 1]
        case _:
            countTotal = oeisIDfor_n('A006658', n)
    return countTotal

def A006659(n: int, f: Literal['A008828', 'A002057', 'A002694'] | LiteralString | None = None) -> int:
    """
    Compute A006659(n) as a function of A008828 or A002057 or A002694.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A006659
    is "Number of closed meander systems of order n+1 with n components.".

    The domain of A006659 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of closed meander systems of order n+1 with n components.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A006659
    """
    countTotal: int
    match f:
        case 'A008828':
            countTotal = getTriangle('A008828')[n + 1][n - 1]
        case 'A002057':
            countTotal = (n + 1) * A002057(n - 1)
        case 'A002694':
            countTotal = 2 * A002694(n + 1)
        case _:
            if True:
                countTotal = 2 * factorial(2 * n + 2) // (factorial(n - 1) * factorial(n + 3))
            else:
                countTotal = oeisIDfor_n('A006659', n)
    return countTotal

def A006660(n: int, f: Literal['A259974', 'A380369'] | LiteralString | None = None) -> int:
    """
    Compute A006660(n) as a function of A259974 or A380369.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A006660
    is "Number of meanders in which first bridge is 3.".

    The domain of A006660 starts at 3, therefore for values of `n` < 3, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 42.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of meanders in which first bridge is 3.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A006660
    """
    countTotal: int
    match f:
        case 'A259974':
            countTotal = getTriangle('A259974')[n][2 - 1]
        case 'A380369':
            if (n + 1) % 2 == 0:
                countTotal = getTriangle('A380369')[(n - 1) // 2][1 - 0]
            else:
                countTotal = oeisIDfor_n('A006660', n)
        case _:
            countTotal = oeisIDfor_n('A006660', n)
    return countTotal

def A006661(n: int, f: Literal['A400429', 'A259974', 'A005316 and A076876'] | LiteralString | None = None) -> int:
    """
    Compute A006661(n) as a function of A400429 or A259974 or A005316 and A076876.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A006661
    is "Number of meanders in which first bridge is 5.".

    The domain of A006661 starts at 5, therefore for values of `n` < 5, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 41.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of meanders in which first bridge is 5.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A006661
    """
    countTotal: int
    match f:
        case 'A400429':
            if 10 <= n and n % 2 == 0:
                countTotal = (getTriangle('A400429')[n - 4][3 - 1] + 2 * getTriangle('A400429')[n - 3][1 - 1] + getTriangle('A400429')[n - 2][2 - 1] - 2 * getTriangle('A400429')[n - 4][2 - 1] - 2 * getTriangle('A400429')[n - 3][2 - 1]) // 2
            else:
                countTotal = oeisIDfor_n('A006661', n)
        case 'A259974':
            countTotal = getTriangle('A259974')[n][3 - 1]
        case 'A005316 and A076876':
            if 6 <= n and n % 2 == 0:
                countTotal = A005316(n - 4) + A076876(n - 4)
            else:
                countTotal = oeisIDfor_n('A006661', n)
        case _:
            countTotal = oeisIDfor_n('A006661', n)
    return countTotal

def A006662(n: int, f: Literal['A259974'] | LiteralString | None = None) -> int:
    """
    Compute A006662(n) as a function of A259974.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A006662
    is "Number of meanders in which first bridge is 7.".

    The domain of A006662 starts at 7, therefore for values of `n` < 7, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 41.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of meanders in which first bridge is 7.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A006662
    """
    countTotal: int
    match f:
        case 'A259974':
            countTotal = getTriangle('A259974')[n][4 - 1]
        case _:
            countTotal = oeisIDfor_n('A006662', n)
    return countTotal

def A007746(n: int, f: Literal['A008828'] | LiteralString | None = None) -> int:
    """
    Compute A007746(n) as a function of A008828.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A007746
    is "Number of ways for n-3 nonintersecting loops to cross a line 2n times.".

    The domain of A007746 starts at 4, therefore for values of `n` < 4, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 51.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of ways for n-3 nonintersecting loops to cross a line 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A007746
    """
    countTotal: int
    match f:
        case 'A008828':
            countTotal = getTriangle('A008828')[n][n - 3 - 1]
        case _:
            if True:
                countTotal = 4 * (n ** 4 + 20 * n ** 3 + 107 * n ** 2 + 15 - 107 * n) * factorial(2 * n) // (3 * factorial(n - 4) * factorial(n + 6))
            else:
                countTotal = oeisIDfor_n('A007746', n)
    return countTotal

def A007822(n: int, f: Literal['A001010'] | LiteralString | None = None) -> int:
    # TODO Disambiguate the symmetry terminology in this generated OEIS description at its source.
    """
    Compute A007822(n) as a function of A001010.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A007822
    is "Number of symmetric foldings of 2n+1 stamps.".

    The domain of A007822 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 27.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of symmetric foldings of 2n+1 stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A007822
    """
    countTotal: int
    match f:
        case 'A001010':
            if 2 <= n:
                countTotal = A001010(2 * n - 1) // 2
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A007822', n)
        case _:
            if n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A007822', n)
    return countTotal

def A014137(n: int, f: Literal['A000108', 'A000108 and A014137', 'A014140', 'A000108, A014137, and A401693', 'A401693 and A401694', 'A014137'] | LiteralString | None = None) -> int:
    """
    Compute A014137(n) as a function of A000108 or A000108 and A014137 or A014140 or A000108, A014137, and A401693 or A401693 and A401694 or A014137.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A014137
    is "Partial sums of Catalan numbers (A000108).".

    The domain of A014137 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Partial sums of Catalan numbers (A000108).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A014137
    """
    countTotal: int
    match f:
        case 'A000108':
            countTotal = sum(map(A000108, range(n + 1), strict=True))
        case 'A000108 and A014137':
            if 1 <= n:
                countTotal = A000108(n) + A014137(n - 1)
            else:
                countTotal = oeisIDfor_n('A014137', n)
        case 'A014140':
            if 1 <= n:
                countTotal = A014140(n) - A014140(n - 1)
            else:
                countTotal = oeisIDfor_n('A014137', n)
        case 'A000108, A014137, and A401693':
            if 4 <= n and n % 2 == 0:
                countTotal = (A000108(n) + 2 * A014137((n - 2) // 2) + A401693(n) - A401693(n - 2)) // 2
            else:
                countTotal = oeisIDfor_n('A014137', n)
        case 'A000108, A014137, and A401693':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = (A000108(n) + 2 * A014137((n - 1) // 2) + A401693(n) - A401693(n - 2)) // 2
            else:
                countTotal = oeisIDfor_n('A014137', n)
        case 'A401693 and A401694':
            if 1 <= n:
                countTotal = (A401693(2 * n) + A401693(2 * n - 1) + A401693(2 * n + 1) + 1 - A401694(2 * n)) // 2
            else:
                countTotal = oeisIDfor_n('A014137', n)
        case 'A401693 and A401694':
            if 2 <= n:
                countTotal = (A401693(2 * n) + A401693(2 * n - 2) + A401693(2 * n - 1) + 2 - A401694(2 * n - 1)) // 2
            else:
                countTotal = oeisIDfor_n('A014137', n)
        case 'A014137':
            if 2 <= n:
                countTotal = ((5 * n - 1) * A014137(n - 1) - 2 * (2 * n - 1) * A014137(n - 2)) // (n + 1)
            else:
                countTotal = oeisIDfor_n('A014137', n)
        case _:
            countTotal = oeisIDfor_n('A014137', n)
    return countTotal

def A014140(n: int, f: Literal['A000108', 'A014137 and A014140', 'A014140 and A401693', 'A014140 and A401694'] | LiteralString | None = None) -> int:
    """
    Compute A014140(n) as a function of A000108 or A014137 and A014140 or A014140 and A401693 or A014140 and A401694.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A014140
    is "Apply partial sum operator twice to Catalan numbers.".

    The domain of A014140 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 201.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Apply partial sum operator twice to Catalan numbers.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A014140
    """
    countTotal: int
    match f:
        case 'A000108':
            countTotal = sum(map(mul, map((n + 1).__sub__, range(n + 1)), map(A000108, range(n + 1), strict=True), strict=True))
        case 'A014137 and A014140':
            if 1 <= n:
                countTotal = A014137(n) + A014140(n - 1)
            else:
                countTotal = oeisIDfor_n('A014140', n)
        case 'A014140 and A401693':
            if 2 <= n and n % 2 == 0:
                countTotal = 2 * A014140((n - 2) // 2) + A401693(n) + 1
            else:
                countTotal = oeisIDfor_n('A014140', n)
        case 'A014140 and A401693':
            if 1 <= n and (n + 1) % 2 == 0:
                countTotal = 2 * A014140((n - 1) // 2) + A401693(n)
            else:
                countTotal = oeisIDfor_n('A014140', n)
        case 'A014140 and A401694':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = 2 * A014140((n - 3) // 2) + 4 * A014140((n - 1) // 2) + A401694(n - 1) - A014140(n - 2) - A014140(n - 1)
            else:
                countTotal = oeisIDfor_n('A014140', n)
        case 'A014140 and A401694':
            if 4 <= n and n % 2 == 0:
                countTotal = 2 * A014140(n // 2) + 2 * A014140((n - 4) // 2) + 2 * A014140((n - 2) // 2) + A401694(n - 1) - A014140(n - 2) - A014140(n - 1)
            else:
                countTotal = oeisIDfor_n('A014140', n)
        case _:
            countTotal = oeisIDfor_n('A014140', n)
    return countTotal

def A045445(n: int, f: Literal['A045445'] | LiteralString | None = None) -> int:
    # TODO Disambiguate the symmetry terminology in this generated OEIS description at its source.
    """
    Compute A045445(n) as a function of A045445.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A045445
    is "Number of nonisomorphic systems of catafusenes for the unsymmetrical schemes (group C_s) with two appendages (see references for precise definition).".

    The domain of A045445 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of nonisomorphic systems of catafusenes for the unsymmetrical schemes (group C_s) with two appendages (see references for precise definition).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A045445
    """
    countTotal: int
    match f:
        case 'A045445':
            if 3 <= n:
                countTotal = (n - 1) * (3 * (2 * n - 1) * A045445(n - 1) - 5 * (n - 2) * A045445(n - 2)) // ((n - 2) * (n + 2))
            else:
                countTotal = oeisIDfor_n('A045445', n)
        case _:
            countTotal = oeisIDfor_n('A045445', n)
    return countTotal

def A060206(n: int, f: Literal['A000682', 'A005315, A005316, and A077460'] | LiteralString | None = None) -> int:
    # TODO Disambiguate the symmetry terminology in this generated OEIS description at its source.
    """
    Compute A060206(n) as a function of A000682 or A005315, A005316, and A077460.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A060206
    is "Number of rotationally symmetric closed meanders of length 4n+2.".

    The domain of A060206 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 25.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of rotationally symmetric closed meanders of length 4n+2.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A060206
    """
    countTotal: int
    match f:
        case 'A000682':
            countTotal = A000682(2 * n + 1)
        case 'A005315, A005316, and A077460':
            if 1 <= n:
                countTotal = 4 * A077460(2 * n + 1) - A005315(2 * n + 1) - A005316(2 * n + 1)
            else:
                countTotal = oeisIDfor_n('A060206', n)
        case _:
            countTotal = oeisIDfor_n('A060206', n)
    return countTotal

def A068253(n: int, f: Literal['A078099'] | LiteralString | None = None) -> int:
    """
    Compute A068253(n) as a function of A078099.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A068253
    is "1/3 of the number of colorings of an n X n square array with 3 colors.".

    The domain of A068253 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 25.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        1/3 of the number of colorings of an n X n square array with 3 colors.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A068253
    """
    countTotal: int
    match f:
        case 'A078099':
            countTotal = getTriangle('A078099')[2 * n - 1][n - 1]
        case _:
            countTotal = oeisIDfor_n('A068253', n)
    return countTotal

def A076876(n: int, f: Literal['A400429', 'A005316 and A006661', 'A380367'] | LiteralString | None = None) -> int:
    """
    Compute A076876(n) as a function of A400429 or A005316 and A006661 or A380367.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A076876
    is "Meandric numbers for a river crossing two parallel roads at n points.".

    The domain of A076876 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 43.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Meandric numbers for a river crossing two parallel roads at n points.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A076876
    """
    countTotal: int
    match f:
        case 'A400429':
            if 6 <= n and n % 2 == 0:
                countTotal = (getTriangle('A400429')[n][3 - 1] + getTriangle('A400429')[n + 1][1 - 1] + getTriangle('A400429')[n + 2][2 - 1] - getTriangle('A400429')[n][2 - 1] - 2 * getTriangle('A400429')[n + 1][2 - 1]) // 2
            else:
                countTotal = oeisIDfor_n('A076876', n)
        case 'A005316 and A006661':
            if 2 <= n and n % 2 == 0:
                countTotal = A006661(n + 4) - A005316(n)
            else:
                countTotal = oeisIDfor_n('A076876', n)
        case 'A380367':
            countTotal = getTriangle('A380367')[n + 1][n - 0]
        case _:
            countTotal = oeisIDfor_n('A076876', n)
    return countTotal

def A077014(n: int, f: Literal['A005316'] | LiteralString | None = None) -> int:
    """
    Compute A077014(n) as a function of A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077014
    is "Number of ways that a directed line (or river) that starts in the south can cross an east-west road n times.".

    The domain of A077014 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 60.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of ways that a directed line (or river) that starts in the south can cross an east-west road n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A077014
    """
    countTotal: int
    match f:
        case 'A005316':
            countTotal = (2 - n % 2) * A005316(n)
        case _:
            countTotal = oeisIDfor_n('A077014', n)
    return countTotal

def A077054(n: int, f: Literal['A005316', 'A005315 and A085973', 'A400429', 'A380369'] | LiteralString | None = None) -> int:
    """
    Compute A077054(n) as a function of A005316 or A005315 and A085973 or A400429 or A380369.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077054
    is "Number of ways a river can cross a road 2n times.".

    The domain of A077054 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 30.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of ways a river can cross a road 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A077054
    """
    countTotal: int
    match f:
        case 'A005316':
            countTotal = A005316(2 * n)
        case 'A005315 and A085973':
            if 1 <= n:
                countTotal = A085973(n) - A005315(n)
            else:
                countTotal = oeisIDfor_n('A077054', n)
        case 'A400429':
            if 2 <= n:
                countTotal = (getTriangle('A400429')[2 * n + 1][1 - 1] - getTriangle('A400429')[2 * n][2 - 1]) // 2
            else:
                countTotal = oeisIDfor_n('A077054', n)
        case 'A380369':
            countTotal = sum(map(getitem, repeat(getTriangle('A380369')[n], len(range(n + 1))), map(0 .__rsub__, range(n + 1)), strict=True))
        case _:
            countTotal = oeisIDfor_n('A077054', n)
    return countTotal

def A077055(n: int, f: Literal['A078592', 'A005316', 'A005316 and A223096', 'A223096 and A400429', 'A400429'] | LiteralString | None = None) -> int:
    """
    Compute A077055(n) as a function of A078592 or A005316 or A005316 and A223096 or A223096 and A400429 or A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077055
    is "Call two meanders from A005316 equivalent if they differ by a reflection in the Y axis (if n even) or by reflections in the X or Y axes (if n odd). Sequence gives number of inequivalent meanders with n crossings.".

    The domain of A077055 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 60.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Call two meanders from A005316 equivalent if they differ by a reflection in the Y axis (if n even) or by reflections in the X or Y axes (if n odd). Sequence gives number of inequivalent meanders with n crossings.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A077055
    """
    countTotal: int
    match f:
        case 'A078592':
            if 2 <= n and n % 2 == 0:
                countTotal = A078592(n // 2)
            elif n <= 3:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A077055', n)
        case 'A005316':
            if 2 <= n and n % 2 == 0:
                countTotal = (A005316(n // 2) + A005316(n)) // 2
            elif n <= 3:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A077055', n)
        case 'A005316 and A223096':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = (A005316(n) + 2 * A223096((n - 1) // 2)) // 4
            elif n <= 3:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A077055', n)
        case 'A223096 and A400429':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = (2 * A223096((n - 1) // 2) + getTriangle('A400429')[n][1 - 1]) // 4
            elif n <= 3:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A077055', n)
        case 'A400429':
            if 6 <= n and n % 2 == 0:
                countTotal = ((n // 2 % 2 + 1 and (n // 2 % 2 + 1) * getTriangle('A400429')[(n + 2) // 2][1 - 1]) + getTriangle('A400429')[n + 1][1 - 1] - (1 - n // 2 % 2 and (1 - n // 2 % 2) * getTriangle('A400429')[n // 2][2 - 1]) - getTriangle('A400429')[n][2 - 1]) // 4
            elif n <= 3:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A077055', n)
        case _:
            if n <= 3:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A077055', n)
    return countTotal

def A077056(n: int, f: Literal['A380369'] | LiteralString | None = None) -> int:
    """
    Compute A077056(n) as a function of A380369.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077056
    is "Total number of "humps" in all A005316(2n) open meanders with 2n crossings.".

    The domain of A077056 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 23.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Total number of "humps" in all A005316(2n) open meanders with 2n crossings.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A077056
    """
    countTotal: int
    match f:
        case 'A380369':
            if 1 <= n:
                countTotal = sum(map(mul, range(1, n + 1), map(getitem, repeat(getTriangle('A380369')[n], len(range(1, n + 1))), map(0 .__rsub__, range(1, n + 1)), strict=True), strict=True))
            elif n <= 0:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A077056', n)
        case _:
            if n <= 0:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A077056', n)
    return countTotal

def A077460(n: int, f: Literal['A005315, A005316, and A060206'] | LiteralString | None = None) -> int:
    """
    Compute A077460(n) as a function of A005315, A005316, and A060206.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077460
    is "Number of closed meanders (loops crossing a road 2n times) that are nonisomorphic under reflection and rotation.".

    The domain of A077460 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 39.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of closed meanders (loops crossing a road 2n times) that are nonisomorphic under reflection and rotation.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A077460
    """
    countTotal: int
    match f:
        case 'A005315, A005316, and A060206':
            if 2 <= n:
                if n % 2 == 1:
                    countTotal = (A005315(n) + A005316(n) + A060206((n - 1) // 2)) // 4
                else:
                    countTotal = (A005315(n) + 2 * A005316(n)) // 4
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A077460', n)
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A077460', n)
    return countTotal

def A078591(n: int, f: Literal['A005315'] | LiteralString | None = None) -> int:
    """
    Compute A078591(n) as a function of A005315.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A078591
    is "Number of nonisomorphic ways a loop can cross a road (running East-West) 2n times.".

    The domain of A078591 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 39.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of nonisomorphic ways a loop can cross a road (running East-West) 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A078591
    """
    countTotal: int
    match f:
        case 'A005315':
            if 2 <= n:
                countTotal = A005315(n) // 2
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A078591', n)
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A078591', n)
    return countTotal

def A078592(n: int, f: Literal['A005316', 'A077055'] | LiteralString | None = None) -> int:
    """
    Compute A078592(n) as a function of A005316 or A077055.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A078592
    is "Call two meanders from A005316 with 2n crossings equivalent if they differ by reflections in the X or Y axes. Sequence gives number of inequivalent meanders.".

    The domain of A078592 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 30.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Call two meanders from A005316 with 2n crossings equivalent if they differ by reflections in the X or Y axes. Sequence gives number of inequivalent meanders.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A078592
    """
    countTotal: int
    match f:
        case 'A005316':
            countTotal = (A005316(n) + A005316(2 * n)) // 2
        case 'A077055':
            if 1 <= n:
                countTotal = A077055(2 * n)
            else:
                countTotal = oeisIDfor_n('A078592', n)
        case _:
            countTotal = oeisIDfor_n('A078592', n)
    return countTotal

def A085973(n: int, f: Literal['A005315 and A077054'] | LiteralString | None = None) -> int:
    """
    Compute A085973(n) as a function of A005315 and A077054.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A085973
    is "Number of ways a loop can cross two parallel roads 2n times.".

    The domain of A085973 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 30.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of ways a loop can cross two parallel roads 2n times.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A085973
    """
    countTotal: int
    match f:
        case 'A005315 and A077054':
            if 1 <= n:
                countTotal = A005315(n) + A077054(n)
            elif n <= 0:
                countTotal = 3
            else:
                countTotal = oeisIDfor_n('A085973', n)
        case _:
            if n <= 0:
                countTotal = 3
            else:
                countTotal = oeisIDfor_n('A085973', n)
    return countTotal

def A204352(n: int, f: Literal['A380367'] | LiteralString | None = None) -> int:
    """
    Compute A204352(n) as a function of A380367.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A204352
    is "Meandric numbers for a river crossing up to 3 parallel roads at n points.".

    The domain of A204352 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 41.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Meandric numbers for a river crossing up to 3 parallel roads at n points.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A204352
    """
    countTotal: int
    match f:
        case 'A380367':
            countTotal = getTriangle('A380367')[n + 2][n - 0]
        case _:
            countTotal = oeisIDfor_n('A204352', n)
    return countTotal

def A208357(n: int, f: Literal['A005315'] | LiteralString | None = None) -> int:
    """
    Compute A208357(n) as a function of A005315.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A208357
    is "Number of meanders of order 2n+1 (4*n+2 crossings of the infinite line) with central 1-1 cut.".

    The domain of A208357 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 38.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of meanders of order 2n+1 (4*n+2 crossings of the infinite line) with central 1-1 cut.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A208357
    """
    countTotal: int
    match f:
        case 'A005315':
            countTotal = A005315(n + 1) ** 2
        case _:
            countTotal = oeisIDfor_n('A208357', n)
    return countTotal

def A217310(n: int, f: Literal['A223093', 'A005316, A217318, and A227167'] | LiteralString | None = None) -> int:
    """
    Compute A217310(n) as a function of A223093 or A005316, A217318, and A227167.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A217310
    is "The number of meandering curves of order n with only one extremity covered by their arcs.".

    The domain of A217310 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 49.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        The number of meandering curves of order n with only one extremity covered by their arcs.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A217310
    """
    countTotal: int
    match f:
        case 'A223093':
            countTotal = (n % 2 + 1) * A223093(n)
        case 'A005316, A217318, and A227167':
            countTotal = A227167(n) - A005316(n) - A217318(n)
        case _:
            countTotal = oeisIDfor_n('A217310', n)
    return countTotal

def A217318(n: int, f: Literal['A223095', 'A005316, A217310, and A227167'] | LiteralString | None = None) -> int:
    """
    Compute A217318(n) as a function of A223095 or A005316, A217310, and A227167.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A217318
    is "The number of meandering curves of order n with both extremities covered by their arcs.".

    The domain of A217318 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 49.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        The number of meandering curves of order n with both extremities covered by their arcs.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A217318
    """
    countTotal: int
    match f:
        case 'A223095':
            countTotal = (n % 2 + 1) * A223095(n) // 2
        case 'A005316, A217310, and A227167':
            countTotal = A227167(n) - A005316(n) - A217310(n)
        case _:
            countTotal = oeisIDfor_n('A217318', n)
    return countTotal

def A223093(n: int, f: Literal['A217310', 'A000682 and A005316', 'A223094 and A223095'] | LiteralString | None = None) -> int:
    """
    Compute A223093(n) as a function of A217310 or A000682 and A005316 or A223094 and A223095.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A223093
    is "Number of foldings of n labeled stamps in which leaf 1 is inwards and leaf n outwards (or leaf 1 outwards and leaf n inwards).".

    The domain of A223093 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 49.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of foldings of n labeled stamps in which leaf 1 is inwards and leaf n outwards (or leaf 1 outwards and leaf n inwards).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A223093
    """
    countTotal: int
    match f:
        case 'A217310':
            countTotal = A217310(n) // (n % 2 + 1)
        case 'A000682 and A005316':
            countTotal = A000682(n + 1) - (2 - n % 2) * A005316(n)
        case 'A223094 and A223095':
            countTotal = A223094(n) - A223095(n)
        case _:
            countTotal = oeisIDfor_n('A223093', n)
    return countTotal

def A223094(n: int, f: Literal['A000682', 'A223093 and A223095'] | LiteralString | None = None) -> int:
    """
    Compute A223094(n) as a function of A000682 or A223093 and A223095.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A223094
    is "Number of foldings of n labeled stamps in which leaf n is inwards.".

    The domain of A223094 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 49.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of foldings of n labeled stamps in which leaf n is inwards.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A223094
    """
    countTotal: int
    match f:
        case 'A000682':
            countTotal = n * A000682(n) - A000682(n + 1)
        case 'A223093 and A223095':
            countTotal = A223093(n) + A223095(n)
        case _:
            countTotal = oeisIDfor_n('A223094', n)
    return countTotal

def A223095(n: int, f: Literal['A217318', 'A223093 and A223094'] | LiteralString | None = None) -> int:
    """
    Compute A223095(n) as a function of A217318 or A223093 and A223094.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A223095
    is "Number of foldings of n labeled stamps in which both end leaves are inwards.".

    The domain of A223095 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 49.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of foldings of n labeled stamps in which both end leaves are inwards.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A223095
    """
    countTotal: int
    match f:
        case 'A217318':
            countTotal = 2 * A217318(n) // (n % 2 + 1)
        case 'A223093 and A223094':
            countTotal = A223094(n) - A223093(n)
        case _:
            countTotal = oeisIDfor_n('A223095', n)
    return countTotal

def A223096(n: int, f: Literal['A005316 and A077055', 'A077055 and A400429'] | LiteralString | None = None) -> int:
    # TODO Disambiguate the symmetry terminology in this generated OEIS description at its source.
    """
    Compute A223096(n) as a function of A005316 and A077055 or A077055 and A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A223096
    is "Number of symmetric meander shapes with 2n+1 crossings.".

    The domain of A223096 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 32.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of symmetric meander shapes with 2n+1 crossings.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A223096
    """
    countTotal: int
    match f:
        case 'A005316 and A077055':
            if 1 <= n:
                countTotal = (4 * A077055(2 * n + 1) - A005316(2 * n + 1)) // 2
            else:
                countTotal = oeisIDfor_n('A223096', n)
        case 'A077055 and A400429':
            if 1 <= n:
                countTotal = (4 * A077055(2 * n + 1) - getTriangle('A400429')[2 * n + 1][1 - 1]) // 2
            else:
                countTotal = oeisIDfor_n('A223096', n)
        case _:
            countTotal = oeisIDfor_n('A223096', n)
    return countTotal

def A227167(n: int, f: Literal['A000682', 'A005316, A217310, and A217318'] | LiteralString | None = None) -> int:
    """
    Compute A227167(n) as a function of A000682 or A005316, A217310, and A217318.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A227167
    is "The number of meandering curves of order n.".

    The domain of A227167 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 50.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        The number of meandering curves of order n.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A227167
    """
    countTotal: int
    match f:
        case 'A000682':
            countTotal = n * A000682(n) // (2 - n % 2)
        case 'A005316, A217310, and A217318':
            countTotal = A005316(n) + A217310(n) + A217318(n)
        case _:
            countTotal = oeisIDfor_n('A227167', n)
    return countTotal

def A259700(n: int, f: Literal['A259698'] | LiteralString | None = None) -> int:
    """
    Compute A259700(n) as a function of A259698.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A259700
    is "a(n) = Sum_{k=2..n-1} k*A259698(n,k).".

    The domain of A259700 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 15.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        a(n) = Sum_{k=2..n-1} k*A259698(n,k).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A259700
    """
    countTotal: int
    match f:
        case 'A259698':
            countTotal = sum(map(mul, range(1, n), map(getitem, repeat(getTriangle('A259698')[n], len(range(1, n))), map(1 .__rsub__, range(1, n)), strict=True), strict=True))
        case _:
            countTotal = oeisIDfor_n('A259700', n)
    return countTotal

def A259702(n: int, f: Literal['A000682', 'A301620', 'A259701'] | LiteralString | None = None) -> int:
    """
    Compute A259702(n) as a function of A000682 or A301620 or A259701.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A259702
    is "Row sums of A259701 except first column.".

    The domain of A259702 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 50.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Row sums of A259701 except first column.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A259702
    """
    countTotal: int
    match f:
        case 'A000682':
            if 3 <= n:
                countTotal = (A000682(n) - 2 * A000682(n - 1)) // 2
            elif n <= 2:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A259702', n)
        case 'A301620':
            if 3 <= n:
                countTotal = A301620(n - 2) // 2
            elif n <= 2:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A259702', n)
        case 'A259701':
            countTotal = sum(map(getitem, repeat(getTriangle('A259701')[n], len(range(2, n))), map(1 .__rsub__, range(2, n)), strict=True))
        case _:
            if n <= 2:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A259702', n)
    return countTotal

def A260785(n: int, f: Literal['A259689'] | LiteralString | None = None) -> int:
    """
    Compute A260785(n) as a function of A259689.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A260785
    is "Leading diagonal of triangle in A259689: a(n)= number of permutations without overlaps that generate exactly 2 permutations without overlaps in a(n+1).".

    The domain of A260785 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 27.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Leading diagonal of triangle in A259689: a(n)= number of permutations without overlaps that generate exactly 2 permutations without overlaps in a(n+1).

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A260785
    """
    countTotal: int
    match f:
        case 'A259689':
            countTotal = getTriangle('A259689')[n][2 - 2]
        case _:
            countTotal = oeisIDfor_n('A260785', n)
    return countTotal

def A301620(n: int, f: Literal['A000682', 'A301620 and A334615', 'A339179', 'A259702', 'A259689', 'A259689 and A322507'] | LiteralString | None = None) -> int:
    """
    Compute A301620(n) as a function of A000682 or A301620 and A334615 or A339179 or A259702 or A259689 or A259689 and A322507.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A301620
    is "a(n) is the total number of top arches with exactly one covering arch for semi-meanders with n top arches.".

    The domain of A301620 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 48.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        a(n) is the total number of top arches with exactly one covering arch for semi-meanders with n top arches.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A301620
    """
    countTotal: int
    match f:
        case 'A000682':
            countTotal = A000682(n + 2) - 2 * A000682(n + 1)
        case 'A301620 and A334615':
            if 2 <= n:
                countTotal = 2 * A301620(n - 1) + A334615(n + 2)
            else:
                countTotal = oeisIDfor_n('A301620', n)
        case 'A339179':
            countTotal = sum(map(getitem, repeat(getTriangle('A339179')[n + 3], len(range(4, (n + 3) // 2 + 2))), map(2 .__rsub__, range(4, (n + 3) // 2 + 2)), strict=True))
        case 'A259702':
            countTotal = 2 * A259702(n + 2)
        case 'A259689':
            countTotal = sum(map(mul, map(2 .__rsub__, range(2, (n + 1) // 2 + 2)), map(getitem, repeat(getTriangle('A259689')[n + 1], len(range(2, (n + 1) // 2 + 2))), map(2 .__rsub__, range(2, (n + 1) // 2 + 2)), strict=True), strict=True))
        case 'A259689 and A322507':
            if 3 <= n:
                countTotal = A322507(n - 1) + sum(map(mul, map(mul, range(2, (n - 1) // 2 + 2), map(1 .__rsub__, range(2, (n - 1) // 2 + 2)), strict=True), map(getitem, repeat(getTriangle('A259689')[n - 1], len(range(2, (n - 1) // 2 + 2))), map(2 .__rsub__, range(2, (n - 1) // 2 + 2)), strict=True), strict=True))
            else:
                countTotal = oeisIDfor_n('A301620', n)
        case _:
            countTotal = oeisIDfor_n('A301620', n)
    return countTotal

def A322507(n: int, f: Literal['A259689 and A301620'] | LiteralString | None = None) -> int:
    """
    Compute A322507(n) as a function of A259689 and A301620.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A322507
    is "a(n) is the number of arches for semi-meanders with n top arches that have exactly two covering arches.".

    The domain of A322507 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 27.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        a(n) is the number of arches for semi-meanders with n top arches that have exactly two covering arches.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A322507
    """
    countTotal: int
    match f:
        case 'A259689 and A301620':
            if 2 <= n:
                countTotal = A301620(n + 1) - sum(map(mul, map(mul, range(2, n // 2 + 2), map(1 .__rsub__, range(2, n // 2 + 2)), strict=True), map(getitem, repeat(getTriangle('A259689')[n], len(range(2, n // 2 + 2))), map(2 .__rsub__, range(2, n // 2 + 2)), strict=True), strict=True))
            elif n <= 1:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A322507', n)
        case _:
            if n <= 1:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A322507', n)
    return countTotal

def A333971(n: int, f: Literal['A000682'] | LiteralString | None = None) -> int:
    """
    Compute A333971(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A333971
    is "a(n) is the number of semi-meanders with n top arches that have at least one arch with length 1 adjacent to the center of the top arch configuration or at either end of the arch configuration.".

    The domain of A333971 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 51.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        a(n) is the number of semi-meanders with n top arches that have at least one arch with length 1 adjacent to the center of the top arch configuration or at either end of the arch configuration.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A333971
    """
    countTotal: int
    match f:
        case 'A000682':
            if 4 <= n:
                countTotal = 4 * A000682(n - 1) - 4 * A000682(n - 2)
            elif n <= 3:
                countTotal = n - 1
            else:
                countTotal = oeisIDfor_n('A333971', n)
        case _:
            if n <= 3:
                countTotal = n - 1
            else:
                countTotal = oeisIDfor_n('A333971', n)
    return countTotal

def A334615(n: int, f: Literal['A000682', 'A301620'] | LiteralString | None = None) -> int:
    """
    Compute A334615(n) as a function of A000682 or A301620.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A334615
    is "a(n) is the number of semi-meanders with n top arches that has no arch of length 1 at the ends of the top arch configuration and no arch of length 1 adjacent to the center of the top arch configuration.".

    The domain of A334615 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 50.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        a(n) is the number of semi-meanders with n top arches that has no arch of length 1 at the ends of the top arch configuration and no arch of length 1 adjacent to the center of the top arch configuration.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A334615
    """
    countTotal: int
    match f:
        case 'A000682':
            if 4 <= n:
                countTotal = A000682(n) + 4 * A000682(n - 2) - 4 * A000682(n - 1)
            elif n <= 3:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A334615', n)
        case 'A301620':
            if 4 <= n:
                countTotal = A301620(n - 2) - 2 * A301620(n - 3)
            elif n <= 3:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A334615', n)
        case _:
            if n <= 3:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A334615', n)
    return countTotal

def A337581(n: int, f: Literal['A000682'] | LiteralString | None = None) -> int:
    """
    Compute A337581(n) as a function of A000682.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A337581
    is "a(n) is the number of semi-meanders with n top arches that have both an arch of length 1 adjacent to the center of the top arch configuration and an arch of length 1 starting or ending the top arch configuration.".

    The domain of A337581 starts at 2, therefore for values of `n` < 2, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 52.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        a(n) is the number of semi-meanders with n top arches that have both an arch of length 1 adjacent to the center of the top arch configuration and an arch of length 1 starting or ending the top arch configuration.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A337581
    """
    countTotal: int
    match f:
        case 'A000682':
            if 4 <= n:
                countTotal = 4 * A000682(n - 2)
            elif n <= 3:
                countTotal = n - 1
            else:
                countTotal = oeisIDfor_n('A337581', n)
        case _:
            if n <= 3:
                countTotal = n - 1
            else:
                countTotal = oeisIDfor_n('A337581', n)
    return countTotal

def A400630(n: int, f: Literal['A400429', 'A400630'] | LiteralString | None = None) -> int:
    """
    Compute A400630(n) as a function of A400429 or A400630.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400630
    is "Number of semi-meanders with n crossings and winding number n-4, for n >= 4.".

    The domain of A400630 starts at 4, therefore for values of `n` < 4, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 10001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-4, for n >= 4.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400630
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 1 - 1]
        case 'A400630':
            if 6 <= n:
                countTotal = 2 * n + A400630(n - 2)
            elif True:
                countTotal = (n ** 2 + 2 * n + n % 2 - 20) // 2
            else:
                countTotal = oeisIDfor_n('A400630', n)
        case _:
            if True:
                countTotal = (n ** 2 + 2 * n + n % 2 - 20) // 2
            else:
                countTotal = oeisIDfor_n('A400630', n)
    return countTotal

def A400631(n: int, f: Literal['A400429'] | LiteralString | None = None) -> int:
    """
    Compute A400631(n) as a function of A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400631
    is "Number of semi-meanders with n crossings and winding number n-6, for n >= 6.".

    The domain of A400631 starts at 6, therefore for values of `n` < 6, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 10006.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-6, for n >= 6.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400631
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 2 - 1]
        case _:
            if True:
                countTotal = (13 * n ** 4 + 24 * n ** 3 + 9 * (2 * n ** 2 + 21 - 8 * n) * (n % 2) + 17920 - 284 * n ** 2 - 4560 * n - 64 * 0 ** (n % 3)) // 288
            else:
                countTotal = oeisIDfor_n('A400631', n)
    return countTotal

def A400632(n: int, f: Literal['A400429'] | LiteralString | None = None) -> int:
    """
    Compute A400632(n) as a function of A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400632
    is "Number of semi-meanders with n crossings and winding number n-8, for n >= 8.".

    The domain of A400632 starts at 8, therefore for values of `n` < 8, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 301.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-8, for n >= 8.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400632
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 3 - 1]
        case _:
            if True:
                countTotal = (294 * n ** 6 + 16955410 * n + 20480 * 0 ** (n % 3) * (n - 2) + 25920 * 0 ** (n % 8) + 61440 * 0 ** ((n + 2) % 3) + 43200 * 0 ** ((n + 3) % 8) + 43200 * 0 ** ((n + 6) % 8) + 45 * (1 - 2 * (n % 2)) * (44 * n ** 3 + 358 * n + 4473 - 5 * n ** 4 - 289 * n ** 2) - 60 * n ** 5 - 3895 * n ** 4 - 232300 * n ** 3 - 326739 * n ** 2 - 20480 * 0 ** ((n + 1) % 3) * (n + 1) - 8640 * 0 ** ((n + 1) % 8) - 25920 * 0 ** ((n + 2) % 8) - 43200 * 0 ** ((n + 4) % 8) - 8640 * 0 ** ((n + 5) % 8) - 25920 * 0 ** ((n + 7) % 8) - 48570245) // 138240
            else:
                countTotal = oeisIDfor_n('A400632', n)
    return countTotal

def A400633(n: int, f: Literal['A400429'] | LiteralString | None = None) -> int:
    """
    Compute A400633(n) as a function of A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400633
    is "Number of semi-meanders with n crossings and winding number n-10, for n >= 10.".

    The domain of A400633 starts at 10, therefore for values of `n` < 10, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 301.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-10, for n >= 10.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400633
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 4 - 1]
        case _:
            if True:
                countTotal = (5163390 * n ** 8 + 56558460 * n ** 6 + 600812982000 * n ** 3 + 5767984511820 * n ** 2 + 163296000 * n * (40 * 0 ** (n % 4) + 33 * 0 ** ((n + 1) % 4) - 40 * 0 ** ((n + 2) % 4) - 33 * 0 ** ((n + 3) % 4)) + 458752000 * 0 ** (n % 3) * (108 * n + 50 - 9 * n ** 2) + 18856191452 * 0 ** (n % 5) + 2612736000 * 0 ** (n % 8) * (-(17 * n) - 14) + 458752000 * 0 ** ((n + 1) % 3) * (18 * n ** 2 - 15 * n - 61) + 18856191452 * 0 ** ((n + 1) % 5) + 2612736000 * 0 ** ((n + 1) % 8) * (9 * n + 37) + 458752000 * 0 ** ((n + 2) % 3) * (11 - 9 * n ** 2 - 93 * n) + 117006080547 * 0 ** ((n + 2) % 5) + 7838208000 * 0 ** ((n + 3) % 8) * (11 - 3 * n) + 117006080547 * 0 ** ((n + 4) % 5) + 2612736000 * 0 ** ((n + 4) % 8) * (17 * n + 14) + 2612736000 * 0 ** ((n + 5) % 8) * (-(9 * n) - 37) + 156764160000 * 0 ** ((n + 6) % 8) + 7838208000 * 0 ** ((n + 7) % 8) * (3 * n - 11) + 8505 * (1 - 2 * (n % 2)) * (6992 * n ** 5 + 346000 * n ** 3 + 519600 * n + 17692977 - 476 * n ** 6 - 73490 * n ** 4 - 68012 * n ** 2) + 166068141699607 - 20844000 * n ** 7 - 6784359120 * n ** 5 - 31881482430 * n ** 4 - 68694275761200 * n - 156764160000 * 0 ** ((n + 2) % 8) - 41803776000 * 0 ** ((n + 3) % 5)) // 83607552000
            else:
                countTotal = oeisIDfor_n('A400633', n)
    return countTotal

def A400634(n: int, f: Literal['A400429'] | LiteralString | None = None) -> int:
    """
    Compute A400634(n) as a function of A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400634
    is "Number of semi-meanders with n crossings and winding number n-12, for n >= 12.".

    The domain of A400634 starts at 12, therefore for values of `n` < 12, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 10012.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-12, for n >= 12.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400634
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 5 - 1]
        case _:
            countTotal = oeisIDfor_n('A400634', n)
    return countTotal

def A400635(n: int, f: Literal['A400429'] | LiteralString | None = None) -> int:
    """
    Compute A400635(n) as a function of A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400635
    is "Number of semi-meanders with n crossings and winding number n-14, for n >= 14.".

    The domain of A400635 starts at 14, therefore for values of `n` < 14, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 10014.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-14, for n >= 14.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400635
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 6 - 1]
        case _:
            countTotal = oeisIDfor_n('A400635', n)
    return countTotal

def A400636(n: int, f: Literal['A400429'] | LiteralString | None = None) -> int:
    """
    Compute A400636(n) as a function of A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400636
    is "Number of semi-meanders with n crossings and winding number n-16, for n >= 16.".

    The domain of A400636 starts at 16, therefore for values of `n` < 16, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 10016.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-16, for n >= 16.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400636
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 7 - 1]
        case _:
            countTotal = oeisIDfor_n('A400636', n)
    return countTotal

def A400637(n: int, f: Literal['A400429'] | LiteralString | None = None) -> int:
    """
    Compute A400637(n) as a function of A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400637
    is "Number of semi-meanders with n crossings and winding number n-18, for n >= 18.".

    The domain of A400637 starts at 18, therefore for values of `n` < 18, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 10018.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-18, for n >= 18.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400637
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 8 - 1]
        case _:
            countTotal = oeisIDfor_n('A400637', n)
    return countTotal

def A400638(n: int, f: Literal['A400429'] | LiteralString | None = None) -> int:
    """
    Compute A400638(n) as a function of A400429.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A400638
    is "Number of semi-meanders with n crossings and winding number n-20, for n >= 20.".

    The domain of A400638 starts at 20, therefore for values of `n` < 20, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 10020.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of semi-meanders with n crossings and winding number n-20, for n >= 20.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A400638
    """
    countTotal: int
    match f:
        case 'A400429':
            countTotal = getTriangle('A400429')[n][n // 2 - 9 - 1]
        case _:
            countTotal = oeisIDfor_n('A400638', n)
    return countTotal

def A401693(n: int, f: Literal['A014140', 'A000108, A014137, and A401693', 'A000108 and A401693', 'A014137, A401693, and A401694'] | LiteralString | None = None) -> int:
    """
    Compute A401693(n) as a function of A014140 or A000108, A014137, and A401693 or A000108 and A401693 or A014137, A401693, and A401694.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A401693
    is "Number of ways to cut a nonempty Dyck word into a prefix and a suffix with both lengths at most n and of the same parity as n.".

    The domain of A401693 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of ways to cut a nonempty Dyck word into a prefix and a suffix with both lengths at most n and of the same parity as n.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A401693
    """
    countTotal: int
    match f:
        case 'A014140':
            if 2 <= n and n % 2 == 0:
                countTotal = A014140(n) - 2 * A014140((n - 2) // 2) - 1
            else:
                countTotal = oeisIDfor_n('A401693', n)
        case 'A014140':
            if (n + 1) % 2 == 0:
                countTotal = A014140(n) - 2 * A014140((n - 1) // 2)
            else:
                countTotal = oeisIDfor_n('A401693', n)
        case 'A000108, A014137, and A401693':
            if 4 <= n and n % 2 == 0:
                countTotal = 2 * A014137(n) + A401693(n - 2) - A000108(n) - 2 * A014137((n - 2) // 2)
            else:
                countTotal = oeisIDfor_n('A401693', n)
        case 'A000108, A014137, and A401693':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = 2 * A014137(n) + A401693(n - 2) - A000108(n) - 2 * A014137((n - 1) // 2)
            else:
                countTotal = oeisIDfor_n('A401693', n)
        case 'A000108 and A401693':
            if 6 <= n and n % 2 == 0:
                countTotal = (2 * (n + 1) * A401693(n - 2) + 5 * (5 * n - 7) * A000108(n - 2) - 2 * (n + 1) * A000108((n - 2) // 2) - (n + 1) * A401693(n - 4)) // (n + 1)
            else:
                countTotal = oeisIDfor_n('A401693', n)
        case 'A000108 and A401693':
            if 5 <= n and (n + 1) % 2 == 0:
                countTotal = (2 * (n + 1) * A401693(n - 2) + 5 * (5 * n - 7) * A000108(n - 2) - 2 * (n + 1) * A000108((n - 1) // 2) - (n + 1) * A401693(n - 4)) // (n + 1)
            else:
                countTotal = oeisIDfor_n('A401693', n)
        case 'A014137, A401693, and A401694':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = 2 * A014137((n - 1) // 2) + A401694(n - 1) - A401693(n - 2) - A401693(n - 1) - 1
            else:
                countTotal = oeisIDfor_n('A401693', n)
        case 'A014137, A401693, and A401694':
            if 4 <= n and n % 2 == 0:
                countTotal = 2 * A014137(n // 2) + A401694(n - 1) - A401693(n - 2) - A401693(n - 1) - 2
            else:
                countTotal = oeisIDfor_n('A401693', n)
        case _:
            countTotal = oeisIDfor_n('A401693', n)
    return countTotal

def A401694(n: int, f: Literal['A014140', 'A014137 and A401693'] | LiteralString | None = None) -> int:
    """
    Compute A401694(n) as a function of A014140 or A014137 and A401693.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A401694
    is "Number of ways to apply one of four moves (see Comments) to a nonempty Dyck word with a cut, where both sides of the cut have lengths at most n+1 and of the same parity as n+1 before the move and at most n after the move.".

    The domain of A401694 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 1001.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        Number of ways to apply one of four moves (see Comments) to a nonempty Dyck word with a cut, where both sides of the cut have lengths at most n+1 and of the same parity as n+1 before the move and at most n after the move.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A401694
    """
    countTotal: int
    match f:
        case 'A014140':
            if 2 <= n and n % 2 == 0:
                countTotal = A014140(n) + A014140(n - 1) + A014140(n + 1) - 4 * A014140(n // 2) - 2 * A014140((n - 2) // 2)
            elif n <= 1:
                countTotal = 3
            else:
                countTotal = oeisIDfor_n('A401694', n)
        case 'A014140':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = A014140(n) + A014140(n - 1) + A014140(n + 1) - 2 * A014140((n - 3) // 2) - 2 * A014140((n - 1) // 2) - 2 * A014140((n + 1) // 2)
            elif n <= 1:
                countTotal = 3
            else:
                countTotal = oeisIDfor_n('A401694', n)
        case 'A014137 and A401693':
            if 2 <= n and n % 2 == 0:
                countTotal = A401693(n) + A401693(n - 1) + A401693(n + 1) + 1 - 2 * A014137(n // 2)
            elif n <= 1:
                countTotal = 3
            else:
                countTotal = oeisIDfor_n('A401694', n)
        case 'A014137 and A401693':
            if 3 <= n and (n + 1) % 2 == 0:
                countTotal = A401693(n) + A401693(n - 1) + A401693(n + 1) + 2 - 2 * A014137((n + 1) // 2)
            elif n <= 1:
                countTotal = 3
            else:
                countTotal = oeisIDfor_n('A401694', n)
        case _:
            if n <= 1:
                countTotal = 3
            else:
                countTotal = oeisIDfor_n('A401694', n)
    return countTotal
