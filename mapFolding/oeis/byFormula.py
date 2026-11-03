"""Compute a(n) for an OEIS ID by computing other OEIS IDs.

This is a generated file; edit `mapFolding.oeis._symbolic.boxOfRelations` and run `easyRun/synthesizeModules.py`.
"""
from __future__ import annotations

from mapFolding.oeis import oeisIDfor_n
from math import factorial, isqrt
from typing import Literal, LiteralString

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

def A000560(n: int, f: Literal['A000682'] | LiteralString | None = None) -> int:
    """
    Compute A000560(n) as a function of A000682.

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
        case _:
            countTotal = oeisIDfor_n('A000560', n)
    return countTotal

def A000682(n: int, f: Literal['A000136', 'A000560', 'A001010', 'A001010 and A001011', 'A060206', 'A005316 and A223093', 'A000682 and A223094', 'A227167', 'A000682 and A259702', 'A000682 and A301620', 'A000682 and A333971', 'A000682 and A334615', 'A337581', 'A223094', 'A301620'] | LiteralString | None = None) -> int:
    """
    Compute A000682(n) as a function of A000136 or A000560 or A001010 or A001010 and A001011 or A060206 or A005316 and A223093 or A000682 and A223094 or A227167 or A000682 and A259702 or A000682 and A301620 or A000682 and A333971 or A000682 and A334615 or A337581 or A223094 or A301620.

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
        case 'A060206':
            if (n + 1) % 2 == 0:
                countTotal = A060206((n - 1) // 2)
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
        case 'A000682 and A259702':
            if 3 <= n:
                countTotal = 2 * A000682(n - 1) + 2 * A259702(n)
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
        case 'A223094':
            if 4 <= n:
                countTotal = factorial(n - 1) - sum(map(lambda k: A223094(k) * factorial(n - 1) // factorial(k), range(3, n)))
            elif n <= 2:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A000682', n)
        case 'A301620':
            if 3 <= n:
                countTotal = 2 ** (n - 2) + sum(map(lambda k: 2 ** (n - k - 2) * A301620(k), range(3, n - 1)))
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

def A005315(n: int, f: Literal['A005316', 'A077054 and A085973', 'A005316, A060206, and A077460', 'A078591', 'A208357'] | LiteralString | None = None) -> int:
    """
    Compute A005315(n) as a function of A005316 or A077054 and A085973 or A005316, A060206, and A077460 or A078591 or A208357.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A005315
    is "Closed meandric numbers (or meanders): number of ways a loop can cross a road 2n times.".

    The domain of A005315 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
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
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005315', n)
    return countTotal

def A005316(n: int, f: Literal['A005315', 'A077054', 'A077014', 'A005315, A060206, and A077460', 'A005316 and A078592', 'A000682 and A223093', 'A217310, A217318, and A227167'] | LiteralString | None = None) -> int:
    """
    Compute A005316(n) as a function of A005315 or A077054 or A077014 or A005315, A060206, and A077460 or A005316 and A078592 or A000682 and A223093 or A217310, A217318, and A227167.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A005316
    is "Meandric numbers: number of ways a river can cross a road n times.".

    The domain of A005316 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 59.

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
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A005316', n)
    return countTotal

def A007822(n: int, f: Literal['A001010'] | LiteralString | None = None) -> int:
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
            elif n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A007822', n)
        case _:
            if n <= 1:
                countTotal = 1
            else:
                countTotal = oeisIDfor_n('A007822', n)
    return countTotal

def A060206(n: int, f: Literal['A000682', 'A005315, A005316, and A077460'] | LiteralString | None = None) -> int:
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

def A077014(n: int, f: Literal['A005316'] | LiteralString | None = None) -> int:
    """
    Compute A077014(n) as a function of A005316.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077014
    is "Number of ways that a directed line (or river) that starts in the south can cross an east-west road n times.".

    The domain of A077014 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 59.

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

def A077054(n: int, f: Literal['A005316', 'A005315 and A085973'] | LiteralString | None = None) -> int:
    """
    Compute A077054(n) as a function of A005316 or A005315 and A085973.

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
        case _:
            countTotal = oeisIDfor_n('A077054', n)
    return countTotal

def A077460(n: int, f: Literal['A005315, A005316, and A060206'] | LiteralString | None = None) -> int:
    """
    Compute A077460(n) as a function of A005315, A005316, and A060206.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A077460
    is "Number of closed meanders (loops crossing a road 2n times) that are nonisomorphic under reflection and rotation.".

    The domain of A077460 starts at 0, therefore for values of `n` < 0, a(n) is undefined. The smallest value of n for which a(n)
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

def A078592(n: int, f: Literal['A005316'] | LiteralString | None = None) -> int:
    """
    Compute A078592(n) as a function of A005316.

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

def A208357(n: int, f: Literal['A005315'] | LiteralString | None = None) -> int:
    """
    Compute A208357(n) as a function of A005315.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A208357
    is "Number of meanders of order 2n+1 (4*n+2 crossings of the infinite line) with central 1-1 cut.".

    The domain of A208357 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 29.

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

def A259702(n: int, f: Literal['A000682', 'A301620'] | LiteralString | None = None) -> int:
    """
    Compute A259702(n) as a function of A000682 or A301620.

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
        case _:
            if n <= 2:
                countTotal = 0
            else:
                countTotal = oeisIDfor_n('A259702', n)
    return countTotal

def A301620(n: int, f: Literal['A000682', 'A259702', 'A301620 and A334615'] | LiteralString | None = None) -> int:
    """
    Compute A301620(n) as a function of A000682 or A259702 or A301620 and A334615.

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
        case 'A259702':
            countTotal = 2 * A259702(n + 2)
        case 'A301620 and A334615':
            if 2 <= n:
                countTotal = 2 * A301620(n - 1) + A334615(n + 2)
            else:
                countTotal = oeisIDfor_n('A301620', n)
        case _:
            countTotal = oeisIDfor_n('A301620', n)
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
