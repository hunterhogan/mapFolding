"""Compute a(n) for an OEIS ID by computing other OEIS IDs.

TODO Implement A178961 for unknown values of A001010
TODO A223094 For n >= 3: a(n) = n! - Sum_{k=3..n-1} (a(k)*n!/k!) - A000682(n+1). - _Roger Ford_, Aug 24 2024
TODO A301620 a(n) = Sum_{k=3..floor((n+3)/2)} (A259689(n+1,k)*(k-2)). - _Roger Ford_, Dec 10 2018

NOTE: This is a generated file; edit the source file.
"""
from functools import cache
from mapFolding import dictionaryOEIS
from mapFolding.basecamp import NOTcountingFolds

@cache
def A000136(n: int) -> int:
    """
    Compute A000136(n) as a function of A000682.

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
    return n * _A000682(n)

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

def A001010(n: int) -> int:
    """
    Compute A001010(n) as a function of A000682 or A007822.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of A001010 is: "Number of symmetric foldings of a strip of n stamps."

    The domain of A001010 starts at 1, therefore for values of `n` < 1, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is 53.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        Number of symmetric foldings of a strip of n stamps.

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/A001010
    """
    if n == 1:
        countTotal = 1
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
        countTotal = 1
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
        countTotal = 1
    else:
        countTotal = _A005316(2 * n - 1)
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
        countTotal = 1
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
        countTotal = 1
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
    for n下i in range(1, n + 1):
        countTotal += A001010valuesKnown[n下i]
    return countTotal

def A223094(n: int) -> int:
    """
    Compute A223094(n) as a function of A000136 and A000682.

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
    return A000136(n) - _A000682(n + 1)

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
        countTotal = 0
    else:
        countTotal = _A000682(n) // 2 - _A000682(n - 1)
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
    return NOTcountingFolds('A000682', n)

def _A007822(n: int) -> int:
    return NOTcountingFolds('A007822', n)

@cache
def _A005316(n: int) -> int:
    return NOTcountingFolds('A005316', n)
