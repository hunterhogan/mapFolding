"""Directly based on code Chai Wah Wu, https://oeis.org/wiki/User:Chai_Wah_Wu, posted on OEIS."""
from fractions import Fraction
from functools import cache
from itertools import combinations
from math import factorial, gcd
from sympy import divisors
from sympy.functions.combinatorial.numbers import mobius
from sympy.utilities.iterables import partitions

@cache
def _GoCountryDancing(romeo: int, copiesRomeo: int, sierra: int, copiesSierra: int) -> int:
	return copiesRomeo * copiesSierra * gcd(romeo, sierra)

@cache
def _goRight(integer: int, copies: int) -> int:
	return ((integer - 1) >> 1) * copies + (integer * copies * (copies - 1) >> 1)

@cache
def _deFactorial(integer: int, copies: int) -> int:
	return integer ** copies * factorial(copies)

@cache
def _blender(n: int) -> int:
	sumReBletionary: int = 0
	for partitionary in partitions(n):
		numbinations: int = 0
		nummaNumma: int = 0
		denominator: int = 1
		for (romeo, copiesRomeo), (sierra, copiesSierra) in combinations(partitionary.items(), 2):
			numbinations += _GoCountryDancing(romeo, copiesRomeo, sierra, copiesSierra)
		for integer, copies in partitionary.items():
			nummaNumma += _goRight(integer, copies)
			denominator *= _deFactorial(integer, copies)
		numerator: int = 3 ** (numbinations + nummaNumma)
		sumReBletionary += Fraction(numerator, denominator)
	return sumReBletionary

@cache
def _recurser(n: int) -> int:
	sumReBlender: int = 0
	for k in range(1, n):
		sumReBlender += _recurser(k) * _blender(n - k)
	return n * _blender(n) - sumReBlender

def A086345(n: int) -> int:
	"""Compute 'Number of connected oriented graphs (i.e., connected directed graphs with no bidirected edges) on n nodes'.

	Parameters
	----------
	n : int
		Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

	Returns
	-------
	a(n) : int
		Number of connected oriented graphs (i.e., connected directed graphs with no bidirected edges) on n nodes.

	Notes
	-----
	- The largest performance gains are from caching the recursion and `sympy.utilities.iterables.partitions` functions.
	- Interestingly, there is a small but noticeable penalty for choosing comprehension instead of `for`.

	Would You Like to Know More?
	----------------------------
	OEIS : webpage
		https://oeis.org/A086345
	"""
	if n == 0:
		aOFn: int = 1
	else:
		aOFn: int = 0
		for aDivisor in divisors(n, generator=True):
			aOFn += mobius(aDivisor) * _recurser(n//aDivisor)
		aOFn //= n
	return aOFn
