# ruff: noqa: ANN001, T201, D103, D100, ANN201, ANN202
# pyright: standard
from fractions import Fraction
from functools import cache
from itertools import combinations
from math import factorial, gcd, prod
from sympy import divisors
from sympy.functions.combinatorial.numbers import mobius
from sympy.utilities.iterables import partitions
import time

# NOTE Because `b` and `c` are inside `A086345` instead of peers, the run time is increased by a factor of 6.

def A086345(n):
	@cache
	def b(n): return int(sum(Fraction(3**(sum(p[r]*p[s]*gcd(r, s) for r, s in combinations(p.keys(), 2))+sum((q-1>>1)*r+(q*r*(r-1)>>1) for q, r in p.items())), prod(q**r*factorial(r) for q, r in p.items())) for p in partitions(n))) # pyright: ignore[reportAttributeAccessIssue, reportOperatorIssue]
	@cache
	def c(n): return n*b(n)-sum(c(k)*b(n-k) for k in range(1, n))
	return sum(mobius(d)*c(n//d) for d in divisors(n, generator=True))//n if n else 1 # Chai Wah Wu, Jul 15 2024

if __name__ == "__main__":
	timeStart = time.perf_counter()
	for n in range(51):
		print(n, A086345(n))
	print(time.perf_counter() - timeStart)
