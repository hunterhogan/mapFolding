"""
A generally faithful translation of the original Atlas Autocode code by W. F. Lunnon to Python using NumPy.

This implementation transforms Lunnon's 1971 algorithm to leverage NumPy's array operations for improved
performance while maintaining algorithmic fidelity. It preserves the core logic and variable naming
conventions of the original algorithm but benefits from NumPy's vectorized operations and efficient
memory management.

Key characteristics:
- Uses NumPy arrays instead of Python lists for better memory efficiency
- Maintains the original algorithm structure and control flow
- Preserves variable naming for algorithmic clarity
- Offers significant performance improvements over pure Python implementations

Reference:
W. F. Lunnon, Multi-dimensional map-folding, The Computer Journal, Volume 14, Issue 1, 1971,
Pages 75-80, https://doi.org/10.1093/comjnl/14.1.75
"""

import numpy

def foldings(p: list[int]) -> int:
	"""
	Run loop with (A, B) on each folding of a p[1] x ... x p[d] map, where A and B are the above and below vectors.

	Parameters:
		p: A list of integers representing the dimensions of the map.

	Returns:
		G: The number of distinct foldings for the given map dimensions.

	NOTE If there are fewer than two dimensions, any dimensions are not positive, or any dimensions are not integers, the output will be unreliable.
	"""

	g: int = 0
	d: int = len(p)
	n: int = 1
	for i in range(d):
		n = n * p[i]

	# d dimensions and n leaves

	A = numpy.zeros(n + 1, dtype=int)
	B = numpy.zeros(n + 1, dtype=int)
	count = numpy.zeros(n + 1, dtype=int)
	gapter = numpy.zeros(n + 1, dtype=int)
	gap = numpy.zeros(n * n + 1, dtype=int)

	# B[m] is the leaf below leaf m in the current folding,
	# A[m] the leaf above. count[m] is the no. of sections in which
	# there is a gap for the new leaf l below leaf m,
	# gap[gapter[l - 1] + j] is the j-th (possible or actual) gap for leaf l,
	# and later gap[gapter[l]] is the gap where leaf l is currently inserted

	P = numpy.ones(d + 1, dtype=int)
	C = numpy.zeros((d + 1, n + 1), dtype=int)
	D = numpy.zeros((d + 1, n + 1, n + 1), dtype=int)

	for i in range(1, d + 1):
		P[i] = P[i - 1] * p[i - 1]

	for i in range(1, d + 1):
		for m in range(1, n + 1):
			C[i][m] = ((m - 1) // P[i - 1]) % p[i - 1] + 1 # NOTE Because modulo is available, this statement is simpler.

	for i in range(1, d + 1):
		for l in range(1, n + 1):
			for m in range(1, l + 1):
				if C[i][l] - C[i][m] == (C[i][l] - C[i][m]) // 2 * 2:
					if C[i][m] == 1:
						D[i][l][m] = m
					else:
						D[i][l][m] = m - P[i - 1]
				else:
					if C[i][m] == p[i - 1] or m + P[i - 1] > l:
						D[i][l][m] = m
					else:
						D[i][l][m] = m + P[i - 1]
	# P[i] = p[1] x ... x p[i], C[i][m] = i-th co-ordinate of leaf m,
	# D[i][l][m] = leaf connected to m in section i when inserting l;

	G: int = 0
	l = 1

	# kick off with null folding
	while l > 0:
		if l <= 1 or B[0] == 1: # NOTE This statement is part of a significant divergence from the 1971 paper. As a result, this version is greater than one order of magnitude faster.
			if l > n:
				G = G + n # NOTE Due to `B[0] == 1`, this implementation increments the counted foldings in batches of `n`-many foldings, rather than immediately incrementing when a folding is found, i.e. `G = G + 1`
			else:
				dd: int = 0
				gg: int = gapter[l - 1]
				g = gg
				# dd is the no. of sections in which l is unconstrained,
				# gg the no. of possible and g the no. of actual gaps for l, + gapter[l - 1]

				# find the possible gaps for leaf l in each section,
				# then discard those not common to all. All possible if dd = d
				for i in range(1, d + 1):
					if D[i][l][l] == l:
						dd = dd + 1
					else:
						m = D[i][l][l]
						while m != l:
							gap[gg] = m
							if count[m] == 0:
								gg = gg + 1
							count[m] += 1
							m = D[i][l][B[m]]

				if dd == d:
					for m in range(l):
						gap[gg] = m
						gg = gg + 1

				for j in range(g, gg):
					gap[g] = gap[j]
					if count[gap[j]] == d - dd:
						g = g + 1
					count[gap[j]] = 0

		# for each gap insert leaf l, [the main while loop shall progress],
		# remove leaf l
		while l > 0 and g == gapter[l - 1]:
			l = l - 1
			B[A[l]] = B[l]
			A[B[l]] = A[l]

		if l > 0:
			g = g - 1
			A[l] = gap[g]
			B[l] = B[A[l]]
			B[A[l]] = l
			A[B[l]] = l
			gapter[l] = g
			l = l + 1
	return G
