"""
A largely faithful translation of the original Atlas Autocode code by W. F. Lunnon to Python using `while`.

This implementation closely follows the structure and logic of Lunnon's 1971 paper, preserving the
variable names and core algorithm design. It uses while loops instead of Atlas Autocode's procedural
control structures, maintaining the imperative programming style of the original.

Key characteristics:
- Preserves original algorithm structure for historical accuracy
- Uses primarily scalar operations and explicit loops
- Maintains the original variable naming from Lunnon's work
- Provides a baseline for comparison against optimized implementations

Reference:
W. F. Lunnon, Multi-dimensional map-folding, The Computer Journal, Volume 14, Issue 1, 1971,
Pages 75-80, https://doi.org/10.1093/comjnl/14.1.75
"""

def foldings(p: list[int]) -> int:
	"""
	Run loop with (A, B) on each folding of a p[1] x ... x p[d] map, where A and B are the above and below vectors.

	Parameters:
		p: An array of integers representing the dimensions of the map.

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

	A = [0] * (n + 1)
	B = [0] * (n + 1)
	count = [0] * (n + 1)
	gapter = [0] * (n + 1)
	gap = [0] * (n * n + 1)

	# B[m] is the leaf below leaf m in the current folding,
	# A[m] the leaf above. count[m] is the no. of sections in which
	# there is a gap for the new leaf l below leaf m,
	# gap[gapter[l - 1] + j] is the j-th (possible or actual) gap for leaf l,
	# and later gap[gapter[l]] is the gap where leaf l is currently inserted

	P = [1] * (d + 1)
	C = [[0] * (n + 1) for _dimension1 in range(d + 1)]
	D = [[[0] * (n + 1) for _dimension2 in range(n + 1)] for _dimension1 in range(d + 1)]

	for i in range(1, d + 1):
		P[i] = P[i - 1] * p[i - 1]

	for i in range(1, d + 1):
		for m in range(1, n + 1):
			C[i][m] = ((m - 1) // P[i - 1]) - ((m - 1) // P[i]) * p[i - 1] + 1

	for i in range(1, d + 1):
		for l in range(1, n + 1):
			for m in range(1, l + 1):
				if C[i][l] - C[i][m] == (C[i][l] - C[i][m]) // 2 * 2: # !
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
		if l > n:
			G = G + 1
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
