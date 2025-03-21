"""
Ported from the Java version by Sean A. Irvine:
https://github.com/archmageirvine/joeis/blob/80e3e844b11f149704acbab520bc3a3a25ac34ff/src/irvine/oeis/a001/A001415.java

This implementation is a conversion from a well-known Java implementation of Lunnon's algorithm
by Sean A. Irvine, a contributor to the OEIS project. It provides a clean, procedural implementation
with straightforward variable naming and control flow that may be more approachable for
programmers familiar with modern languages.

Key characteristics:
- Clear variable naming following modern programming conventions
- Procedural implementation style similar to Java but adapted for Python
- Follows the same algorithmic structure as Lunnon's original but with cleaner organization
- Uses primitive Python data structures (lists) without NumPy dependencies

Citation: https://github.com/hunterhogan/mapFolding/blob/134f2e6ecdf59fb6f6829c775475544a6aaaa800/citations/jOEIS.bibtex
"""

def foldings(p: list[int], res: int = 0, mod: int = 0) -> int:
	"""
	Compute the total number of foldings for a map with dimensions specified in p.

	Parameters:
		p: List of integers representing the dimensions of the map.
		res: Residue for modulo operation (integer).
		mod: Modulus for modulo operation (integer).

	Returns:
		total_count: The total number of foldings (integer).
	"""
	n = 1  # Total number of leaves
	d = len(p)  # Number of dimensions
	for dimension in p:
		n *= dimension

	# Initialize arrays/lists
	A = [0] * (n + 1)	   # Leaf above leaf m
	B = [0] * (n + 1)	   # Leaf below leaf m
	count = [0] * (n + 1)   # Counts for potential gaps
	gapter = [0] * (n + 1)  # Indices for gap stack per leaf
	gap = [0] * (n * n + 1) # Stack of potential gaps

	# Compute arrays P, C, D as per the algorithm
	P = [1] * (d + 1)
	for i in range(1, d + 1):
		P[i] = P[i - 1] * p[i - 1]

	# C[i][m] holds the i-th coordinate of leaf m
	C = [[0] * (n + 1) for _ in range(d + 1)]
	for i in range(1, d + 1):
		for m in range(1, n + 1):
			C[i][m] = ((m - 1) // P[i - 1]) - ((m - 1) // P[i]) * p[i - 1] + 1

	# D[i][l][m] computes the leaf connected to m in section i when inserting l
	D = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(d + 1)]
	for i in range(1, d + 1):
		for l in range(1, n + 1):
			for m in range(1, l + 1):
				delta = C[i][l] - C[i][m]
				if delta % 2 == 0:
					# If delta is even
					if C[i][m] == 1:
						D[i][l][m] = m
					else:
						D[i][l][m] = m - P[i - 1]
				else:
					# If delta is odd
					if C[i][m] == p[i - 1] or m + P[i - 1] > l:
						D[i][l][m] = m
					else:
						D[i][l][m] = m + P[i - 1]

	# Initialize variables for backtracking
	total_count = 0  # Total number of foldings
	g = 0			# Gap index
	l = 1			# Current leaf

	# Start backtracking loop
	while l > 0:
		# If we have processed all leaves, increment total count
		if l > n:
			total_count += 1
		else:
			dd = 0	 # Number of sections where leaf l is unconstrained
			gg = g	 # Temporary gap index
			g = gapter[l - 1]  # Reset gap index for current leaf

			# Count possible gaps for leaf l in each section
			for i in range(1, d + 1):
				if D[i][l][l] == l:
					dd += 1
				else:
					m = D[i][l][l]
					while m != l:
						if mod == 0 or l != mod or m % mod == res:
							gap[gg] = m
							if count[m] == 0:
								gg += 1
							count[m] += 1
						m = D[i][l][B[m]]

			# If leaf l is unconstrained in all sections, it can be inserted anywhere
			if dd == d:
				for m in range(l):
					gap[gg] = m
					gg += 1

			# Filter gaps that are common to all sections
			for j in range(g, gg):
				gap[g] = gap[j]
				if count[gap[j]] == d - dd:
					g += 1
				count[gap[j]] = 0  # Reset count for next iteration

		# Recursive backtracking steps
		while l > 0 and g == gapter[l - 1]:
			# No more gaps to try, backtrack to previous leaf
			l -= 1
			B[A[l]] = B[l]
			A[B[l]] = A[l]

		if l > 0:
			# Try next gap for leaf l
			g -= 1
			A[l] = gap[g]
			B[l] = B[A[l]]
			B[A[l]] = l
			A[B[l]] = l
			gapter[l] = g  # Save current gap index
			l += 1		 # Move to next leaf

	return total_count
