"""
High-performance Numba-accelerated implementation of Lunnon's algorithm.

This implementation focuses on maximum computational performance by leveraging Numba's
just-in-time (JIT) compilation capabilities to generate native machine code. It represents
a manually optimized version that served as inspiration for the automated transformation
framework in the someAssemblyRequired package.

Key characteristics:
- Optimized data structures using NumPy typed arrays with appropriate data types
- Function decorators for Numba JIT compilation with performance-oriented settings
- Memory-efficient implementation with careful type management
- Reduced Python overhead through native code execution
- Algorithmic optimizations tailored for numerical computation

Performance considerations:
- Up to 1000× faster than pure Python implementations
- Optimized for larger map dimensions where computational demands increase exponentially
- Incorporates lessons from multiple implementation strategies

Note: This serves as a reference for manually-optimized code before the development of
the automated transformation assembly-line in the main package.
"""

from typing import Any
import numba
import numpy

@numba.jit(cache=True, nopython=True, fastmath=True)
def countFolds(listDimensions: list[int]) -> int:
	"""
	Count the number of distinct ways to fold a map with at least two positive dimensions.

	Parameters:
		listDimensions: A list of integers representing the dimensions of the map. Error checking and DRY code are impermissible in the numba and jax universes. Validate the list yourself before passing here. There might be some tools for that in this package unless I have become a pyL33t coder.

	Returns:
		foldsTotal: The total number of distinct folds for the given map dimensions.
	"""
	def integerSmall(value: numpy.integer[Any] | Any) -> numpy.uint8:
		return numpy.uint8(value)

	def integerLarge(value: numpy.integer[Any] | Any) -> numpy.uint64:
		return numpy.uint64(value)

	dtypeMedium = numpy.uint8
	dtypeMaximum = numpy.uint16

	leavesTotal = integerSmall(1)
	for 个 in listDimensions:
		leavesTotal = leavesTotal * integerSmall(个)
	dimensionsTotal = integerSmall(len(listDimensions))

	"""How to build a leaf connection graph, also called a "Cartesian Product Decomposition"
	or a "Dimensional Product Mapping", with sentinels:
	Step 1: find the cumulative product of the map's dimensions"""
	cumulativeProduct = numpy.ones(dimensionsTotal + 1, dtype=dtypeMedium)
	for dimension1ndex in range(1, dimensionsTotal + 1):
		cumulativeProduct[dimension1ndex] = cumulativeProduct[dimension1ndex - 1] * listDimensions[dimension1ndex - 1]

	"""Step 2: for each dimension, create a coordinate system """
	"""coordinateSystem[dimension1ndex, leaf1ndex] holds the dimension1ndex-th coordinate of leaf leaf1ndex"""
	coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=dtypeMedium)
	for dimension1ndex in range(1, dimensionsTotal + 1):
		for leaf1ndex in range(1, leavesTotal + 1):
			coordinateSystem[dimension1ndex, leaf1ndex] = ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) % listDimensions[dimension1ndex - 1] + 1

	"""Step 3: create a huge empty connection graph"""
	connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=dtypeMedium)

	"""Step for... for... for...: fill the connection graph"""
	for dimension1ndex in range(1, dimensionsTotal + 1):
		for leaf1ndex in range(1, leavesTotal + 1):
			for leafConnectee in range(1, leaf1ndex + 1):
				connectionGraph[dimension1ndex, leaf1ndex, leafConnectee] = (0 if leafConnectee == 0
								else ((leafConnectee if coordinateSystem[dimension1ndex, leafConnectee] == 1
											else leafConnectee - cumulativeProduct[dimension1ndex - 1])
									if (coordinateSystem[dimension1ndex, leaf1ndex] & 1) == (coordinateSystem[dimension1ndex, leafConnectee] & 1)
									else (leafConnectee if coordinateSystem[dimension1ndex, leafConnectee] == listDimensions[dimension1ndex-1]
											or leafConnectee + cumulativeProduct[dimension1ndex - 1] > leaf1ndex
											else leafConnectee + cumulativeProduct[dimension1ndex - 1])))

	"""Indices of array `track` (to "track" the execution state), which is a collection of one-dimensional arrays each of length `leavesTotal + 1`."""
	leafAbove = numba.literally(0)
	leafBelow = numba.literally(1)
	countDimensionsGapped = numba.literally(2)
	gapRangeStart = numba.literally(3)
	track = numpy.zeros((4, leavesTotal + 1), dtype=dtypeMedium)

	gapsWhere = numpy.zeros(integerLarge(integerLarge(leavesTotal) * integerLarge(leavesTotal) + 1), dtype=dtypeMaximum)

	foldsTotal = integerLarge(0)
	leaf1ndex = integerSmall(1)
	gap1ndex = integerSmall(0)

	while leaf1ndex > 0:
		if leaf1ndex <= 1 or track[leafBelow, 0] == 1:
			if leaf1ndex > leavesTotal:
				foldsTotal += leavesTotal
			else:
				dimensionsUnconstrained = integerSmall(0)
				"""Track possible gaps for leaf1ndex in each section"""
				gap1ndexCeiling = track[gapRangeStart, leaf1ndex - 1]

				"""Count possible gaps for leaf1ndex in each section"""
				dimension1ndex = integerSmall(1)
				while dimension1ndex <= dimensionsTotal:
					if connectionGraph[dimension1ndex, leaf1ndex, leaf1ndex] == leaf1ndex:
						dimensionsUnconstrained += 1
					else:
						leafConnectee = connectionGraph[dimension1ndex, leaf1ndex, leaf1ndex]
						while leafConnectee != leaf1ndex:
							gapsWhere[gap1ndexCeiling] = leafConnectee
							if track[countDimensionsGapped, leafConnectee] == 0:
								gap1ndexCeiling += 1
							track[countDimensionsGapped, leafConnectee] += 1
							leafConnectee = connectionGraph[dimension1ndex, leaf1ndex, track[leafBelow, leafConnectee]]
					dimension1ndex += 1

				"""If leaf1ndex is unconstrained in all sections, it can be inserted anywhere"""
				if dimensionsUnconstrained == dimensionsTotal:
					indexLeaf = integerSmall(0)
					while indexLeaf < leaf1ndex:
						gapsWhere[gap1ndexCeiling] = indexLeaf
						gap1ndexCeiling += 1
						indexLeaf += 1

				"""Filter gaps that are common to all sections"""
				indexMiniGap = gap1ndex
				while indexMiniGap < gap1ndexCeiling:
					gapsWhere[gap1ndex] = gapsWhere[indexMiniGap]
					if track[countDimensionsGapped, gapsWhere[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
						gap1ndex += 1
					"""Reset track[countDimensionsGapped] for next iteration"""
					track[countDimensionsGapped, gapsWhere[indexMiniGap]] = 0
					indexMiniGap += 1

		"""Recursive backtracking steps"""
		while leaf1ndex > 0 and gap1ndex == track[gapRangeStart, leaf1ndex - 1]:
			leaf1ndex -= 1
			track[leafBelow, track[leafAbove, leaf1ndex]] = track[leafBelow, leaf1ndex]
			track[leafAbove, track[leafBelow, leaf1ndex]] = track[leafAbove, leaf1ndex]

		"""Place leaf in valid position"""
		if leaf1ndex > 0:
			gap1ndex -= 1
			track[leafAbove, leaf1ndex] = gapsWhere[gap1ndex]
			track[leafBelow, leaf1ndex] = track[leafBelow, track[leafAbove, leaf1ndex]]
			track[leafBelow, track[leafAbove, leaf1ndex]] = leaf1ndex
			track[leafAbove, track[leafBelow, leaf1ndex]] = leaf1ndex
			"""Save current gap index"""
			track[gapRangeStart, leaf1ndex] = gap1ndex
			"""Move to next leaf"""
			leaf1ndex += 1

	return int(foldsTotal)
