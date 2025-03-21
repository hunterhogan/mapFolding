"""
Alternative algorithm entry point implementation for Lunnon's map folding algorithm.

This implementation demonstrates an interesting algorithmic variation where the main processing
loop is "rotated" to enter at a different point in the execution flow. Specifically, it's
structured to enter at the modulo operator rather than the traditional starting point.

Key characteristics:
- Restructures the control flow by reorganizing the entry point of the algorithm
- Separates preparation work from the main computational loop
- Uses explicit variable naming with index constants for clarity
- Demonstrates how the same algorithm can be approached from different entry points

Note: This implementation is intentionally incomplete and requires supporting code from
other modules to function. It serves primarily as a demonstration of how algorithmic
structure can be creatively redesigned while maintaining the core computational approach.
"""

from mapFolding import outfitFoldings
from numba import njit
import numpy
from numpy.typing import NDArray

"""
It is possible to enter the main `while` loop from an arbitrary point. This version is "rotated" to effectively enter at the modulo operator.
"""

# Indices of array `track`, which is a collection of one-dimensional arrays each of length `the[leavesTotal] + 1`.
# The values in the array cells are dynamic, small, unsigned integers.
A = leafAbove = 0
"""Leaf above leaf m"""
B = leafBelow = 1
"""Leaf below leaf m"""
count = countDimensionsGapped = 2
"""Number of gaps available for leaf l"""
gapter = gapRangeStart = 3
"""Index of gap stack for leaf l"""

# Indices of array `my`, which holds dynamic, small, unsigned, integer values.
tricky = [
(leaf1ndex := 0),
(gap1ndex := 1),
(unconstrainedLeaf := 2),
(gap1ndexCeiling := 3),
(leafConnectee := 4),
(taskIndex := 5),
(dimension1ndex := 6),
(foldingsSubtotal := 7),
]

COUNTindicesDynamic = len(tricky)

# Indices of array `the`, which holds unchanging, small, unsigned, integer values.
tricky = [
(dimensionsPlus1 := 0),
(dimensionsTotal := 1),
(leavesTotal := 2),
]

COUNTindicesStatic = len(tricky)

def countFolds(listDimensions: list[int]):
	static = numpy.zeros(COUNTindicesStatic, dtype=numpy.int64)

	listDimensions, static[leavesTotal], D, track,gapsWhere = outfitFoldings(listDimensions)

	static[dimensionsTotal] = len(listDimensions)
	static[dimensionsPlus1] = static[dimensionsTotal] + 1

	# Pass listDimensions and taskDivisions to _sherpa for benchmarking
	foldingsTotal = _sherpa(track, gapsWhere, static, D, listDimensions)
	return foldingsTotal

# @recordBenchmarks()
def _sherpa(track: NDArray, gap: NDArray, static: NDArray, D: NDArray, p: list[int]):
	"""Performance critical section that counts foldings.

	Parameters:
		track: Array tracking folding state
		gap: Array for potential gaps
		static: Array containing static configuration values
		D: Array of leaf connections
		p: List of dimensions for benchmarking
	"""
	foldingsTotal = countFoldings(track, gap, static, D)
	return foldingsTotal

@njit(cache=True, parallel=False, fastmath=False)
def countFoldings(TEMPLATEtrack: NDArray,
					TEMPLATEgapsWhere: NDArray,
					the: NDArray,
					connectionGraph: NDArray
					):

	TEMPLATEmy = numpy.zeros(COUNTindicesDynamic, dtype=numpy.int64)
	TEMPLATEmy[leaf1ndex] = 1

	taskDivisions = 0
	# taskDivisions = the[leavesTotal]
	TEMPLATEmy[taskIndex] = taskDivisions - 1 # the first modulo is leavesTotal - 1

	def prepareWork(track: NDArray,
					gapsWhere: NDArray,
					my: NDArray) -> tuple[NDArray, NDArray, NDArray]:
		foldingsTotal = 0
		while True:
			if my[leaf1ndex] <= 1 or track[leafBelow][0] == 1:
				if my[leaf1ndex] > the[leavesTotal]:
					foldingsTotal += the[leavesTotal]
				else:
					my[unconstrainedLeaf] = 0
					my[gap1ndexCeiling] = track[gapRangeStart][my[leaf1ndex] - 1]
					my[gap1ndex] = my[gap1ndexCeiling]

					for PREPAREdimension1ndex in range(1, the[dimensionsPlus1]):
						if connectionGraph[PREPAREdimension1ndex][my[leaf1ndex]][my[leaf1ndex]] == my[leaf1ndex]:
							my[unconstrainedLeaf] += 1
						else:
							my[leafConnectee] = connectionGraph[PREPAREdimension1ndex][my[leaf1ndex]][my[leaf1ndex]]
							while my[leafConnectee] != my[leaf1ndex]:

								if my[leafConnectee] != my[leaf1ndex]:
									my[dimension1ndex] = PREPAREdimension1ndex
									return track, gapsWhere, my

								if my[leaf1ndex] != the[leavesTotal]:
									gapsWhere[my[gap1ndexCeiling]] = my[leafConnectee]
									if track[countDimensionsGapped][my[leafConnectee]] == 0:
										my[gap1ndexCeiling] += 1
									track[countDimensionsGapped][my[leafConnectee]] += 1
								else:
									print("else")
									my[dimension1ndex] = PREPAREdimension1ndex
									return track, gapsWhere, my
									# PREPAREmy[leafConnectee] % the[leavesTotal] == PREPAREmy[taskIndex]
								my[leafConnectee] = connectionGraph[dimension1ndex][my[leaf1ndex]][track[leafBelow][my[leafConnectee]]]

					if my[unconstrainedLeaf] == the[dimensionsTotal]:
						for indexLeaf in range(my[leaf1ndex]):
							gapsWhere[my[gap1ndexCeiling]] = indexLeaf
							my[gap1ndexCeiling] += 1

					for indexMiniGap in range(my[gap1ndex], my[gap1ndexCeiling]):
						gapsWhere[my[gap1ndex]] = gapsWhere[indexMiniGap]
						if track[countDimensionsGapped][gapsWhere[indexMiniGap]] == the[dimensionsTotal] - my[unconstrainedLeaf]:
							my[gap1ndex] += 1
						track[countDimensionsGapped][gapsWhere[indexMiniGap]] = 0

			while my[leaf1ndex] > 0 and my[gap1ndex] == track[gapRangeStart][my[leaf1ndex] - 1]:
				my[leaf1ndex] -= 1
				track[leafBelow][track[leafAbove][my[leaf1ndex]]] = track[leafBelow][my[leaf1ndex]]
				track[leafAbove][track[leafBelow][my[leaf1ndex]]] = track[leafAbove][my[leaf1ndex]]

			if my[leaf1ndex] > 0:
				my[gap1ndex] -= 1
				track[leafAbove][my[leaf1ndex]] = gapsWhere[my[gap1ndex]]
				track[leafBelow][my[leaf1ndex]] = track[leafBelow][track[leafAbove][my[leaf1ndex]]]
				track[leafBelow][track[leafAbove][my[leaf1ndex]]] = my[leaf1ndex]
				track[leafAbove][track[leafBelow][my[leaf1ndex]]] = my[leaf1ndex]
				track[gapRangeStart][my[leaf1ndex]] = my[gap1ndex]
				my[leaf1ndex] += 1

	RETURNtrack, RETURNgapsWhere, RETURNmy = prepareWork(TEMPLATEtrack.copy(), TEMPLATEgapsWhere.copy(), TEMPLATEmy.copy())

	foldingsTotal = doWork(RETURNtrack.copy(), RETURNgapsWhere.copy(), RETURNmy.copy(), the, connectionGraph, taskDivisions)

	return foldingsTotal

@njit(cache=True, parallel=False, fastmath=False)
def doWork(track: NDArray,
				gapsWhere: NDArray,
				my: NDArray,
				the: NDArray,
				connectionGraph: NDArray,
				taskDivisions: int = 0
				):

	papasGotABrandNewBag = True
	if_activeLeaf1ndex_LTE_1_or_leafBelow_index_0_equals_1 = True
	for_dimension1ndex_in_range_1_to_dimensionsPlus1 = True
	while_leaf1ndexConnectee_notEquals_activeLeaf1ndex = True

	thisIsNotTheFirstPass = False

	while papasGotABrandNewBag:
		if my[leaf1ndex] <= 1 or track[leafBelow][0] == 1 or if_activeLeaf1ndex_LTE_1_or_leafBelow_index_0_equals_1 == True:
			if_activeLeaf1ndex_LTE_1_or_leafBelow_index_0_equals_1 = False
			if my[leaf1ndex] > the[leavesTotal] and thisIsNotTheFirstPass:
				my[foldingsSubtotal] += the[leavesTotal]
			else:
				if thisIsNotTheFirstPass:
					my[unconstrainedLeaf] = 0
					my[gap1ndexCeiling] = track[gapRangeStart][my[leaf1ndex] - 1]
					my[gap1ndex] = my[gap1ndexCeiling]

				for_dimension1ndex_in_range_1_to_dimensionsPlus1 = True
				while for_dimension1ndex_in_range_1_to_dimensionsPlus1 == True:
					for_dimension1ndex_in_range_1_to_dimensionsPlus1 = False
					if connectionGraph[my[dimension1ndex]][my[leaf1ndex]][my[leaf1ndex]] == my[leaf1ndex] and thisIsNotTheFirstPass:
						my[unconstrainedLeaf] += 1
					else:
						if thisIsNotTheFirstPass:
							my[leafConnectee] = connectionGraph[my[dimension1ndex]][my[leaf1ndex]][my[leaf1ndex]]
						if my[leafConnectee] != my[leaf1ndex]:
							while_leaf1ndexConnectee_notEquals_activeLeaf1ndex = True

						while while_leaf1ndexConnectee_notEquals_activeLeaf1ndex == True:
							while_leaf1ndexConnectee_notEquals_activeLeaf1ndex = False
							thisIsNotTheFirstPass = True
							if taskDivisions==0 or my[leaf1ndex] != taskDivisions:
								myTask = True
							else:
								modulo = my[leafConnectee] % the[leavesTotal]
								if modulo == my[taskIndex]: myTask = True
								else:
									myTask = False
							if myTask:
								gapsWhere[my[gap1ndexCeiling]] = my[leafConnectee]
								if track[countDimensionsGapped][my[leafConnectee]] == 0:
									my[gap1ndexCeiling] += 1
								track[countDimensionsGapped][my[leafConnectee]] += 1
							my[leafConnectee] = connectionGraph[my[dimension1ndex]][my[leaf1ndex]][track[leafBelow][my[leafConnectee]]]
							if my[leafConnectee] != my[leaf1ndex]:
								while_leaf1ndexConnectee_notEquals_activeLeaf1ndex = True
					my[dimension1ndex] += 1
					if my[dimension1ndex] < the[dimensionsPlus1]:
						for_dimension1ndex_in_range_1_to_dimensionsPlus1 = True
					else:
						my[dimension1ndex] = 1

				if my[unconstrainedLeaf] == the[dimensionsTotal]:
					for leaf1ndex in range(my[leaf1ndex]):
						gapsWhere[my[gap1ndexCeiling]] = leaf1ndex
						my[gap1ndexCeiling] += 1

				for indexMiniGap in range(my[gap1ndex], my[gap1ndexCeiling]):
					gapsWhere[my[gap1ndex]] = gapsWhere[indexMiniGap]
					if track[countDimensionsGapped][gapsWhere[indexMiniGap]] == the[dimensionsTotal] - my[unconstrainedLeaf]:
						my[gap1ndex] += 1
					track[countDimensionsGapped][gapsWhere[indexMiniGap]] = 0

		while my[leaf1ndex] > 0 and my[gap1ndex] == track[gapRangeStart][my[leaf1ndex] - 1]:
			my[leaf1ndex] -= 1
			track[leafBelow][track[leafAbove][my[leaf1ndex]]] = track[leafBelow][my[leaf1ndex]]
			track[leafAbove][track[leafBelow][my[leaf1ndex]]] = track[leafAbove][my[leaf1ndex]]

		if my[leaf1ndex] > 0:
			my[gap1ndex] -= 1
			track[leafAbove][my[leaf1ndex]] = gapsWhere[my[gap1ndex]]
			track[leafBelow][my[leaf1ndex]] = track[leafBelow][track[leafAbove][my[leaf1ndex]]]
			track[leafBelow][track[leafAbove][my[leaf1ndex]]] = my[leaf1ndex]
			track[leafAbove][track[leafBelow][my[leaf1ndex]]] = my[leaf1ndex]
			track[gapRangeStart][my[leaf1ndex]] = my[gap1ndex]
			my[leaf1ndex] += 1

		if my[leaf1ndex] <= 0:
			return my[foldingsSubtotal]
