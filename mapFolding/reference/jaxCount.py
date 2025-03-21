"""
I was able to implement the algorithm with JAX, but I didn't see an advantage and it's a pain in the ass.

Experimental JAX implementation of Lunnon's algorithm for potential GPU acceleration.

This implementation explores the use of JAX (Just After eXecution) for potential GPU acceleration
of the map folding algorithm. It represents an experimental approach that attempts to leverage
JAX's transformation capabilities and hardware acceleration through XLA compilation.

Key characteristics:
- Uses JAX's functional programming model with lax control flow primitives
- Attempts to leverage GPU acceleration for numerical operations
- Demonstrates JAX-specific patterns for handling loops and conditions
- Supports jit compilation for performance optimization

As noted in the file's comment, this implementation didn't demonstrate significant advantages
over other approaches and presented additional complexity. It serves as a valuable reference
for exploring alternative acceleration strategies and understanding the limitations of
different computational frameworks for this specific algorithm.
"""
from flattened import validateListDimensions, getLeavesTotal, makeConnectionGraph
from numpy.typing import NDArray
from typing import Any
import jax
import jaxtyping
import numpy

dtypeMedium = jax.numpy.uint32
dtypeMaximum = jax.numpy.uint32

def countFolds(listDimensions: list[int]) -> int:
	listDimensionsPositive: list[int] = validateListDimensions(listDimensions)

	n: int = getLeavesTotal(listDimensionsPositive)
	d: int = len(listDimensions)
	D: NDArray[numpy.integer[Any]] = makeConnectionGraph(listDimensionsPositive)
	connectionGraph = jax.numpy.asarray(D, dtype=dtypeMedium)
	del listDimensionsPositive

	return foldingsJAX(n, d, connectionGraph)

def foldingsJAX(leavesTotal: jaxtyping.UInt32, dimensionsTotal: jaxtyping.UInt32, connectionGraph: jaxtyping.Array) -> jaxtyping.UInt32:

	def doNothing(argument: Any):
		return argument

	def while_activeLeaf1ndex_greaterThan_0(comparisonValues: tuple):
		comparand = comparisonValues[6]
		return comparand > 0

	def countFoldings(allValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
		_0, leafBelow, _2, _3, _4, _5, activeLeaf1ndex, _7 = allValues

		sentinel = leafBelow.at[0].get().astype(jax.numpy.uint32)

		allValues = jax.lax.cond(findGapsCondition(sentinel, activeLeaf1ndex),
							lambda argumentX: dao(findGapsDo(argumentX)),
							lambda argumentY: jax.lax.cond(incrementCondition(sentinel, activeLeaf1ndex), lambda argumentZ: dao(incrementDo(argumentZ)), dao, argumentY),
							allValues)

		return allValues

	def findGapsCondition(leafBelowSentinel, activeLeafNumber):
		return jax.numpy.logical_or(jax.numpy.logical_and(leafBelowSentinel == 1, activeLeafNumber <= leavesTotal), activeLeafNumber <= 1)

	def findGapsDo(allValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
		def for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1(comparisonValues: tuple):
			return comparisonValues[-1] <= dimensionsTotal

		def for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
			def ifLeafIsUnconstrainedCondition(comparand):
				return jax.numpy.equal(connectionGraph[comparand, activeLeaf1ndex, activeLeaf1ndex], activeLeaf1ndex)

			def ifLeafIsUnconstrainedDo(unconstrainedValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
				unconstrained_unconstrainedLeaf = unconstrainedValues[3]
				unconstrained_unconstrainedLeaf = 1 + unconstrained_unconstrainedLeaf
				return (unconstrainedValues[0], unconstrainedValues[1], unconstrainedValues[2], unconstrained_unconstrainedLeaf)

			def ifLeafIsUnconstrainedElse(unconstrainedValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
				def while_leaf1ndexConnectee_notEquals_activeLeaf1ndex(comparisonValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
					return comparisonValues[-1] != activeLeaf1ndex

				def countGaps(countGapsDoValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
					countGapsCountDimensionsGapped, countGapsPotentialGaps, countGapsGap1ndexLowerBound, countGapsLeaf1ndexConnectee = countGapsDoValues

					countGapsPotentialGaps = countGapsPotentialGaps.at[countGapsGap1ndexLowerBound].set(countGapsLeaf1ndexConnectee)
					countGapsGap1ndexLowerBound = jax.numpy.where(jax.numpy.equal(countGapsCountDimensionsGapped[countGapsLeaf1ndexConnectee], 0), countGapsGap1ndexLowerBound + 1, countGapsGap1ndexLowerBound)
					countGapsCountDimensionsGapped = countGapsCountDimensionsGapped.at[countGapsLeaf1ndexConnectee].add(1)
					countGapsLeaf1ndexConnectee = connectionGraph.at[dimensionNumber, activeLeaf1ndex, leafBelow.at[countGapsLeaf1ndexConnectee].get()].get().astype(jax.numpy.uint32)

					return (countGapsCountDimensionsGapped, countGapsPotentialGaps, countGapsGap1ndexLowerBound, countGapsLeaf1ndexConnectee)

				unconstrained_countDimensionsGapped, unconstrained_gapsWhere, unconstrained_gap1ndexCeiling, unconstrained_unconstrainedLeaf = unconstrainedValues

				leaf1ndexConnectee = connectionGraph.at[dimensionNumber, activeLeaf1ndex, activeLeaf1ndex].get().astype(jax.numpy.uint32)

				countGapsValues = (unconstrained_countDimensionsGapped, unconstrained_gapsWhere, unconstrained_gap1ndexCeiling, leaf1ndexConnectee)
				countGapsValues = jax.lax.while_loop(while_leaf1ndexConnectee_notEquals_activeLeaf1ndex, countGaps, countGapsValues)
				unconstrained_countDimensionsGapped, unconstrained_gapsWhere, unconstrained_gap1ndexCeiling, leaf1ndexConnectee = countGapsValues

				return (unconstrained_countDimensionsGapped, unconstrained_gapsWhere, unconstrained_gap1ndexCeiling, unconstrained_unconstrainedLeaf)

			dimensions_countDimensionsGapped, dimensions_gapsWhere, dimensions_gap1ndexCeiling, dimensions_unconstrainedLeaf, dimensionNumber = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values

			ifLeafIsUnconstrainedValues = (dimensions_countDimensionsGapped, dimensions_gapsWhere, dimensions_gap1ndexCeiling, dimensions_unconstrainedLeaf)
			ifLeafIsUnconstrainedValues = jax.lax.cond(ifLeafIsUnconstrainedCondition(dimensionNumber), ifLeafIsUnconstrainedDo, ifLeafIsUnconstrainedElse, ifLeafIsUnconstrainedValues)
			dimensions_countDimensionsGapped, dimensions_gapsWhere, dimensions_gap1ndexCeiling, dimensions_unconstrainedLeaf = ifLeafIsUnconstrainedValues

			dimensionNumber = 1 + dimensionNumber
			return (dimensions_countDimensionsGapped, dimensions_gapsWhere, dimensions_gap1ndexCeiling, dimensions_unconstrainedLeaf, dimensionNumber)

		def almostUselessCondition(comparand):
			return comparand == dimensionsTotal

		def almostUselessConditionDo(for_leaf1ndex_in_range_activeLeaf1ndexValues: tuple[jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
			def for_leaf1ndex_in_range_activeLeaf1ndex(comparisonValues):
				return comparisonValues[-1] < activeLeaf1ndex

			def for_leaf1ndex_in_range_activeLeaf1ndex_do(for_leaf1ndex_in_range_activeLeaf1ndexValues: tuple[jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
				leafInRangePotentialGaps, gapNumberLowerBound, leafNumber = for_leaf1ndex_in_range_activeLeaf1ndexValues
				leafInRangePotentialGaps = leafInRangePotentialGaps.at[gapNumberLowerBound].set(leafNumber)
				gapNumberLowerBound = 1 + gapNumberLowerBound
				leafNumber = 1 + leafNumber
				return (leafInRangePotentialGaps, gapNumberLowerBound, leafNumber)
			return jax.lax.while_loop(for_leaf1ndex_in_range_activeLeaf1ndex, for_leaf1ndex_in_range_activeLeaf1ndex_do, for_leaf1ndex_in_range_activeLeaf1ndexValues)

		def for_range_from_activeGap1ndex_to_gap1ndexCeiling(comparisonValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
			return comparisonValues[-1] < gap1ndexCeiling

		def miniGapDo(gapToGapValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
			gapToGapCountDimensionsGapped, gapToGapPotentialGaps, activeGapNumber, index = gapToGapValues
			gapToGapPotentialGaps = gapToGapPotentialGaps.at[activeGapNumber].set(gapToGapPotentialGaps.at[index].get())
			activeGapNumber = jax.numpy.where(jax.numpy.equal(gapToGapCountDimensionsGapped.at[gapToGapPotentialGaps.at[index].get()].get(), dimensionsTotal - unconstrainedLeaf), activeGapNumber + 1, activeGapNumber).astype(jax.numpy.uint32)
			gapToGapCountDimensionsGapped = gapToGapCountDimensionsGapped.at[gapToGapPotentialGaps.at[index].get()].set(0)
			index = 1 + index
			return (gapToGapCountDimensionsGapped, gapToGapPotentialGaps, activeGapNumber, index)

		_0, leafBelow, countDimensionsGapped, gapRangeStart, gapsWhere, _5, activeLeaf1ndex, activeGap1ndex = allValues

		unconstrainedLeaf = jax.numpy.uint32(0)
		dimension1ndex = jax.numpy.uint32(1)
		gap1ndexCeiling = gapRangeStart.at[activeLeaf1ndex - 1].get().astype(jax.numpy.uint32)
		activeGap1ndex = gap1ndexCeiling
		for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = (countDimensionsGapped, gapsWhere, gap1ndexCeiling, unconstrainedLeaf, dimension1ndex)
		for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = jax.lax.while_loop(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values)
		countDimensionsGapped, gapsWhere, gap1ndexCeiling, unconstrainedLeaf, dimension1ndex = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values
		del dimension1ndex

		leaf1ndex = jax.numpy.uint32(0)
		for_leaf1ndex_in_range_activeLeaf1ndexValues = (gapsWhere, gap1ndexCeiling, leaf1ndex)
		for_leaf1ndex_in_range_activeLeaf1ndexValues = jax.lax.cond(almostUselessCondition(unconstrainedLeaf), almostUselessConditionDo, doNothing, for_leaf1ndex_in_range_activeLeaf1ndexValues)
		gapsWhere, gap1ndexCeiling, leaf1ndex = for_leaf1ndex_in_range_activeLeaf1ndexValues
		del leaf1ndex

		indexMiniGap = activeGap1ndex
		miniGapValues = (countDimensionsGapped, gapsWhere, activeGap1ndex, indexMiniGap)
		miniGapValues = jax.lax.while_loop(for_range_from_activeGap1ndex_to_gap1ndexCeiling, miniGapDo, miniGapValues)
		countDimensionsGapped, gapsWhere, activeGap1ndex, indexMiniGap = miniGapValues
		del indexMiniGap

		return (allValues[0], leafBelow, countDimensionsGapped, gapRangeStart, gapsWhere, allValues[5], activeLeaf1ndex, activeGap1ndex)

	def incrementCondition(leafBelowSentinel, activeLeafNumber):
		return jax.numpy.logical_and(activeLeafNumber > leavesTotal, leafBelowSentinel == 1)

	def incrementDo(allValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
		foldingsSubTotal = allValues[5]
		foldingsSubTotal = leavesTotal + foldingsSubTotal
		return (allValues[0], allValues[1], allValues[2], allValues[3], allValues[4], foldingsSubTotal, allValues[6], allValues[7])

	def dao(allValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
		def whileBacktrackingCondition(backtrackingValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32]):
			comparand = backtrackingValues[2]
			return jax.numpy.logical_and(comparand > 0, jax.numpy.equal(activeGap1ndex, gapRangeStart.at[comparand - 1].get()))

		def whileBacktrackingDo(backtrackingValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32]):
			backtrackAbove, backtrackBelow, activeLeafNumber = backtrackingValues

			activeLeafNumber = activeLeafNumber - 1
			backtrackBelow = backtrackBelow.at[backtrackAbove.at[activeLeafNumber].get()].set(backtrackBelow.at[activeLeafNumber].get())
			backtrackAbove = backtrackAbove.at[backtrackBelow.at[activeLeafNumber].get()].set(backtrackAbove.at[activeLeafNumber].get())

			return (backtrackAbove, backtrackBelow, activeLeafNumber)

		def if_activeLeaf1ndex_greaterThan_0(activeLeafNumber):
			return activeLeafNumber > 0

		def if_activeLeaf1ndex_greaterThan_0_do(leafPlacementValues: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
			placeLeafAbove, placeLeafBelow, placeGapRangeStart, activeLeafNumber, activeGapNumber = leafPlacementValues
			activeGapNumber = activeGapNumber - 1
			placeLeafAbove = placeLeafAbove.at[activeLeafNumber].set(gapsWhere.at[activeGapNumber].get())
			placeLeafBelow = placeLeafBelow.at[activeLeafNumber].set(placeLeafBelow.at[placeLeafAbove.at[activeLeafNumber].get()].get())
			placeLeafBelow = placeLeafBelow.at[placeLeafAbove.at[activeLeafNumber].get()].set(activeLeafNumber)
			placeLeafAbove = placeLeafAbove.at[placeLeafBelow.at[activeLeafNumber].get()].set(activeLeafNumber)
			placeGapRangeStart = placeGapRangeStart.at[activeLeafNumber].set(activeGapNumber)

			activeLeafNumber = 1 + activeLeafNumber
			return (placeLeafAbove, placeLeafBelow, placeGapRangeStart, activeLeafNumber, activeGapNumber)

		leafAbove, leafBelow, _2, gapRangeStart, gapsWhere, _5, activeLeaf1ndex, activeGap1ndex = allValues

		whileBacktrackingValues = (leafAbove, leafBelow, activeLeaf1ndex)
		whileBacktrackingValues = jax.lax.while_loop(whileBacktrackingCondition, whileBacktrackingDo, whileBacktrackingValues)
		leafAbove, leafBelow, activeLeaf1ndex = whileBacktrackingValues

		if_activeLeaf1ndex_greaterThan_0_values = (leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex)
		if_activeLeaf1ndex_greaterThan_0_values = jax.lax.cond(if_activeLeaf1ndex_greaterThan_0(activeLeaf1ndex), if_activeLeaf1ndex_greaterThan_0_do, doNothing, if_activeLeaf1ndex_greaterThan_0_values)
		leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex = if_activeLeaf1ndex_greaterThan_0_values

		return (leafAbove, leafBelow, allValues[2], gapRangeStart, gapsWhere, allValues[5], activeLeaf1ndex, activeGap1ndex)

	# Dynamic values
	A = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeMedium)
	B = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeMedium)
	count = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeMedium)
	gapter = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeMedium)
	gap = jax.numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)

	foldingsTotal = jax.numpy.uint32(0)
	l = jax.numpy.uint32(1)
	g = jax.numpy.uint32(0)

	foldingsValues = (A, B, count, gapter, gap, foldingsTotal, l, g)
	foldingsValues = jax.lax.while_loop(while_activeLeaf1ndex_greaterThan_0, countFoldings, foldingsValues)
	return foldingsValues[5]

foldingsJAX = jax.jit(foldingsJAX, static_argnums=(0, 1))
