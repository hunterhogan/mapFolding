from mapFolding import validateListDimensions, getLeavesTotal, makeConnectionGraph
from typing import List, Tuple
import jax
import jaxtyping

dtypeDefault = jax.numpy.uint32
dtypeMaximum = jax.numpy.uint32

def countFolds(listDimensions: List[int]) -> int:
    listDimensionsPositive: List[int] = validateListDimensions(listDimensions)

    n: int = getLeavesTotal(listDimensionsPositive)
    d: int = len(listDimensions)
    import numpy
    D: numpy.ndarray[numpy.int32, numpy.dtype[numpy.int32]] = makeConnectionGraph(listDimensionsPositive)
    connectionGraph = jax.numpy.asarray(D, dtype=dtypeDefault)
    del listDimensionsPositive

    return foldingsJAX(n, d, connectionGraph)

def foldingsJAX(leavesTotal: jaxtyping.UInt32, dimensionsTotal: jaxtyping.UInt32, connectionGraph: jaxtyping.Array) -> jaxtyping.UInt32:

    def doNothing(argument):
        return argument

    def while_activeLeaf1ndex_greaterThan_0(comparisonValues: Tuple):
        comparand = comparisonValues[6]
        return comparand > 0

    def countFoldings(allValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
        _0, leafBelow, _2, _3, _4, _5, activeLeaf1ndex, _7 = allValues

        sentinel = leafBelow.at[0].get().astype(jax.numpy.uint32)

        allValues = jax.lax.cond(findGapsCondition(sentinel, activeLeaf1ndex),
                            lambda argumentX: dao(findGapsDo(argumentX)),
                            lambda argumentY: jax.lax.cond(incrementCondition(sentinel, activeLeaf1ndex), lambda argumentZ: dao(incrementDo(argumentZ)), dao, argumentY),
                            allValues)

        return allValues

    def findGapsCondition(leafBelowSentinel, activeLeafNumber):
        return jax.numpy.logical_or(jax.numpy.logical_and(leafBelowSentinel == 1, activeLeafNumber <= leavesTotal), activeLeafNumber <= 1)

    def findGapsDo(allValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
        def for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1(comparisonValues: Tuple):
            return comparisonValues[-1] <= dimensionsTotal

        def for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
            def ifLeafIsUnconstrainedCondition(comparand):
                return jax.numpy.equal(connectionGraph[comparand, activeLeaf1ndex, activeLeaf1ndex], activeLeaf1ndex)

            def ifLeafIsUnconstrainedDo(unconstrainedValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
                unconstrained_unconstrainedLeaf = unconstrainedValues[3]
                unconstrained_unconstrainedLeaf = 1 + unconstrained_unconstrainedLeaf
                return (unconstrainedValues[0], unconstrainedValues[1], unconstrainedValues[2], unconstrained_unconstrainedLeaf)

            def ifLeafIsUnconstrainedElse(unconstrainedValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
                def while_leaf1ndexConnectee_notEquals_activeLeaf1ndex(comparisonValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
                    return comparisonValues[-1] != activeLeaf1ndex

                def countGaps(countGapsDoValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
                    countGapsCountDimensionsGapped, countGapsPotentialGaps, countGapsGap1ndexLowerBound, countGapsLeaf1ndexConnectee = countGapsDoValues

                    countGapsPotentialGaps = countGapsPotentialGaps.at[countGapsGap1ndexLowerBound].set(countGapsLeaf1ndexConnectee)
                    countGapsGap1ndexLowerBound = jax.numpy.where(jax.numpy.equal(countGapsCountDimensionsGapped[countGapsLeaf1ndexConnectee], 0), countGapsGap1ndexLowerBound + 1, countGapsGap1ndexLowerBound)
                    countGapsCountDimensionsGapped = countGapsCountDimensionsGapped.at[countGapsLeaf1ndexConnectee].add(1)
                    countGapsLeaf1ndexConnectee = connectionGraph.at[dimensionNumber, activeLeaf1ndex, leafBelow.at[countGapsLeaf1ndexConnectee].get()].get().astype(jax.numpy.uint32)

                    return (countGapsCountDimensionsGapped, countGapsPotentialGaps, countGapsGap1ndexLowerBound, countGapsLeaf1ndexConnectee)

                unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, unconstrained_unconstrainedLeaf = unconstrainedValues

                leaf1ndexConnectee = connectionGraph.at[dimensionNumber, activeLeaf1ndex, activeLeaf1ndex].get().astype(jax.numpy.uint32)

                countGapsValues = (unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, leaf1ndexConnectee)
                countGapsValues = jax.lax.while_loop(while_leaf1ndexConnectee_notEquals_activeLeaf1ndex, countGaps, countGapsValues)
                unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, leaf1ndexConnectee = countGapsValues

                return (unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, unconstrained_unconstrainedLeaf)

            dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf, dimensionNumber = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values

            ifLeafIsUnconstrainedValues = (dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf)
            ifLeafIsUnconstrainedValues = jax.lax.cond(ifLeafIsUnconstrainedCondition(dimensionNumber), ifLeafIsUnconstrainedDo, ifLeafIsUnconstrainedElse, ifLeafIsUnconstrainedValues)
            dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf = ifLeafIsUnconstrainedValues

            dimensionNumber = 1 + dimensionNumber
            return (dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf, dimensionNumber)

        def almostUselessCondition(comparand):
            return comparand == dimensionsTotal

        def almostUselessConditionDo(for_leaf1ndex_in_range_activeLeaf1ndexValues: Tuple[jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
            def for_leaf1ndex_in_range_activeLeaf1ndex(comparisonValues):
                return comparisonValues[-1] < activeLeaf1ndex

            def for_leaf1ndex_in_range_activeLeaf1ndex_do(for_leaf1ndex_in_range_activeLeaf1ndexValues: Tuple[jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
                leafInRangePotentialGaps, gapNumberLowerBound, leafNumber = for_leaf1ndex_in_range_activeLeaf1ndexValues
                leafInRangePotentialGaps = leafInRangePotentialGaps.at[gapNumberLowerBound].set(leafNumber)
                gapNumberLowerBound = 1 + gapNumberLowerBound
                leafNumber = 1 + leafNumber
                return (leafInRangePotentialGaps, gapNumberLowerBound, leafNumber)
            return jax.lax.while_loop(for_leaf1ndex_in_range_activeLeaf1ndex, for_leaf1ndex_in_range_activeLeaf1ndex_do, for_leaf1ndex_in_range_activeLeaf1ndexValues)

        def for_range_from_activeGap1ndex_to_gap1ndexLowerBound(comparisonValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
            return comparisonValues[-1] < gap1ndexLowerBound

        def miniGapDo(gapToGapValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
            gapToGapCountDimensionsGapped, gapToGapPotentialGaps, activeGapNumber, index = gapToGapValues
            gapToGapPotentialGaps = gapToGapPotentialGaps.at[activeGapNumber].set(gapToGapPotentialGaps.at[index].get())
            activeGapNumber = jax.numpy.where(jax.numpy.equal(gapToGapCountDimensionsGapped.at[gapToGapPotentialGaps.at[index].get()].get(), dimensionsTotal - unconstrainedLeaf), activeGapNumber + 1, activeGapNumber).astype(jax.numpy.uint32)
            gapToGapCountDimensionsGapped = gapToGapCountDimensionsGapped.at[gapToGapPotentialGaps.at[index].get()].set(0)
            index = 1 + index
            return (gapToGapCountDimensionsGapped, gapToGapPotentialGaps, activeGapNumber, index)

        _0, leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, _5, activeLeaf1ndex, activeGap1ndex = allValues

        unconstrainedLeaf = jax.numpy.uint32(0)
        dimension1ndex = jax.numpy.uint32(1)
        gap1ndexLowerBound = gapRangeStart.at[activeLeaf1ndex - 1].get().astype(jax.numpy.uint32)
        activeGap1ndex = gap1ndexLowerBound
        for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = (countDimensionsGapped, potentialGaps, gap1ndexLowerBound, unconstrainedLeaf, dimension1ndex)
        for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = jax.lax.while_loop(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values)
        countDimensionsGapped, potentialGaps, gap1ndexLowerBound, unconstrainedLeaf, dimension1ndex = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values
        del dimension1ndex

        leaf1ndex = jax.numpy.uint32(0)
        for_leaf1ndex_in_range_activeLeaf1ndexValues = (potentialGaps, gap1ndexLowerBound, leaf1ndex)
        for_leaf1ndex_in_range_activeLeaf1ndexValues = jax.lax.cond(almostUselessCondition(unconstrainedLeaf), almostUselessConditionDo, doNothing, for_leaf1ndex_in_range_activeLeaf1ndexValues)
        potentialGaps, gap1ndexLowerBound, leaf1ndex = for_leaf1ndex_in_range_activeLeaf1ndexValues
        del leaf1ndex

        indexMiniGap = activeGap1ndex
        miniGapValues = (countDimensionsGapped, potentialGaps, activeGap1ndex, indexMiniGap)
        miniGapValues = jax.lax.while_loop(for_range_from_activeGap1ndex_to_gap1ndexLowerBound, miniGapDo, miniGapValues)
        countDimensionsGapped, potentialGaps, activeGap1ndex, indexMiniGap = miniGapValues
        del indexMiniGap

        return (allValues[0], leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, allValues[5], activeLeaf1ndex, activeGap1ndex)

    def incrementCondition(leafBelowSentinel, activeLeafNumber):
        return jax.numpy.logical_and(activeLeafNumber > leavesTotal, leafBelowSentinel == 1)

    def incrementDo(allValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
        foldingsSubTotal = allValues[5]
        foldingsSubTotal = leavesTotal + foldingsSubTotal
        return (allValues[0], allValues[1], allValues[2], allValues[3], allValues[4], foldingsSubTotal, allValues[6], allValues[7])

    def dao(allValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
        def whileBacktrackingCondition(backtrackingValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32]):
            comparand = backtrackingValues[2]
            return jax.numpy.logical_and(comparand > 0, jax.numpy.equal(activeGap1ndex, gapRangeStart.at[comparand - 1].get()))

        def whileBacktrackingDo(backtrackingValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32]):
            backtrackAbove, backtrackBelow, activeLeafNumber = backtrackingValues

            activeLeafNumber = activeLeafNumber - 1
            backtrackBelow = backtrackBelow.at[backtrackAbove.at[activeLeafNumber].get()].set(backtrackBelow.at[activeLeafNumber].get())
            backtrackAbove = backtrackAbove.at[backtrackBelow.at[activeLeafNumber].get()].set(backtrackAbove.at[activeLeafNumber].get())

            return (backtrackAbove, backtrackBelow, activeLeafNumber)

        def if_activeLeaf1ndex_greaterThan_0(activeLeafNumber):
            return activeLeafNumber > 0

        def if_activeLeaf1ndex_greaterThan_0_do(leafPlacementValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
            placeLeafAbove, placeLeafBelow, placeGapRangeStart, activeLeafNumber, activeGapNumber = leafPlacementValues
            activeGapNumber = activeGapNumber - 1
            placeLeafAbove = placeLeafAbove.at[activeLeafNumber].set(potentialGaps.at[activeGapNumber].get())
            placeLeafBelow = placeLeafBelow.at[activeLeafNumber].set(placeLeafBelow.at[placeLeafAbove.at[activeLeafNumber].get()].get())
            placeLeafBelow = placeLeafBelow.at[placeLeafAbove.at[activeLeafNumber].get()].set(activeLeafNumber)
            placeLeafAbove = placeLeafAbove.at[placeLeafBelow.at[activeLeafNumber].get()].set(activeLeafNumber)
            placeGapRangeStart = placeGapRangeStart.at[activeLeafNumber].set(activeGapNumber)

            activeLeafNumber = 1 + activeLeafNumber
            return (placeLeafAbove, placeLeafBelow, placeGapRangeStart, activeLeafNumber, activeGapNumber)

        leafAbove, leafBelow, _2, gapRangeStart, potentialGaps, _5, activeLeaf1ndex, activeGap1ndex = allValues

        whileBacktrackingValues = (leafAbove, leafBelow, activeLeaf1ndex)
        whileBacktrackingValues = jax.lax.while_loop(whileBacktrackingCondition, whileBacktrackingDo, whileBacktrackingValues)
        leafAbove, leafBelow, activeLeaf1ndex = whileBacktrackingValues

        if_activeLeaf1ndex_greaterThan_0_values = (leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex)
        if_activeLeaf1ndex_greaterThan_0_values = jax.lax.cond(if_activeLeaf1ndex_greaterThan_0(activeLeaf1ndex), if_activeLeaf1ndex_greaterThan_0_do, doNothing, if_activeLeaf1ndex_greaterThan_0_values)
        leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex = if_activeLeaf1ndex_greaterThan_0_values

        return (leafAbove, leafBelow, allValues[2], gapRangeStart, potentialGaps, allValues[5], activeLeaf1ndex, activeGap1ndex)

    # Dynamic values
    A = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    B = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    count = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    gapter = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    gap = jax.numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)

    foldingsTotal = jax.numpy.uint32(0)
    l = jax.numpy.uint32(1)
    g = jax.numpy.uint32(0)

    foldingsValues = (A, B, count, gapter, gap, foldingsTotal, l, g)
    foldingsValues = jax.lax.while_loop(while_activeLeaf1ndex_greaterThan_0, countFoldings, foldingsValues)
    return foldingsValues[5]

foldingsJAX = jax.jit(foldingsJAX, static_argnums=(0, 1))