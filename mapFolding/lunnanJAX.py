"""
Plan to refactor in JAX
- change the control flow to JAX style without using JAX
- identifiers for variables must signal the scope from which they are drawing their value.
- function nesting:
    - So that functions may use variables with a "higher" scope, nest a function that uses a variable from a higher scope.
    - So that a function does not unintentionally use a variable from a higher scope, do not nest a function that does not use a variable from a higher scope.
- comparison functions and identifiers
    - to avoid using a value from the wrong scope, use generic identifiers for comparison functions
    - to signal the original Python statement on which the function is based, use the original Python statement as the identifier
- numpy usage must be conspicuous
"""

from mapFolding import validateListDimensions, getLeavesTotal
from mapFolding.beDRY import makeConnectionGraph
from typing import List
# import chex
import jax
# import jaxtyping
import numpy as NUMERICALPYTHON

dtypeMaximumNUMERICALPYTHON = NUMERICALPYTHON.uint32
dtypeDefaultNUMERICALPYTHON = NUMERICALPYTHON.uint8
dtypeDefault = jax.numpy.uint8
dtypeMaximum = jax.numpy.uint32

def Z0Z_cond(condition, doX, doY, argument):
    if condition:
        return doX(argument)
    else:
        return doY(argument)

def Z0Z_while_loop(condition, do, argument):
    while condition(argument):
        argument = do(argument)
    return argument

def doNothing(argument):
    return argument

def foldings(listDimensions: List[int]) -> int:
    listDimensionsPositive: List[int] = validateListDimensions(listDimensions)

    # Unchanging values
    leavesTotal: int = getLeavesTotal(listDimensionsPositive)
    dimensionsTotal: int = len(listDimensions)
    connectionGraph: NUMERICALPYTHON.ndarray[NUMERICALPYTHON.int32, NUMERICALPYTHON.dtype[NUMERICALPYTHON.int32]] = makeConnectionGraph(listDimensionsPositive)
    # D: NUMERICALPYTHON.ndarray[NUMERICALPYTHON.int32, NUMERICALPYTHON.dtype[NUMERICALPYTHON.int32]] = makeConnectionGraph(listDimensionsPositive)
    # connectionGraph = jax.numpy.asarray(D, dtype=dtypeDefault)
    # del D
    del listDimensionsPositive

    def while_activeLeaf1ndex_greaterThan_0(countFoldingsValues):
        comparand = countFoldingsValues[6]
        return comparand > 0

    def countFoldings(allValues):
        leafBelow: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[1]
        activeLeaf1ndex: int = allValues[6]

        allValues = Z0Z_cond(findGapsCondition(leafBelow[0], activeLeaf1ndex),
                            lambda argumentX: dao(findGapsDo(argumentX)),
                            lambda argumentY: Z0Z_cond(incrementCondition(leafBelow[0], activeLeaf1ndex),
                                                        lambda argumentZ: dao(incrementDo(argumentZ)),
                                                        dao,
                                                        argumentY),
                            allValues)
        
        return allValues

    def findGapsCondition(leafBelowSentinel, activeLeafNumber):
        return NUMERICALPYTHON.logical_or(NUMERICALPYTHON.logical_and(leafBelowSentinel == 1, activeLeafNumber <= leavesTotal), activeLeafNumber <= 1)

    def findGapsDo(allValues):
        leafAbove: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[0]
        leafBelow: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[1]
        countDimensionsGapped: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[2]
        gapRangeStart: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[3]
        potentialGaps: NUMERICALPYTHON.ndarray[dtypeMaximumNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeMaximumNUMERICALPYTHON]] = allValues[4]
        foldingsSubTotal: int = allValues[5]
        activeLeaf1ndex: int = allValues[6]
        activeGap1ndex: int = allValues[7]

        unconstrainedLeaf: int = 0
        gap1ndexLowerBound: int = int(gapRangeStart[activeLeaf1ndex - 1])
        activeGap1ndex = gap1ndexLowerBound

        def for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values):
            comparand = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values[4]
            return comparand <= dimensionsTotal

        def for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values):
            dimensions_countDimensionsGapped: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values[0]
            dimensions_potentialGaps: NUMERICALPYTHON.ndarray[dtypeMaximumNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeMaximumNUMERICALPYTHON]] = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values[1]
            dimensions_gap1ndexLowerBound: int = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values[2]
            dimensions_unconstrainedLeaf: int = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values[3]
            dimensionNumber: int = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values[4]

            def ifLeafIsUnconstrainedCondition(comparand):
                return connectionGraph[comparand][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex

            def ifLeafIsUnconstrainedDo(unconstrainedValues):
                unconstrained_countDimensionsGapped: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = unconstrainedValues[0]
                unconstrained_potentialGaps: NUMERICALPYTHON.ndarray[dtypeMaximumNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeMaximumNUMERICALPYTHON]] = unconstrainedValues[1]
                unconstrained_gap1ndexLowerBound: int = unconstrainedValues[2]
                unconstrained_unconstrainedLeaf: int = unconstrainedValues[3]

                unconstrained_unconstrainedLeaf += 1

                unconstrainedValues = (unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, unconstrained_unconstrainedLeaf)
                return unconstrainedValues

            def ifLeafIsUnconstrainedElse(unconstrainedValues):
                unconstrained_countDimensionsGapped: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = unconstrainedValues[0]
                unconstrained_potentialGaps: NUMERICALPYTHON.ndarray[dtypeMaximumNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeMaximumNUMERICALPYTHON]] = unconstrainedValues[1]
                unconstrained_gap1ndexLowerBound: int = unconstrainedValues[2]
                unconstrained_unconstrainedLeaf: int = unconstrainedValues[3]

                leaf1ndexConnectee: int = connectionGraph[dimensionNumber][activeLeaf1ndex][activeLeaf1ndex]

                def while_leaf1ndexConnectee_notEquals_activeLeaf1ndex(countGapsValues):
                    comparand = countGapsValues[3]
                    return comparand != activeLeaf1ndex

                def countGaps(countGapsValues):
                    countGapsCountDimensionsGapped: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = countGapsValues[0]
                    countGapsPotentialGaps: NUMERICALPYTHON.ndarray[dtypeMaximumNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeMaximumNUMERICALPYTHON]] = countGapsValues[1]
                    countGapsGap1ndexLowerBound: int = countGapsValues[2]
                    countGapsLeaf1ndexConnectee: int = countGapsValues[3]

                    countGapsPotentialGaps[countGapsGap1ndexLowerBound] = countGapsLeaf1ndexConnectee
                    if countGapsCountDimensionsGapped[countGapsLeaf1ndexConnectee] == 0:
                        countGapsGap1ndexLowerBound += 1
                    countGapsCountDimensionsGapped[countGapsLeaf1ndexConnectee] += 1
                    countGapsLeaf1ndexConnectee = connectionGraph[dimensionNumber][activeLeaf1ndex][leafBelow[countGapsLeaf1ndexConnectee]]

                    countGapsValues = (countGapsCountDimensionsGapped, countGapsPotentialGaps, countGapsGap1ndexLowerBound, countGapsLeaf1ndexConnectee)
                    return countGapsValues

                countGapsValues = (unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, leaf1ndexConnectee)

                countGapsValues = Z0Z_while_loop(while_leaf1ndexConnectee_notEquals_activeLeaf1ndex, countGaps, countGapsValues)

                unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, leaf1ndexConnectee = countGapsValues

                unconstrainedValues = (unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, unconstrained_unconstrainedLeaf)

                return unconstrainedValues

            ifLeafIsUnconstrainedValues = (dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf)

            ifLeafIsUnconstrainedValues = Z0Z_cond(ifLeafIsUnconstrainedCondition(dimensionNumber), ifLeafIsUnconstrainedDo, ifLeafIsUnconstrainedElse, ifLeafIsUnconstrainedValues)

            dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf = ifLeafIsUnconstrainedValues

            dimensionNumber += 1
            for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = (dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf, dimensionNumber)
            return for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values

        dimension1ndex = 1
        for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = (countDimensionsGapped, potentialGaps, gap1ndexLowerBound, unconstrainedLeaf, dimension1ndex)
        
        for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = Z0Z_while_loop(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values)

        countDimensionsGapped, potentialGaps, gap1ndexLowerBound, unconstrainedLeaf, dimension1ndex = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values
        del dimension1ndex

        """If activeLeaf1ndex is unconstrained in all sections, it can be inserted anywhere"""
        leaf1ndex = 0
        for_leaf1ndex_in_range_activeLeaf1ndexValues = (potentialGaps, gap1ndexLowerBound, leaf1ndex)

        def almostUselessCondition(comparand):
            return comparand == dimensionsTotal
        
        def almostUselessConditionDo(for_leaf1ndex_in_range_activeLeaf1ndexValues):
            def for_leaf1ndex_in_range_activeLeaf1ndex(for_leaf1ndex_in_range_activeLeaf1ndexValues):
                comparand = for_leaf1ndex_in_range_activeLeaf1ndexValues[2]
                return comparand < activeLeaf1ndex
            
            def for_leaf1ndex_in_range_activeLeaf1ndex_do(for_leaf1ndex_in_range_activeLeaf1ndexValues):
                leafInRangePotentialGaps, gapNumberLowerBound, leafNumber = for_leaf1ndex_in_range_activeLeaf1ndexValues
                leafInRangePotentialGaps[gapNumberLowerBound] = leafNumber
                gapNumberLowerBound += 1
                leafNumber += 1
                return (leafInRangePotentialGaps, gapNumberLowerBound, leafNumber)
        
            return Z0Z_while_loop(
                for_leaf1ndex_in_range_activeLeaf1ndex,
                for_leaf1ndex_in_range_activeLeaf1ndex_do,
                for_leaf1ndex_in_range_activeLeaf1ndexValues
            )
        
        for_leaf1ndex_in_range_activeLeaf1ndexValues = Z0Z_cond(
            almostUselessCondition(unconstrainedLeaf),
            almostUselessConditionDo,
            doNothing,
            for_leaf1ndex_in_range_activeLeaf1ndexValues
        )

        potentialGaps, gap1ndexLowerBound, leaf1ndex = for_leaf1ndex_in_range_activeLeaf1ndexValues
        del leaf1ndex

        """Filter gaps that are common to all sections"""
        for indexMiniGap in range(activeGap1ndex, gap1ndexLowerBound):
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if countDimensionsGapped[potentialGaps[indexMiniGap]] == dimensionsTotal - unconstrainedLeaf:
                activeGap1ndex += 1
            """Reset countDimensionsGapped for next iteration"""
            countDimensionsGapped[potentialGaps[indexMiniGap]] = 0

        allValues = (leafAbove, leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, foldingsSubTotal, activeLeaf1ndex, activeGap1ndex)

        return allValues

    def incrementCondition(leafBelowSentinel, activeLeafNumber):
        return NUMERICALPYTHON.logical_and(activeLeafNumber > leavesTotal, leafBelowSentinel == 1)

    def incrementDo(allValues):
        leafAbove: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[0]
        leafBelow: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[1]
        countDimensionsGapped: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[2]
        gapRangeStart: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[3]
        potentialGaps: NUMERICALPYTHON.ndarray[dtypeMaximumNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeMaximumNUMERICALPYTHON]] = allValues[4]
        foldingsSubTotal: int = allValues[5]
        activeLeaf1ndex: int = allValues[6]
        activeGap1ndex: int = allValues[7]

        """Increment foldingsSubTotal"""
        foldingsSubTotal += leavesTotal

        allValues = (leafAbove, leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, foldingsSubTotal, activeLeaf1ndex, activeGap1ndex)

        return allValues

    def dao(allValues):
        leafAbove: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[0]
        leafBelow: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[1]
        countDimensionsGapped: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[2]
        gapRangeStart: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = allValues[3]
        potentialGaps: NUMERICALPYTHON.ndarray[dtypeMaximumNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeMaximumNUMERICALPYTHON]] = allValues[4]
        foldingsSubTotal: int = allValues[5]
        activeLeaf1ndex: int = allValues[6]
        activeGap1ndex: int = allValues[7]

        def whileBacktrackingCondition(backtrackingValues):
            activeLeafNumber = backtrackingValues[2]
            return NUMERICALPYTHON.logical_and(activeLeafNumber > 0, activeGap1ndex == gapRangeStart[activeLeafNumber - 1])

        def whileBacktrackingDo(backtrackingValues):
            backtrackAbove: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = backtrackingValues[0]
            backtrackBelow: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = backtrackingValues[1]
            activeLeafNumber: int = backtrackingValues[2]

            activeLeafNumber -= 1
            backtrackBelow[backtrackAbove[activeLeafNumber]] = backtrackBelow[activeLeafNumber]
            backtrackAbove[backtrackBelow[activeLeafNumber]] = backtrackAbove[activeLeafNumber]

            backtrackingValues = (backtrackAbove, backtrackBelow, activeLeafNumber)
            return backtrackingValues

        whileBacktrackingValues = (leafAbove, leafBelow, activeLeaf1ndex)

        whileBacktrackingValues = Z0Z_while_loop(whileBacktrackingCondition, whileBacktrackingDo, whileBacktrackingValues)

        leafAbove, leafBelow, activeLeaf1ndex = whileBacktrackingValues

        def if_activeLeaf1ndex_greaterThan_0(activeLeafNumber):
            return activeLeafNumber > 0

        def if_activeLeaf1ndex_greaterThan_0_do(leafPlacementValues):
            placeLeafAbove, placeLeafBelow, placeGapRangeStart, activeLeafNumber, activeGapNumber = leafPlacementValues
            activeGapNumber -= 1
            placeLeafAbove[activeLeafNumber] = potentialGaps[activeGapNumber]
            placeLeafBelow[activeLeafNumber] = placeLeafBelow[placeLeafAbove[activeLeafNumber]]
            placeLeafBelow[placeLeafAbove[activeLeafNumber]] = activeLeafNumber
            placeLeafAbove[placeLeafBelow[activeLeafNumber]] = activeLeafNumber
            placeGapRangeStart[activeLeafNumber] = activeGapNumber
            activeLeafNumber += 1
            return (placeLeafAbove, placeLeafBelow, placeGapRangeStart, activeLeafNumber, activeGapNumber)

        if_activeLeaf1ndex_greaterThan_0_values = (leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex)

        if_activeLeaf1ndex_greaterThan_0_values = Z0Z_cond(if_activeLeaf1ndex_greaterThan_0(activeLeaf1ndex),
                                                        if_activeLeaf1ndex_greaterThan_0_do,
                                                        doNothing,
                                                        if_activeLeaf1ndex_greaterThan_0_values)

        leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex = if_activeLeaf1ndex_greaterThan_0_values

        allValues = (leafAbove, leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, foldingsSubTotal, activeLeaf1ndex, activeGap1ndex)

        return allValues

    # Dynamic values
    A = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    B = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    count = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    gapter = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    gap = NUMERICALPYTHON.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximumNUMERICALPYTHON)

    # A = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    # B = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    # count = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    # gapter = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    # gap = jax.numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)

    foldingsTotal: int = 0
    l: int = 1
    g: int = 0

    foldingsTotal = jax.numpy.uint32(0)
    l: int = 1
    g = jax.numpy.uint8(0)

    foldingsValues = (A, B, count, gapter, gap, foldingsTotal, l, g)

    foldingsValues = Z0Z_while_loop(while_activeLeaf1ndex_greaterThan_0, countFoldings, foldingsValues)

    foldingsTotal = foldingsValues[5]
    return foldingsTotal
