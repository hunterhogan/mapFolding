"""
Plan to refactor in JAX
- function nesting:
    - So that functions may use variables with a "higher" scope, nest a function that uses a variable from a higher scope.
    - So that a function does not unintentionally use a variable from a higher scope, do not nest a function that does not use a variable from a higher scope.
- comparison functions and identifiers
    - to avoid using a value from the wrong scope, use generic identifiers for comparison functions
    - to signal the original Python statement on which the function is based, use the original Python statement as the identifier
- numpy usage must be conspicuous
- use `chex`
    assert when values should have changed
    assert when values should not have changed
    assert shapes/ranks
    assert dtypes
    create chex statements before change to jax and use chex.disable_asserts() to disable them
- Typing, tuples, assignments, and `chex`
    Instead of unpacking and using typing
    Unpack with one statement and use chex assertions where needed
        when unpacking , use `_0`, `_1`, `_2`, ... `_N` as placeholders for the unused elements of a tuple: never just `_`
    Then, `=` is a strong signal that an assignment is happening and that I must conform to JAX rules
"""

from mapFolding import validateListDimensions, getLeavesTotal
from mapFolding.beDRY import makeConnectionGraph
from typing import List
import chex
import jax
import jaxtyping
import numpy as NUMERICALPYTHON

chex.disable_asserts()

dtypeMaximumNUMERICALPYTHON = NUMERICALPYTHON.uint32
dtypeDefaultNUMERICALPYTHON = NUMERICALPYTHON.uint16
dtypeDefault = jax.numpy.uint16
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
    def while_activeLeaf1ndex_greaterThan_0(comparisonValues):
        comparand = comparisonValues[6]
        return comparand > 0

    def countFoldings(allValues):
        _0, leafBelow, _2, _3, _4, _5, activeLeaf1ndex, _7 = allValues

        allValues = Z0Z_cond(findGapsCondition(leafBelow[0], activeLeaf1ndex),
                            lambda argumentX: dao(findGapsDo(argumentX)),
                            lambda argumentY: Z0Z_cond(incrementCondition(leafBelow[0], activeLeaf1ndex), lambda argumentZ: dao(incrementDo(argumentZ)), dao, argumentY),
                            allValues)

        return allValues

    def findGapsCondition(leafBelowSentinel, activeLeafNumber):
        return NUMERICALPYTHON.logical_or(NUMERICALPYTHON.logical_and(leafBelowSentinel == 1, activeLeafNumber <= leavesTotal), activeLeafNumber <= 1)

    def findGapsDo(allValues):
        def for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1(comparisonValues):
            return comparisonValues[-1] <= dimensionsTotal

        def for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values):
            def ifLeafIsUnconstrainedCondition(comparand):
                return connectionGraph[comparand][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex

            def ifLeafIsUnconstrainedDo(unconstrainedValues):
                unconstrained_unconstrainedLeaf = unconstrainedValues[3]
                unconstrained_unconstrainedLeaf += 1
                return (unconstrainedValues[0], unconstrainedValues[1], unconstrainedValues[2], unconstrained_unconstrainedLeaf)

            def ifLeafIsUnconstrainedElse(unconstrainedValues):
                def while_leaf1ndexConnectee_notEquals_activeLeaf1ndex(comparisonValues):
                    return comparisonValues[-1] != activeLeaf1ndex

                def countGaps(countGapsDoValues):
                    countGapsCountDimensionsGapped, countGapsPotentialGaps, countGapsGap1ndexLowerBound, countGapsLeaf1ndexConnectee = countGapsDoValues

                    countGapsPotentialGaps[countGapsGap1ndexLowerBound] = countGapsLeaf1ndexConnectee
                    countGapsGap1ndexLowerBound = NUMERICALPYTHON.where(countGapsCountDimensionsGapped[countGapsLeaf1ndexConnectee] == 0, countGapsGap1ndexLowerBound + 1, countGapsGap1ndexLowerBound)
                    countGapsCountDimensionsGapped[countGapsLeaf1ndexConnectee] += 1
                    countGapsLeaf1ndexConnectee = connectionGraph[dimensionNumber][activeLeaf1ndex][leafBelow[countGapsLeaf1ndexConnectee]]

                    return (countGapsCountDimensionsGapped, countGapsPotentialGaps, countGapsGap1ndexLowerBound, countGapsLeaf1ndexConnectee)

                unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, unconstrained_unconstrainedLeaf = unconstrainedValues

                leaf1ndexConnectee = connectionGraph[dimensionNumber][activeLeaf1ndex][activeLeaf1ndex]

                countGapsValues = (unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, leaf1ndexConnectee)
                countGapsValues = Z0Z_while_loop(while_leaf1ndexConnectee_notEquals_activeLeaf1ndex, countGaps, countGapsValues)
                unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, leaf1ndexConnectee = countGapsValues

                return (unconstrained_countDimensionsGapped, unconstrained_potentialGaps, unconstrained_gap1ndexLowerBound, unconstrained_unconstrainedLeaf)

            dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf, dimensionNumber = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values

            ifLeafIsUnconstrainedValues = (dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf)
            ifLeafIsUnconstrainedValues = Z0Z_cond(ifLeafIsUnconstrainedCondition(dimensionNumber), ifLeafIsUnconstrainedDo, ifLeafIsUnconstrainedElse, ifLeafIsUnconstrainedValues)
            dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf = ifLeafIsUnconstrainedValues

            dimensionNumber += 1
            return (dimensions_countDimensionsGapped, dimensions_potentialGaps, dimensions_gap1ndexLowerBound, dimensions_unconstrainedLeaf, dimensionNumber)

        def almostUselessCondition(comparand):
            return comparand == dimensionsTotal

        def almostUselessConditionDo(for_leaf1ndex_in_range_activeLeaf1ndexValues):
            def for_leaf1ndex_in_range_activeLeaf1ndex(comparisonValues):
                return comparisonValues[-1] < activeLeaf1ndex

            def for_leaf1ndex_in_range_activeLeaf1ndex_do(for_leaf1ndex_in_range_activeLeaf1ndexValues):
                leafInRangePotentialGaps, gapNumberLowerBound, leafNumber = for_leaf1ndex_in_range_activeLeaf1ndexValues
                leafInRangePotentialGaps[gapNumberLowerBound] = leafNumber
                gapNumberLowerBound += 1
                leafNumber += 1
                return (leafInRangePotentialGaps, gapNumberLowerBound, leafNumber)

            return Z0Z_while_loop(for_leaf1ndex_in_range_activeLeaf1ndex, for_leaf1ndex_in_range_activeLeaf1ndex_do, for_leaf1ndex_in_range_activeLeaf1ndexValues)

        def for_range_from_activeGap1ndex_to_gap1ndexLowerBound(comparisonValues):
            return comparisonValues[-1] < gap1ndexLowerBound

        def miniGapDo(gapToGapValues):
            gapToGapCountDimensionsGapped, gapToGapPotentialGaps, activeGapNumber, index = gapToGapValues
            gapToGapPotentialGaps[activeGapNumber] = gapToGapPotentialGaps[index]
            activeGapNumber = int(NUMERICALPYTHON.where(gapToGapCountDimensionsGapped[gapToGapPotentialGaps[index]] == dimensionsTotal - unconstrainedLeaf, activeGapNumber + 1, activeGapNumber))
            gapToGapCountDimensionsGapped[gapToGapPotentialGaps[index]] = 0
            index += 1
            return (gapToGapCountDimensionsGapped, gapToGapPotentialGaps, activeGapNumber, index)

        _0, leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, _5, activeLeaf1ndex, activeGap1ndex = allValues

        # unconstrainedLeaf = jax.numpy.uint16(0)
        # dimension1ndex = jax.numpy.uint16(1)
        unconstrainedLeaf: int = 0
        gap1ndexLowerBound = gapRangeStart[activeLeaf1ndex - 1]
        activeGap1ndex = gap1ndexLowerBound
        dimension1ndex: int = 1
        for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = (countDimensionsGapped, potentialGaps, gap1ndexLowerBound, unconstrainedLeaf, dimension1ndex)
        for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = Z0Z_while_loop(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values)
        countDimensionsGapped, potentialGaps, gap1ndexLowerBound, unconstrainedLeaf, dimension1ndex = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values
        del dimension1ndex

        # leaf1ndex = jax.numpy.uint16(0)
        leaf1ndex: int = 0
        for_leaf1ndex_in_range_activeLeaf1ndexValues = (potentialGaps, gap1ndexLowerBound, leaf1ndex)
        for_leaf1ndex_in_range_activeLeaf1ndexValues = Z0Z_cond(almostUselessCondition(unconstrainedLeaf), almostUselessConditionDo, doNothing, for_leaf1ndex_in_range_activeLeaf1ndexValues)
        potentialGaps, gap1ndexLowerBound, leaf1ndex = for_leaf1ndex_in_range_activeLeaf1ndexValues
        del leaf1ndex

        indexMiniGap = activeGap1ndex
        miniGapValues = (countDimensionsGapped, potentialGaps, activeGap1ndex, indexMiniGap)
        miniGapValues = Z0Z_while_loop(for_range_from_activeGap1ndex_to_gap1ndexLowerBound, miniGapDo, miniGapValues)
        countDimensionsGapped, potentialGaps, activeGap1ndex, indexMiniGap = miniGapValues
        del indexMiniGap

        # Validate array states before processing
        chex.assert_shape(countDimensionsGapped, (leavesTotal + 1,))
        chex.assert_shape(potentialGaps, (leavesTotal * leavesTotal + 1,))
        chex.assert_type([countDimensionsGapped, gapRangeStart], dtypeDefault)
        chex.assert_type(potentialGaps, dtypeMaximum)
        chex.assert_scalar_in(activeLeaf1ndex, 0, leavesTotal + 1)

        # Validate state changes
        chex.assert_tree_all_finite([countDimensionsGapped, potentialGaps])
        return (allValues[0], leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, allValues[5], activeLeaf1ndex, activeGap1ndex)

    def incrementCondition(leafBelowSentinel, activeLeafNumber):
        return NUMERICALPYTHON.logical_and(activeLeafNumber > leavesTotal, leafBelowSentinel == 1)

    def incrementDo(allValues):
        foldingsSubTotal = allValues[5]
        foldingsSubTotal += leavesTotal
        return (allValues[0], allValues[1], allValues[2], allValues[3], allValues[4], foldingsSubTotal, allValues[6], allValues[7])

    def dao(allValues):
        def whileBacktrackingCondition(backtrackingValues):
            comparand = backtrackingValues[2]
            return NUMERICALPYTHON.logical_and(comparand > 0, activeGap1ndex == gapRangeStart[comparand - 1])

        def whileBacktrackingDo(backtrackingValues):
            backtrackAbove, backtrackBelow, activeLeafNumber = backtrackingValues

            activeLeafNumber -= 1
            backtrackBelow[backtrackAbove[activeLeafNumber]] = backtrackBelow[activeLeafNumber]
            backtrackAbove[backtrackBelow[activeLeafNumber]] = backtrackAbove[activeLeafNumber]

            return (backtrackAbove, backtrackBelow, activeLeafNumber)

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

        leafAbove, leafBelow, _2, gapRangeStart, potentialGaps, _5, activeLeaf1ndex, activeGap1ndex = allValues

        whileBacktrackingValues = (leafAbove, leafBelow, activeLeaf1ndex)
        whileBacktrackingValues = Z0Z_while_loop(whileBacktrackingCondition, whileBacktrackingDo, whileBacktrackingValues)
        leafAbove, leafBelow, activeLeaf1ndex = whileBacktrackingValues

        if_activeLeaf1ndex_greaterThan_0_values = (leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex)
        if_activeLeaf1ndex_greaterThan_0_values = Z0Z_cond(if_activeLeaf1ndex_greaterThan_0(activeLeaf1ndex), if_activeLeaf1ndex_greaterThan_0_do, doNothing, if_activeLeaf1ndex_greaterThan_0_values)
        leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex = if_activeLeaf1ndex_greaterThan_0_values

        # Validate array states
        chex.assert_shape([leafAbove, leafBelow], (leavesTotal + 1,))
        chex.assert_type([leafAbove, leafBelow, gapRangeStart], dtypeDefault)
        chex.assert_scalar_in(activeLeaf1ndex, 0, leavesTotal + 1)
        chex.assert_scalar_in(activeGap1ndex, 0, leavesTotal * leavesTotal + 1)

        # Validate final states
        chex.assert_tree_all_finite([leafAbove, leafBelow, gapRangeStart])
        return (leafAbove, leafBelow, allValues[2], gapRangeStart, potentialGaps, allValues[5], activeLeaf1ndex, activeGap1ndex)

    listDimensionsPositive: List[int] = validateListDimensions(listDimensions)

    # Unchanging values
    leavesTotal: int = getLeavesTotal(listDimensionsPositive)
    dimensionsTotal: int = len(listDimensions)
    connectionGraph: NUMERICALPYTHON.ndarray[NUMERICALPYTHON.int32, NUMERICALPYTHON.dtype[NUMERICALPYTHON.int32]] = makeConnectionGraph(listDimensionsPositive)
    # n: int = getLeavesTotal(listDimensionsPositive)
    # d: int = len(listDimensions)
    # leavesTotal = jax.numpy.uint16(n)
    # dimensionsTotal = jax.numpy.uint16(d)
    # D: NUMERICALPYTHON.ndarray[NUMERICALPYTHON.int32, NUMERICALPYTHON.dtype[NUMERICALPYTHON.int32]] = makeConnectionGraph(listDimensionsPositive)
    # connectionGraph = jax.numpy.asarray(D, dtype=dtypeDefault)
    # del n, d, D
    del listDimensionsPositive

    # Validate dimensions
    chex.assert_rank(connectionGraph, 3)  # Should be 3D array [dim][leaf][leaf]
    chex.assert_shape(connectionGraph, (dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1))
    chex.assert_type(connectionGraph, dtypeDefault)

    # Dynamic values
    A = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    B = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    count = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    gapter = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    gap = NUMERICALPYTHON.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximumNUMERICALPYTHON)

    foldingsTotal: int = 0
    l: int = 1
    g: int = 0

    # A = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    # B = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    # count = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    # gapter = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    # gap = jax.numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)

    # foldingsTotal = jax.numpy.uint32(0)
    # l = jax.numpy.uint16(1)
    # g = jax.numpy.uint16(0)

    # Validate dynamic arrays initialization
    chex.assert_shape(A, (leavesTotal + 1,))
    chex.assert_shape(B, (leavesTotal + 1,))
    chex.assert_shape(count, (leavesTotal + 1,))
    chex.assert_shape(gapter, (leavesTotal + 1,))
    chex.assert_shape(gap, (leavesTotal * leavesTotal + 1,))
    
    chex.assert_type([A, B, count, gapter], dtypeDefault)
    chex.assert_type(gap, dtypeMaximum)

    foldingsValues = (A, B, count, gapter, gap, foldingsTotal, l, g)
    foldingsValues = Z0Z_while_loop(while_activeLeaf1ndex_greaterThan_0, countFoldings, foldingsValues)
    return foldingsValues[5]
