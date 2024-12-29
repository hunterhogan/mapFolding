"""
Plan to refactor in JAX
- change the control flow to JAX style without using JAX
- Identifiers and scope:
    - I should have started from the outside and worked my way in, but the innermost function is more-or-less done.
        Despite that one function (which I will need to continuously update as I refactor),
        identifiers for variables must signal the scope from which they are drawing their value.
    - Due to
        - high-risk of bug: using a value from the wrong scope
        - immutable values in JAX
        - state passing in JAX
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
# import jax
# import jaxtyping
import numpy as NUMERICALPYTHON

dtypeMaximumNUMERICALPYTHON = NUMERICALPYTHON.uint32
dtypeDefaultNUMERICALPYTHON = NUMERICALPYTHON.uint8

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
    del listDimensionsPositive

    # Dynamic values
    A = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    B = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    count = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)
    gapter = NUMERICALPYTHON.zeros(leavesTotal + 1, dtype=dtypeDefaultNUMERICALPYTHON)

    gap = NUMERICALPYTHON.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximumNUMERICALPYTHON)

    foldingsTotal: int = 0
    l: int = 1
    g: int = 0

    foldingsValues = (A, B, count, gapter, gap, foldingsTotal, l, g)

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
        """Track possible gaps for activeLeaf1ndex in each section"""
        gap1ndexLowerBound: int = int(gapRangeStart[activeLeaf1ndex - 1])
        """Reset gap index"""
        activeGap1ndex = gap1ndexLowerBound

        """Count possible gaps for activeLeaf1ndex in each section"""
        for dimension1ndex in range(1, dimensionsTotal + 1):
            if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                unconstrainedLeaf += 1
            else:
                leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]

                def while_leaf1ndexConnectee_notEquals_activeLeaf1ndex(while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue):
                    comparand = while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue[3]
                    return comparand != activeLeaf1ndex

                def while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo(while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue):
                    while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_potentialGaps, while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_gap1ndexLowerBound, while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_countDimensionsGapped, while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_leaf1ndexConnectee = while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue
                    while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_potentialGaps[while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_gap1ndexLowerBound] = while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_leaf1ndexConnectee
                    if while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_countDimensionsGapped[while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_leaf1ndexConnectee] == 0:
                        while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_gap1ndexLowerBound += 1
                    while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_countDimensionsGapped[while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_leaf1ndexConnectee] += 1
                    while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][leafBelow[while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_leaf1ndexConnectee]]
                    while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue = (while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_potentialGaps, while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_gap1ndexLowerBound, while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_countDimensionsGapped, while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo_leaf1ndexConnectee)
                    return while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue

                while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue = (potentialGaps, gap1ndexLowerBound, countDimensionsGapped, leaf1ndexConnectee)
                while_leaf1ndexConnectee_notEquals_activeLeaf1ndexComparison = while_leaf1ndexConnectee_notEquals_activeLeaf1ndex(while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue)

                while while_leaf1ndexConnectee_notEquals_activeLeaf1ndexComparison:
                    while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue = while_leaf1ndexConnectee_notEquals_activeLeaf1ndexDo(while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue)
                    while_leaf1ndexConnectee_notEquals_activeLeaf1ndexComparison = while_leaf1ndexConnectee_notEquals_activeLeaf1ndex(while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue)

                potentialGaps, gap1ndexLowerBound, countDimensionsGapped, leaf1ndexConnectee = while_leaf1ndexConnectee_notEquals_activeLeaf1ndexValue

        """If activeLeaf1ndex is unconstrained in all sections, it can be inserted anywhere"""
        if unconstrainedLeaf == dimensionsTotal:
            for leaf1ndex in range(activeLeaf1ndex):
                potentialGaps[gap1ndexLowerBound] = leaf1ndex
                gap1ndexLowerBound += 1

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

        whileBacktrackingValues = (leafAbove, leafBelow, activeLeaf1ndex)

        def whileBacktrackingCondition(whileBacktrackingValues):
            activeLeafNumber = whileBacktrackingValues[2]
            return NUMERICALPYTHON.logical_and(activeLeafNumber > 0, activeGap1ndex == gapRangeStart[activeLeafNumber - 1])

        def whileBacktrackingDo(whileBacktrackingValues):
            backtrackAbove: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = whileBacktrackingValues[0]
            backtrackBelow: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = whileBacktrackingValues[1]
            activeLeafNumber: int = whileBacktrackingValues[2]

            activeLeafNumber -= 1
            backtrackBelow[backtrackAbove[activeLeafNumber]] = backtrackBelow[activeLeafNumber]
            backtrackAbove[backtrackBelow[activeLeafNumber]] = backtrackAbove[activeLeafNumber]

            whileBacktrackingValues = (backtrackAbove, backtrackBelow, activeLeafNumber)
            return whileBacktrackingValues

        whileBacktrackingValues = Z0Z_while_loop(whileBacktrackingCondition, whileBacktrackingDo, whileBacktrackingValues)

        leafAbove, leafBelow, activeLeaf1ndex = whileBacktrackingValues

        def if_activeLeaf1ndex_greaterThan_0(activeLeafNumber):
            return activeLeafNumber > 0

        if_activeLeaf1ndex_greaterThan_0_values = (leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex)

        """Place leaf in valid position"""
        def if_activeLeaf1ndex_greaterThan_0_do(if_activeLeaf1ndex_greaterThan_0_values):
            placeLeafAbove, placeLeafBelow, placeGapRangeStart, activeLeafNumber, activeGapNumber = if_activeLeaf1ndex_greaterThan_0_values
            activeGapNumber -= 1
            placeLeafAbove[activeLeafNumber] = potentialGaps[activeGapNumber]
            placeLeafBelow[activeLeafNumber] = placeLeafBelow[placeLeafAbove[activeLeafNumber]]
            placeLeafBelow[placeLeafAbove[activeLeafNumber]] = activeLeafNumber
            placeLeafAbove[placeLeafBelow[activeLeafNumber]] = activeLeafNumber
            """Save current gap index"""
            placeGapRangeStart[activeLeafNumber] = activeGapNumber
            """Move to next leaf"""
            activeLeafNumber += 1
            return (placeLeafAbove, placeLeafBelow, placeGapRangeStart, activeLeafNumber, activeGapNumber)

        if_activeLeaf1ndex_greaterThan_0_values = Z0Z_cond(if_activeLeaf1ndex_greaterThan_0(activeLeaf1ndex),
                                                        if_activeLeaf1ndex_greaterThan_0_do,
                                                        doNothing,
                                                        if_activeLeaf1ndex_greaterThan_0_values)

        leafAbove, leafBelow, gapRangeStart, activeLeaf1ndex, activeGap1ndex = if_activeLeaf1ndex_greaterThan_0_values

        allValues = (leafAbove, leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, foldingsSubTotal, activeLeaf1ndex, activeGap1ndex)

        return allValues

    foldingsValues = Z0Z_while_loop(while_activeLeaf1ndex_greaterThan_0, countFoldings, foldingsValues)

    foldingsTotal = foldingsValues[5]
    return foldingsTotal
