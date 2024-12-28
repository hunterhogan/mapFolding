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

    FOLDINGSTOTAL: int = 0
    l: int = 1
    g: int = 0

    countFoldingsValues = (A, B, count, gapter, gap, FOLDINGSTOTAL, l, g)

    def while_activeLeaf1ndex_greaterThan_0(countFoldingsValues):
        comparand = countFoldingsValues[6]
        return comparand > 0

    def countFoldings(countFoldingsValues):
        leafAbove: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = countFoldingsValues[0]
        leafBelow: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = countFoldingsValues[1]
        countDimensionsGapped: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = countFoldingsValues[2]
        gapRangeStart: NUMERICALPYTHON.ndarray[dtypeDefaultNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeDefaultNUMERICALPYTHON]] = countFoldingsValues[3]
        potentialGaps: NUMERICALPYTHON.ndarray[dtypeMaximumNUMERICALPYTHON, NUMERICALPYTHON.dtype[dtypeMaximumNUMERICALPYTHON]] = countFoldingsValues[4]
        foldingsSubTotal: int = countFoldingsValues[5]
        activeLeaf1ndex: int = countFoldingsValues[6]
        activeGap1ndex: int = countFoldingsValues[7]

        if (leafBelow[0] == 1 and not activeLeaf1ndex > leavesTotal) or activeLeaf1ndex <= 1:
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
        elif activeLeaf1ndex > leavesTotal and leafBelow[0] == 1:
            foldingsSubTotal += leavesTotal

        """Recursive backtracking steps"""
        while activeLeaf1ndex > 0 and activeGap1ndex == gapRangeStart[activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            leafBelow[leafAbove[activeLeaf1ndex]] = leafBelow[activeLeaf1ndex]
            leafAbove[leafBelow[activeLeaf1ndex]] = leafAbove[activeLeaf1ndex]

        """Place leaf in valid position"""
        if activeLeaf1ndex > 0:
            activeGap1ndex -= 1
            leafAbove[activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            leafBelow[activeLeaf1ndex] = leafBelow[leafAbove[activeLeaf1ndex]]
            leafBelow[leafAbove[activeLeaf1ndex]] = activeLeaf1ndex
            leafAbove[leafBelow[activeLeaf1ndex]] = activeLeaf1ndex
            """Save current gap index"""
            gapRangeStart[activeLeaf1ndex] = activeGap1ndex
            """Move to next leaf"""
            activeLeaf1ndex += 1

        countFoldingsValues = (leafAbove, leafBelow, countDimensionsGapped, gapRangeStart, potentialGaps, foldingsSubTotal, activeLeaf1ndex, activeGap1ndex)

        return countFoldingsValues

    while_activeLeaf1ndex_greaterThan_0Comparison = while_activeLeaf1ndex_greaterThan_0(countFoldingsValues)

    while while_activeLeaf1ndex_greaterThan_0Comparison:
        countFoldingsValues = countFoldings(countFoldingsValues)
        while_activeLeaf1ndex_greaterThan_0Comparison = while_activeLeaf1ndex_greaterThan_0(countFoldingsValues)

    foldingsTotal = countFoldingsValues[5]
    return foldingsTotal
