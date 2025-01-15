from mapFolding import validateListDimensions, getLeavesTotal
from typing import List, Tuple
import jax
import jaxtyping

dtypeDefault = jax.numpy.int32
dtypeMaximum = jax.numpy.int32

def countFolds(listDimensions: List[int]):
    """Calculate foldings across multiple devices using pmap"""
    p = validateListDimensions(listDimensions)
    n = getLeavesTotal(p)

    # Get number of devices (GPUs/TPUs)
    deviceCount = jax.device_count()

    if deviceCount > 1:
        # Split work across devices
        tasksPerDevice = (n + deviceCount - 1) // deviceCount
        paddedTaskCount = tasksPerDevice * deviceCount

        # Create padded array of task indices
        arrayTaskIndices = jax.numpy.arange(paddedTaskCount, dtype=dtypeDefault)
        arrayTaskIndices = arrayTaskIndices.reshape((deviceCount, tasksPerDevice))

        # Create pmapped function
        parallelFoldingsTask = jax.pmap(lambda x: jax.vmap(lambda y: foldingsTask(tuple(p), y))(x))

        # Run computation across devices
        arrayResults = parallelFoldingsTask(arrayTaskIndices)

        # Sum valid results (ignore padding)
        return jax.numpy.sum(arrayResults[:, :min(tasksPerDevice, n - tasksPerDevice * (deviceCount-1))])
    else:
        # Fall back to sequential execution if no multiple devices available
        arrayTaskIndices = jax.numpy.arange(n, dtype=dtypeDefault)
        batchedFoldingsTask = jax.vmap(lambda x: foldingsTask(tuple(p), x))
        return jax.numpy.sum(batchedFoldingsTask(arrayTaskIndices))

def foldingsTask(p, taskIndex) -> jaxtyping.UInt32:
    arrayDimensions = jax.numpy.asarray(p, dtype=dtypeDefault)
    leavesTotal = jax.numpy.prod(arrayDimensions)
    dimensionsTotal = jax.numpy.size(arrayDimensions)

    """How to build a leaf connection graph, also called a "Cartesian Product Decomposition"
    or a "Dimensional Product Mapping", with sentinels:
    Step 1: find the cumulative product of the map's dimensions"""
    cumulativeProduct = jax.numpy.ones(dimensionsTotal + 1, dtype=dtypeDefault)
    cumulativeProduct = cumulativeProduct.at[1:].set(jax.numpy.cumprod(arrayDimensions))

    """Step 2: for each dimension, create a coordinate system """
    """coordinateSystem[dimension1ndex][leaf1ndex] holds the dimension1ndex-th coordinate of leaf leaf1ndex"""
    coordinateSystem = jax.numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=dtypeDefault)

    # Create mesh of indices for vectorized computation
    dimension1ndices, leaf1ndices = jax.numpy.meshgrid(
        jax.numpy.arange(1, dimensionsTotal + 1),
        jax.numpy.arange(1, leavesTotal + 1),
        indexing='ij'
    )

    # Compute all coordinates at once using broadcasting
    coordinateSystem = coordinateSystem.at[1:, 1:].set(
        ((leaf1ndices - 1) // cumulativeProduct.at[dimension1ndices - 1].get()) %
        arrayDimensions.at[dimension1ndices - 1].get() + 1
    )
    del dimension1ndices, leaf1ndices

    """Step 3: create a huge empty connection graph"""
    connectionGraph = jax.numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=dtypeDefault)

    # Create 3D mesh of indices for vectorized computation
    dimension1ndices, activeLeaf1ndices, connectee1ndices = jax.numpy.meshgrid(
        jax.numpy.arange(1, dimensionsTotal + 1),
        jax.numpy.arange(1, leavesTotal + 1),
        jax.numpy.arange(1, leavesTotal + 1),
        indexing='ij'
    )

    # Create masks for valid indices
    maskActiveConnectee = connectee1ndices <= activeLeaf1ndices

    # Calculate coordinate parity comparison
    coordsParity = (coordinateSystem.at[dimension1ndices, activeLeaf1ndices].get() & 1) == \
                    (coordinateSystem.at[dimension1ndices, connectee1ndices].get() & 1)

    # Compute distance conditions
    isFirstCoord = coordinateSystem.at[dimension1ndices, connectee1ndices].get() == 1
    isLastCoord = coordinateSystem.at[dimension1ndices, connectee1ndices].get() == \
                    arrayDimensions.at[dimension1ndices - 1].get()
    exceedsActive = connectee1ndices + cumulativeProduct.at[dimension1ndices - 1].get() > activeLeaf1ndices

    # Compute connection values for even and odd parities
    evenParityValues = jax.numpy.where(
        isFirstCoord,
        connectee1ndices,
        connectee1ndices - cumulativeProduct.at[dimension1ndices - 1].get()
    )

    oddParityValues = jax.numpy.where(
        jax.numpy.logical_or(isLastCoord, exceedsActive),
        connectee1ndices,
        connectee1ndices + cumulativeProduct.at[dimension1ndices - 1].get()
    )

    # Combine based on parity and valid indices
    connectionValues = jax.numpy.where(
        coordsParity,
        evenParityValues,
        oddParityValues
    )

    # Update only valid connections
    connectionGraph = connectionGraph.at[dimension1ndices, activeLeaf1ndices, connectee1ndices].set(
        jax.numpy.where(maskActiveConnectee, connectionValues, 0)
    )

    def doNothing(argument):
        return argument

    def while_activeLeaf1ndex_greaterThan_0(comparisonValues: Tuple):
        comparand = comparisonValues[6]
        return comparand > 0

    def countFoldings(allValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32, jaxtyping.UInt32]):
        _0, leafBelow, _2, _3, _4, _5, activeLeaf1ndex, _7 = allValues

        sentinel = leafBelow.at[0].get().astype(jax.numpy.int32)

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
                    # if taskDivisions == False or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                    def taskDivisionComparison():
                        return jax.numpy.logical_or(activeLeaf1ndex != leavesTotal, jax.numpy.equal(countGapsLeaf1ndexConnectee % leavesTotal, taskIndex))
                        # return taskDivisions == False or jax.numpy.logical_or(activeLeaf1ndex != leavesTotal, jax.numpy.equal(countGapsLeaf1ndexConnectee % leavesTotal, taskIndex))

                    def taskDivisionDo(taskDivisionDoValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32]):
                        taskDivisionCountDimensionsGapped, taskDivisionPotentialGaps, taskDivisionGap1ndexLowerBound = taskDivisionDoValues

                        taskDivisionPotentialGaps = taskDivisionPotentialGaps.at[taskDivisionGap1ndexLowerBound].set(countGapsLeaf1ndexConnectee)
                        taskDivisionGap1ndexLowerBound = jax.numpy.where(
                            jax.numpy.equal(taskDivisionCountDimensionsGapped.at[countGapsLeaf1ndexConnectee].get(), 0), taskDivisionGap1ndexLowerBound + 1, taskDivisionGap1ndexLowerBound)
                        taskDivisionCountDimensionsGapped = taskDivisionCountDimensionsGapped.at[countGapsLeaf1ndexConnectee].add(1)

                        return (taskDivisionCountDimensionsGapped, taskDivisionPotentialGaps, taskDivisionGap1ndexLowerBound)

                    countGapsLeaf1ndexConnectee = countGapsDoValues[3]
                    taskDivisionValues = (countGapsDoValues[0], countGapsDoValues[1], countGapsDoValues[2])
                    taskDivisionValues = jax.lax.cond(taskDivisionComparison(), taskDivisionDo, doNothing, taskDivisionValues)

                    countGapsLeaf1ndexConnectee = connectionGraph.at[dimensionNumber, activeLeaf1ndex, leafBelow.at[countGapsLeaf1ndexConnectee].get()].get().astype(jax.numpy.int32)

                    return (taskDivisionValues[0], taskDivisionValues[1], taskDivisionValues[2], countGapsLeaf1ndexConnectee)

                unconstrained_countDimensionsGapped, unconstrained_gapsWhere, unconstrained_gap1ndexCeiling, unconstrained_unconstrainedLeaf = unconstrainedValues

                leaf1ndexConnectee = connectionGraph.at[dimensionNumber, activeLeaf1ndex, activeLeaf1ndex].get().astype(jax.numpy.int32)

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

        def for_range_from_activeGap1ndex_to_gap1ndexCeiling(comparisonValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
            return comparisonValues[-1] < gap1ndexCeiling

        def miniGapDo(gapToGapValues: Tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.UInt32, jaxtyping.UInt32]):
            gapToGapCountDimensionsGapped, gapToGapPotentialGaps, activeGapNumber, index = gapToGapValues
            gapToGapPotentialGaps = gapToGapPotentialGaps.at[activeGapNumber].set(gapToGapPotentialGaps.at[index].get())
            activeGapNumber = jax.numpy.where(jax.numpy.equal(gapToGapCountDimensionsGapped.at[gapToGapPotentialGaps.at[index].get()].get(), dimensionsTotal - unconstrainedLeaf), activeGapNumber + 1, activeGapNumber).astype(jax.numpy.int32)
            gapToGapCountDimensionsGapped = gapToGapCountDimensionsGapped.at[gapToGapPotentialGaps.at[index].get()].set(0)
            index = 1 + index
            return (gapToGapCountDimensionsGapped, gapToGapPotentialGaps, activeGapNumber, index)

        _0, leafBelow, countDimensionsGapped, gapRangeStart, gapsWhere, _5, activeLeaf1ndex, activeGap1ndex = allValues

        unconstrainedLeaf = jax.numpy.int32(0)
        dimension1ndex = jax.numpy.int32(1)
        gap1ndexCeiling = gapRangeStart.at[activeLeaf1ndex - 1].get().astype(jax.numpy.int32)
        activeGap1ndex = gap1ndexCeiling
        for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = (countDimensionsGapped, gapsWhere, gap1ndexCeiling, unconstrainedLeaf, dimension1ndex)
        for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values = jax.lax.while_loop(for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1_do, for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values)
        countDimensionsGapped, gapsWhere, gap1ndexCeiling, unconstrainedLeaf, dimension1ndex = for_dimension1ndex_in_range_1_to_dimensionsTotalPlus1Values
        del dimension1ndex

        leaf1ndex = jax.numpy.int32(0)
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
    A = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    B = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    count = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    gapter = jax.numpy.zeros(leavesTotal + 1, dtype=dtypeDefault)
    gap = jax.numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)

    foldingsSubTotal = jax.numpy.int32(0)
    l = jax.numpy.int32(1)
    g = jax.numpy.int32(0)

    foldingsValues = (A, B, count, gapter, gap, foldingsSubTotal, l, g)
    foldingsValues = jax.lax.while_loop(while_activeLeaf1ndex_greaterThan_0, countFoldings, foldingsValues)
    return foldingsValues[5]
