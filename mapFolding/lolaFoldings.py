from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import numpy
"""
ALL variables instantiated by `countFoldings` are numpy.NDArray instances.
ALL of those NDArray are indexed by variables defined in `lovelaceIndices.py`.

`doWork` has two `for` loops with a structure of `for identifier in range(p,q)`.
At the moment those two identifiers are primitive integers, rather than embedded in an NDArray instance.

The NDArray:
    Unchanging values
        - the
        - connectionGraph
    Dynamic values that are "personal" to each worker
        - my
        - track
        - potentialGaps

Key concepts
    - A "leaf" is a unit square in the map
    - A "gap" is a potential position where a new leaf can be folded
    - Connections track how leaves can connect above/below each other
    - The algorithm builds foldings incrementally by placing one leaf at a time
    - Backtracking explores all valid combinations
    - Leaves and dimensions are enumerated starting from 1, not 0; hence, leaf1ndex not leafIndex

Algorithm flow
    For each leaf
        - Find valid gaps in each dimension
        - Place leaf in valid position
            - Try to find another lead to put in the adjacent position
            - Repeat until the map is completely folded
        - Backtrack when no valid positions remain
"""
# Indices of array `the`, which holds unchanging, small, unsigned, integer values.
from mapFolding.lovelaceIndices import leafBelow
from mapFolding.lolaIndices import leavesTotal, dimensionsTotal, dimensionsPlus1, COUNTindicesStatic
from mapFolding.lolaIndices import COUNTindicesDynamic, gap1ndexLowerBound

def foldings(listDimensions: List[int]):
    the = numpy.zeros(COUNTindicesStatic, dtype=numpy.int64)

    from mapFolding.beDRY import validateParametersFoldings
    listDimensions, the[leavesTotal], connectionGraph = validateParametersFoldings(listDimensions)

    the[dimensionsTotal] = len(listDimensions)
    the[dimensionsPlus1] = the[dimensionsTotal] + 1

    track = numpy.zeros((4, the[leavesTotal] + 1), dtype=numpy.int64)
    potentialGaps = numpy.zeros(the[leavesTotal] * the[leavesTotal] + 1, dtype=numpy.int64)
    my = numpy.zeros(COUNTindicesDynamic, dtype=numpy.int64)

    from mapFolding.lolaGenerateStates import generateStates, TaskState
    dictionaryStates = generateStates(track, potentialGaps, my, the, connectionGraph)

    from mapFolding.lolaCountSubtotal import countSubtotal
    foldingsTotal = 0
    with ProcessPoolExecutor() as concurrencyManager:
        dictionaryConcurrency = {}
        for taskIndex, taskState in dictionaryStates.items():
            print(f"{taskIndex=}, {taskState['my'][gap1ndexLowerBound]=}")
            # print(f"{taskIndex=}, {taskState['my']=}, {track[leafBelow][0]=}")
            claimTicket = concurrencyManager.submit(
            countSubtotal, 
            taskState['track'], 
            taskState['potentialGaps'], 
            taskState['my'], 
            the, 
            connectionGraph
            )
            dictionaryConcurrency[claimTicket] = taskIndex

        for claimTicket in as_completed(dictionaryConcurrency):
            foldingsSubtotal = claimTicket.result()
            print(f"{dictionaryConcurrency[claimTicket]=}, {foldingsSubtotal=}")
            foldingsTotal += foldingsSubtotal

    return foldingsTotal

