from __future__ import annotations

from copy import deepcopy
from mapFolding.dataBaskets import StateMapFoldingSymmetric
from mapFolding.theTypes import 形TotalFolds
from queue import Queue
from threading import Lock, Thread

boxOfThreads: list[Thread] = []
queueFutures: Queue[StateMapFoldingSymmetric] = Queue()
symmetricTotalFolds: int = 0
LOCKsymmetricTotalFolds = Lock()
STOPsignal = object()

def initializeConcurrencyManager(maxWorkers: int, symmetricFolds: int=0) -> None:
    """Initialize the shared queue, accumulated count, and background workers.

    (AI generated docstring)

    You can use this function in a generated module to reset shared worker state and start
    `maxWorkers` daemon threads. Each thread consumes the shared queue and accumulates filter
    results. The source template contains a placeholder filter; generation supplies the real filter
    [1].

    Parameters
    ----------
    maxWorkers : int
    Number of threads to start; a nonpositive value starts no threads.
    symmetricFolds : int = 0
    Initial value for the shared accumulated symmetric count.

    Examples
    --------
    The module `mapFolding.synthesized.mapFoldingSymmetric.asynchronous` uses this call [2].

    ```python
    initializeConcurrencyManager(maxWorkers)
    ```

    References
    ----------
    [1] `mapFolding.kitAST.mapFoldingSymmetric.makeModulesAsynchronous.addSymmetryCheckAsynchronous`

    [2] `mapFolding.synthesized.mapFoldingSymmetric.asynchronous`
    """
    global boxOfThreads, symmetricTotalFolds, queueFutures
    boxOfThreads = []
    queueFutures = Queue()
    symmetricTotalFolds = symmetricFolds
    indexThread = 0
    while indexThread < maxWorkers:
        thread = Thread(target=_threadDoesSomething, name=f'thread{indexThread}', daemon=True)
        thread.start()
        boxOfThreads.append(thread)
        indexThread += 1

def _threadDoesSomething() -> None:
    """I use this worker target to consume queued states and accumulate filtered counts.

    (AI generated docstring)

    This function waits for queued state, applies the generated filter, and adds the resulting
    symmetric count while holding the shared lock. The function exits after receiving the shared
    stop object.
    """
    global symmetricTotalFolds
    while True:
        state: StateMapFoldingSymmetric = queueFutures.get()
        if state is STOPsignal:
            break
        state = _filterAsymmetricFolds(state)
        with LOCKsymmetricTotalFolds:
            symmetricTotalFolds += state.symmetricFolds

def filterAsymmetricFolds(state: StateMapFoldingSymmetric) -> None:
    """Queue a copied state for background symmetry filtering.

    (AI generated docstring)

    You can use this function to submit a snapshot of `state` without waiting for filtering.
    The function copies `state` with `deepcopy` [1], enqueues the copy, and returns `None`.

    Parameters
    ----------
    state : StateMapFoldingSymmetric
    Current folding state to copy and submit for filtering.

    Examples
    --------
    The module `mapFolding.synthesized.mapFoldingSymmetric.algorithm` uses this call [2].

    ```python
    filterAsymmetricFolds(state)
    ```

    References
    ----------
    [1] Python recursive object copying.
    https://docs.python.org/3/library/copy.html#copy.deepcopy
    [2] `mapFolding.synthesized.mapFoldingSymmetric.algorithm`
    """
    queueFutures.put_nowait(deepcopy(state))

def getSymmetricTotalFolds() -> 形TotalFolds:
    """Stop initialized workers after queued work and return the accumulated symmetric count.

    (AI generated docstring)

    You can use this function to append one stop object per worker, join every worker, and obtain
    the shared symmetric total after the workers exit.

    Returns
    -------
    symmetricTotalFolds : 形TotalFolds
    Shared accumulated count after all initialized worker threads have exited.

    Examples
    --------
    The module `mapFolding.synthesized.mapFoldingSymmetric.asynchronous` uses this call [1].

    ```python
    getSymmetricTotalFolds()
    ```

    References
    ----------
    [1] `mapFolding.synthesized.mapFoldingSymmetric.asynchronous`
    """
    for _thread in boxOfThreads:
        queueFutures.put(STOPsignal)
    for thread in boxOfThreads:
        thread.join()
    return symmetricTotalFolds

def _filterAsymmetricFolds(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Leaf = 1
    state.leafComparison[0] = 1
    state.leafConnectee = 1
    while state.leafConnectee <= state.totalLeaves:
        state.次MiniGap = state.leafBelow[state.次Leaf]
        state.leafComparison[state.leafConnectee] = (state.totalLeaves + state.次MiniGap - state.次Leaf) % state.totalLeaves
        state.次Leaf = state.次MiniGap
        state.leafConnectee += 1
    for arrayOfTuples in state.indices:
        state.leafConnectee = 1
        for 次Left, 次Right in arrayOfTuples:
            if state.leafComparison[次Left] != state.leafComparison[次Right]:
                state.leafConnectee = 0
                break
        state.symmetricFolds += state.leafConnectee
    return state

def activeLeafGreaterThan0(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0

def activeLeafGreaterThanTotalLeaves(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > state.totalLeaves

def activeLeafIsTheFirstLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex <= 1

def activeLeafIsUnconstrainedInAllDimensions(state: StateMapFoldingSymmetric) -> bool:
    return not state.dimensionsUnconstrained

def activeLeafUnconstrainedInThisDimension(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.dimensionsUnconstrained -= 1
    return state

def filterCommonGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.次MiniGap]
    if state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] == state.dimensionsUnconstrained:
        state = incrementActiveGap(state)
    state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] = 0
    return state

def gapAvailable(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0

def incrementActiveGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndex += 1
    return state

def incrementGap1ndexCeiling(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndexCeiling += 1
    return state

def incrementIndexMiniGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次MiniGap += 1
    return state

def initializeIndexMiniGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次MiniGap = state.gap1ndex
    return state

def initializeVariablesToFindGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.dimensionsUnconstrained = state.totalDimensions
    state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
    state.次Dimension = 0
    return state

def insertActiveLeaf(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Leaf = 0
    while state.次Leaf < state.leaf1ndex:
        state.gapsWhere[state.gap1ndexCeiling] = state.次Leaf
        state.gap1ndexCeiling += 1
        state.次Leaf += 1
    return state

def insertActiveLeafAtGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndex -= 1
    state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
    state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
    state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
    state.leaf1ndex += 1
    return state

def leafBelowSentinelIs1(state: StateMapFoldingSymmetric) -> bool:
    return state.leafBelow[0] == 1

def leafConnecteeIsActiveLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leafConnectee == state.leaf1ndex

def lookForGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
    if state.countDimensionsGapped[state.leafConnectee] == 0:
        state = incrementGap1ndexCeiling(state)
    state.countDimensionsGapped[state.leafConnectee] += 1
    return state

def lookupLeafConnecteeInConnectionGraph(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leaf1ndex]
    return state

def loopingLeavesConnectedToActiveLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leafConnectee != state.leaf1ndex

def loopingThroughTheDimensions(state: StateMapFoldingSymmetric) -> bool:
    return state.次Dimension < state.totalDimensions

def loopingToActiveGapCeiling(state: StateMapFoldingSymmetric) -> bool:
    return state.次MiniGap < state.gap1ndexCeiling

def noGapsHere(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]

def tryAnotherLeafConnectee(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
    return state

def tryNextDimension(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Dimension += 1
    return state

def undoLastLeafPlacement(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leaf1ndex -= 1
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
    return state

def count(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    while activeLeafGreaterThan0(state):
        if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
            if activeLeafGreaterThanTotalLeaves(state):
                filterAsymmetricFolds(state)
            else:
                state = initializeVariablesToFindGaps(state)
                while loopingThroughTheDimensions(state):
                    state = lookupLeafConnecteeInConnectionGraph(state)
                    if leafConnecteeIsActiveLeaf(state):
                        state = activeLeafUnconstrainedInThisDimension(state)
                    else:
                        while loopingLeavesConnectedToActiveLeaf(state):
                            state = lookForGaps(state)
                            state = tryAnotherLeafConnectee(state)
                    state = tryNextDimension(state)
                if activeLeafIsUnconstrainedInAllDimensions(state):
                    state = insertActiveLeaf(state)
                state = initializeIndexMiniGap(state)
                while loopingToActiveGapCeiling(state):
                    state = filterCommonGaps(state)
                    state = incrementIndexMiniGap(state)
        while noGapsHere(state):
            state = undoLastLeafPlacement(state)
        if gapAvailable(state):
            state = insertActiveLeafAtGap(state)
    else:
        state.symmetricFolds = getSymmetricTotalFolds()
    state.symmetricFolds = (state.symmetricFolds + 1) // 2
    return state

def doTheNeedful(state: StateMapFoldingSymmetric, maxWorkers: int) -> StateMapFoldingSymmetric:
    initializeConcurrencyManager(maxWorkers)
    state = count(state)
    return state
