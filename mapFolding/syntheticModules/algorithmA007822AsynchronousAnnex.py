from concurrent.futures import Future as ConcurrentFuture, ProcessPoolExecutor
from copy import deepcopy
from mapFolding.dataBaskets import MapFoldingState
from queue import Empty, Queue
from threading import Thread

concurrencyManager = None
queueFutures: Queue[ConcurrentFuture[MapFoldingState]] = Queue()
groupsOfFoldsTotal: int = 0
processingThread = None

def _processCompletedFutures() -> None:
    global queueFutures, groupsOfFoldsTotal
    while True:
        try:
            claimTicket = queueFutures.get(timeout=1)
            if claimTicket is None:
                break
            state: MapFoldingState = claimTicket.result()
            groupsOfFoldsTotal += state.groupsOfFolds
        except Empty:
            continue

def initializeConcurrencyManager(maxWorkers: int | None=None) -> None:
    global concurrencyManager, queueFutures, groupsOfFoldsTotal, processingThread
    concurrencyManager = ProcessPoolExecutor(max_workers=maxWorkers)
    queueFutures = Queue()
    groupsOfFoldsTotal = 0
    processingThread = Thread(target=_processCompletedFutures)
    processingThread.start()

def _filterAsymmetricFolds(state: MapFoldingState) -> MapFoldingState:
    state.indexLeaf = 0
    leafConnectee = 0
    while leafConnectee < state.leavesTotal + 1:
        leafNumber = int(state.leafBelow[state.indexLeaf])
        state.leafComparison[leafConnectee] = (leafNumber - state.indexLeaf + state.leavesTotal) % state.leavesTotal
        state.indexLeaf = leafNumber
        leafConnectee += 1
    indexInMiddle = state.leavesTotal // 2
    state.indexMiniGap = 0
    while state.indexMiniGap < state.leavesTotal + 1:
        ImaSymmetricFold = True
        leafConnectee = 0
        while leafConnectee < indexInMiddle:
            if state.leafComparison[(state.indexMiniGap + leafConnectee) % (state.leavesTotal + 1)] != state.leafComparison[(state.indexMiniGap + state.leavesTotal - 1 - leafConnectee) % (state.leavesTotal + 1)]:
                ImaSymmetricFold = False
                break
            leafConnectee += 1
        if ImaSymmetricFold:
            state.groupsOfFolds += 1
        state.indexMiniGap += 1
    return state

def filterAsymmetricFolds(state: MapFoldingState) -> None:
    global concurrencyManager, queueFutures
    queueFutures.put(concurrencyManager.submit(_filterAsymmetricFolds, deepcopy(state)))

def getAsymmetricFoldsTotal() -> int:
    global concurrencyManager, queueFutures, processingThread
    concurrencyManager.shutdown(wait=True)
    queueFutures.put(None)
    processingThread.join()
    return groupsOfFoldsTotal
