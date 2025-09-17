from concurrent.futures import Future as ConcurrentFuture, ProcessPoolExecutor
from hunterMakesPy import raiseIfNone
from mapFolding import Array1DLeavesTotal
from queue import Empty, Queue
from threading import Thread
import numpy


concurrencyManager = None
groupsOfFoldsTotal: int = 0
processingThread = None
queueFutures: Queue[ConcurrentFuture[int]] = Queue()

def initializeConcurrencyManager(maxWorkers: int | None=None, groupsOfFolds: int=0) -> None:
    global concurrencyManager, queueFutures, groupsOfFoldsTotal, processingThread
    concurrencyManager = ProcessPoolExecutor(max_workers=maxWorkers)
    queueFutures = Queue()
    groupsOfFoldsTotal = groupsOfFolds
    processingThread = Thread(target=_processCompletedFutures)
    processingThread.start()

def _processCompletedFutures() -> None:
    global queueFutures, groupsOfFoldsTotal
    while True:
        try:
            claimTicket: ConcurrentFuture[int] = queueFutures.get(timeout=1)
            if claimTicket is None:
                break
            groupsOfFoldsTotal += claimTicket.result()
        except Empty:
            continue

def _filterAsymmetricFolds(leafBelow: Array1DLeavesTotal) -> int:
    groupsOfFolds = 0
    leafComparison: Array1DLeavesTotal = numpy.zeros_like(leafBelow)
    leavesTotal = leafBelow.size - 1
    indexLeaf = 0
    leafConnectee = 0
    while leafConnectee < leavesTotal + 1:
        leafNumber = int(leafBelow[indexLeaf])
        leafComparison[leafConnectee] = (leafNumber - indexLeaf + leavesTotal) % leavesTotal
        indexLeaf = leafNumber
        leafConnectee += 1
    indexInMiddle = leavesTotal // 2
    indexDistance = 0
    while indexDistance < leavesTotal + 1:
        ImaSymmetricFold = True
        leafConnectee = 0
        while leafConnectee < indexInMiddle:
            if leafComparison[(indexDistance + leafConnectee) % (leavesTotal + 1)] != leafComparison[(indexDistance + leavesTotal - 1 - leafConnectee) % (leavesTotal + 1)]:
                ImaSymmetricFold = False
                break
            leafConnectee += 1
        if ImaSymmetricFold:
            groupsOfFolds += 1
        indexDistance += 1
    return groupsOfFolds

def filterAsymmetricFolds(leafBelow: Array1DLeavesTotal) -> None:
    global concurrencyManager, queueFutures
    queueFutures.put(raiseIfNone(concurrencyManager).submit(_filterAsymmetricFolds, leafBelow.copy()))

def getAsymmetricFoldsTotal() -> int:
    global concurrencyManager, queueFutures, processingThread
    raiseIfNone(concurrencyManager).shutdown(wait=True)
    queueFutures.put(None)
    raiseIfNone(processingThread).join()
    return groupsOfFoldsTotal
