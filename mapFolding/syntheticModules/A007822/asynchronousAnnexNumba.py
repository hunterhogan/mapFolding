from hunterMakesPy import raiseIfNone
from mapFolding import Array1DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal
from os import cpu_count
from queue import Full, Queue
from threading import Lock, Thread
import numba
import numpy

# Thread based asynchronous filtering to avoid large memory growth and ProcessPool overhead.
workersFiltering: list[Thread] | None = None
queueGroupsToFilter: Queue[Array1DLeavesTotal] = Queue()  # unbounded by default; see put logic for mitigation
groupsOfFoldsTotal: int = 0
groupsOfFoldsTotalLock = Lock()
sentinelStop = object()

def initializeConcurrencyManager(maxWorkers: int | None=None, groupsOfFolds: int=0) -> None:
    global workersFiltering, groupsOfFoldsTotal, queueGroupsToFilter  # noqa: PLW0603
    if maxWorkers is None or maxWorkers < 1:
        coresAvailable = cpu_count() or 1
        # Reserve one core for the generating thread; use at least one worker.
        maxWorkers = max(1, coresAvailable - 1)
    workersFiltering = []
    queueGroupsToFilter = Queue()  # reinitialize to clear prior state
    groupsOfFoldsTotal = groupsOfFolds
    indexWorker = 0
    while indexWorker < maxWorkers:
        worker = Thread(target=_workerFilterGroups, name=f"filterWorker{indexWorker}", daemon=True)
        worker.start()
        workersFiltering.append(worker)
        indexWorker += 1

def _workerFilterGroups() -> None:  # runs forever until sentinel received
    global groupsOfFoldsTotal
    while True:
        groupCandidate = queueGroupsToFilter.get()
        if groupCandidate is sentinelStop:  # pyright: ignore[reportUnnecessaryComparison]
            break
        # Compute filtered groups using compiled function
        groupsFound = _filterAsymmetricFolds(groupCandidate)
        with groupsOfFoldsTotalLock:
            groupsOfFoldsTotal += groupsFound

def _enqueueLeafBelow(leafBelow: Array1DLeavesTotal) -> None:
    # Attempt non-blocking enqueue; if queue full (if a maxsize later is applied) process synchronously.
    try:
        queueGroupsToFilter.put_nowait(leafBelow.copy())
    except Full:
        # Fallback: process immediately to avoid blocking and unbounded growth.
        groupsFound = _filterAsymmetricFolds(leafBelow)
        with groupsOfFoldsTotalLock:
            global groupsOfFoldsTotal
            groupsOfFoldsTotal += groupsFound

@numba.jit(cache=True, error_model='numpy', fastmath=True)
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
        groupsOfFolds += ImaSymmetricFold
        indexDistance += 1
    return groupsOfFolds

# @numba.jit(numba.int64(numba.int64[:]), cache=True, error_model='numpy', fastmath=True, forceobj=True)
def filterAsymmetricFolds(leafBelow: Array1DLeavesTotal) -> int:  # non-blocking submission
    # Must not block caller; enqueue for background processing
    _enqueueLeafBelow(leafBelow)
    return 60  # sentinel ignored by caller

# @numba.jit(numba.uint64(), cache=True, error_model='numpy', fastmath=True, forceobj=True)
def getAsymmetricFoldsTotal() -> DatatypeFoldsTotal:
    global workersFiltering  # noqa: PLW0602
    listWorkers = raiseIfNone(workersFiltering)
    # Signal all workers to stop after queue drained
    for _ in listWorkers:
        queueGroupsToFilter.put(sentinelStop)  # pyright: ignore[reportArgumentType]
    for worker in listWorkers:
        worker.join()
    with groupsOfFoldsTotalLock:
        return groupsOfFoldsTotal
