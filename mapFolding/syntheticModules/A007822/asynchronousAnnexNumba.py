from mapFolding import Array1DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal
from queue import Queue
from threading import Lock, Thread
import numba
import numpy

listThreads: list[Thread] = []
queueFutures: Queue[Array1DLeavesTotal] = Queue()
groupsOfFoldsTotal: int = 0
groupsOfFoldsTotalLock = Lock()
sentinelStop = object()

def initializeConcurrencyManager(maxWorkers: int, groupsOfFolds: int=0) -> None:
	global listThreads, groupsOfFoldsTotal, queueFutures  # noqa: PLW0603
	listThreads = []
	queueFutures = Queue()
	groupsOfFoldsTotal = groupsOfFolds
	indexThread = 0
	while indexThread < maxWorkers:
		thread = Thread(target=_threadDoesSomething, name=f"thread{indexThread}", daemon=True)
		thread.start()
		listThreads.append(thread)
		indexThread += 1

def _threadDoesSomething() -> None:
	global groupsOfFoldsTotal  # noqa: PLW0603
	while True:
		leafBelow = queueFutures.get()
		if leafBelow is sentinelStop:  # pyright: ignore[reportUnnecessaryComparison]
			break
		symmetricFolds = _filterAsymmetricFolds(leafBelow)
		with groupsOfFoldsTotalLock:
			groupsOfFoldsTotal += symmetricFolds

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

def filterAsymmetricFolds(leafBelow: Array1DLeavesTotal) -> None:
	"""Queue leafBelow for asynchronous processing.
	
	This function is called from the jitted count function in asynchronousNumba.py
	using numba's objmode context manager. It queues the array for processing
	by worker threads, which call the jitted _filterAsymmetricFolds function.
	"""
	queueFutures.put_nowait(leafBelow.copy())

def getSymmetricFoldsTotal() -> DatatypeFoldsTotal:
	global listThreads  # noqa: PLW0602
	for _thread in listThreads:
		queueFutures.put(sentinelStop)  # pyright: ignore[reportArgumentType]
	for thread in listThreads:
		thread.join()
	return groupsOfFoldsTotal
