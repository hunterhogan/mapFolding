from mapFolding import Array1DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal
from numba import types
from numba.core import cgutils
from numba.core.pythonapi import make_arg_tuple, unpack_tuple
from numba.extending import overload_method, typeof_impl
from queue import Queue
from threading import Lock, Thread
import numba
import numpy

listThreads: list[Thread] = []
queueFutures: Queue[Array1DLeavesTotal] = Queue()
groupsOfFoldsTotal: int = 0
groupsOfFoldsTotalLock = Lock()
sentinelStop = object()

class FoldQueueManager:
	"""Manager for passing leafBelow arrays from jitted code to Python for async processing.
	
	Similar to numba-progress ProgressBar, this class can be passed to jitted functions
	and its push() method can be called from within jitted code.
	"""
	def __init__(self, queue: Queue[Array1DLeavesTotal]) -> None:
		self.queue = queue
	
	def push(self, leafBelow: Array1DLeavesTotal) -> None:
		"""Queue leafBelow array for asynchronous processing."""
		self.queue.put_nowait(leafBelow.copy())

class FoldQueueManagerType(types.Type):
	"""Numba type for FoldQueueManager."""
	def __init__(self) -> None:
		self.name = "FoldQueueManagerType"
		super(FoldQueueManagerType, self).__init__(name=self.name)

fold_queue_manager_type = FoldQueueManagerType()

@typeof_impl.register(FoldQueueManager)
def typeof_fold_queue_manager(val, c):
	"""Register FoldQueueManager with numba's type system."""
	return fold_queue_manager_type

@numba.extending.register_model(FoldQueueManagerType)
class FoldQueueManagerModel(numba.core.datamodel.models.OpaqueModel):
	"""Model for FoldQueueManager - treat as opaque reference."""
	pass

@numba.extending.unbox(FoldQueueManagerType)
def unbox_fold_queue_manager(typ, obj, c):
	"""Convert Python FoldQueueManager to numba representation."""
	return numba.core.pythonapi.NativeValue(obj)

@numba.extending.box(FoldQueueManagerType)
def box_fold_queue_manager(typ, val, c):
	"""Convert numba FoldQueueManager back to Python."""
	return val

@overload_method(FoldQueueManagerType, "push")
def fold_queue_manager_push(manager, leafBelow):
	"""Overload the push method to be callable from jitted code."""
	def push_impl(manager, leafBelow):
		with numba.objmode():
			manager.push(leafBelow)
	return push_impl

def initializeConcurrencyManager(maxWorkers: int, groupsOfFolds: int=0) -> FoldQueueManager:
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
	return FoldQueueManager(queueFutures)

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
	queueFutures.put_nowait(leafBelow.copy())

def getSymmetricFoldsTotal() -> DatatypeFoldsTotal:
	global listThreads  # noqa: PLW0602
	for _thread in listThreads:
		queueFutures.put(sentinelStop)  # pyright: ignore[reportArgumentType]
	for thread in listThreads:
		thread.join()
	return groupsOfFoldsTotal
