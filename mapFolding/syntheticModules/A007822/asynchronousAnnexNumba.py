from mapFolding import Array1DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal
from mapFolding.syntheticModules.A007822.leafBelowSender import LeafBelowSender
from threading import Lock
import numba
import numpy

groupsOfFoldsTotal: int = 0
groupsOfFoldsTotalLock = Lock()
leafBelowSender: LeafBelowSender | None = None

def initializeConcurrencyManager(maxWorkers: int, groupsOfFolds: int=0) -> None:
	"""Initialize the concurrent processing system.
	
	Parameters
	----------
	maxWorkers : int
		Number of worker threads (not used in this implementation, 
		but kept for API compatibility)
	groupsOfFolds : int
		Initial value for groupsOfFoldsTotal counter
	"""
	global leafBelowSender, groupsOfFoldsTotal  # noqa: PLW0603
	groupsOfFoldsTotal = groupsOfFolds
	
	# Create the leaf sender with a processor function
	# Buffer size of 10000 should be sufficient for most cases
	# Array size should match the maximum size of leafBelow arrays
	leafBelowSender = LeafBelowSender(
		buffer_size=10000,
		array_size=100,  # Adjust based on expected maximum leavesTotal
		processor_func=_processLeafBelow
	)

def _processLeafBelow(leafBelow: Array1DLeavesTotal) -> None:
	"""Process a leafBelow array by filtering asymmetric folds.
	
	This function is called by the LeafBelowSender's background thread.
	"""
	global groupsOfFoldsTotal  # noqa: PLW0603
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
	"""Push a leafBelow array for asynchronous processing.
	
	This function is designed to be called from numba-jitted code.
	It uses the LeafBelowSender to pass the array to a background thread.
	
	Parameters
	----------
	leafBelow : Array1DLeavesTotal
		The leaf array to process
	"""
	if leafBelowSender is not None:
		leafBelowSender.push(leafBelow)

def getLeafBelowSender():
	"""Get the LeafBelowSender instance for passing to numba-jitted code.
	
	Returns
	-------
	LeafBelowSender
		The sender instance that can be used from numba
	"""
	return leafBelowSender

def getSymmetricFoldsTotal() -> DatatypeFoldsTotal:
	"""Finalize processing and return the total count of symmetric folds.
	
	This function waits for all pending arrays to be processed and
	returns the final count.
	
	Returns
	-------
	DatatypeFoldsTotal
		Total number of symmetric folds found
	"""
	global leafBelowSender  # noqa: PLW0602
	if leafBelowSender is not None:
		leafBelowSender.close()
	return groupsOfFoldsTotal
