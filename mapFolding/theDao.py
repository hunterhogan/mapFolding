from mapFolding import indexMy, indexTrack
from numba import prange
from numpy import dtype, integer, ndarray
from typing import Any

# `.value` is not necessary for this module or most modules. But, this module is transformed into Numba "jitted" functions, and Numba won't use `Enum` for an ndarray index without `.value`.
def activeLeafConnectedToItself(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leafConnectee.value] == my[indexMy.leaf1ndex.value]

def activeLeafGreaterThan0(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] > 0

def activeLeafGreaterThanLeavesTotal(foldGroups: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] > foldGroups[-1]

def activeLeafIsTheFirstLeaf(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] <= 1

def allDimensionsAreUnconstrained(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return not my[indexMy.dimensionsUnconstrained.value]

def backtrack(my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
	my[indexMy.leaf1ndex.value] -= 1
	track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
	track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]

def countGaps(gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
	gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
	if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
		incrementGap1ndexCeiling(my=my)
	track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1

def decrementDimensionsUnconstrained(my: ndarray[tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.dimensionsUnconstrained.value] -= 1

def dimensionsUnconstrainedCondition(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]

def filterCommonGaps(gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
	gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
	if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == my[indexMy.dimensionsUnconstrained.value]:
		incrementActiveGap(my=my)
	track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] = 0

def incrementActiveGap(my: ndarray[tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.gap1ndex.value] += 1

def incrementGap1ndexCeiling(my: ndarray[tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.gap1ndexCeiling.value] += 1

def incrementIndexDimension(my: ndarray[tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.indexDimension.value] += 1

def incrementIndexMiniGap(my: ndarray[tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.indexMiniGap.value] += 1

def initializeIndexMiniGap(my: ndarray[tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]

def initializeLeafConnectee(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]

def initializeVariablesToFindGaps(my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
	my[indexMy.dimensionsUnconstrained.value] = my[indexMy.dimensionsTotal.value]
	my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
	my[indexMy.indexDimension.value] = 0

def insertUnconstrainedLeaf(gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.indexLeaf.value] = 0
	while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
		gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.indexLeaf.value]
		my[indexMy.gap1ndexCeiling.value] += 1
		my[indexMy.indexLeaf.value] += 1

def leafBelowSentinelIs1(track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> Any:
	return track[indexTrack.leafBelow.value, 0] == 1

def loopingLeavesConnectedToActiveLeaf(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]

def loopingToActiveGapCeiling(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]

def loopUpToDimensionsTotal(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.indexDimension.value] < my[indexMy.dimensionsTotal.value]

def noGapsHere(my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] > 0 and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]

def placeLeaf(gapsWhere: ndarray[tuple[int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
	my[indexMy.gap1ndex.value] -= 1
	track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
	track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
	track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
	track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
	track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
	my[indexMy.leaf1ndex.value] += 1

def thereIsAnActiveLeaf(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] > 0

def thisIsMyTaskIndex(my: ndarray[tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] != my[indexMy.taskDivisions.value] or my[indexMy.leafConnectee.value] % my[indexMy.taskDivisions.value] == my[indexMy.taskIndex.value]

def updateLeafConnectee(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]], my: ndarray[tuple[int], dtype[integer[Any]]], track: ndarray[tuple[int, int], dtype[integer[Any]]]) -> None:
	my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]

def countInitialize(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]]
						, gapsWhere: ndarray[tuple[int]			 , dtype[integer[Any]]]
						,		 my: ndarray[tuple[int]			 , dtype[integer[Any]]]
						,	  track: ndarray[tuple[int, int]	 , dtype[integer[Any]]]
					) -> None:

	while activeLeafGreaterThan0(my=my):
		if activeLeafIsTheFirstLeaf(my=my) or leafBelowSentinelIs1(track=track):
			initializeVariablesToFindGaps(my=my, track=track)
			while loopUpToDimensionsTotal(my=my):
				if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
					decrementDimensionsUnconstrained(my=my)
				else:
					initializeLeafConnectee(connectionGraph=connectionGraph, my=my)
					while loopingLeavesConnectedToActiveLeaf(my=my):
						countGaps(gapsWhere=gapsWhere, my=my, track=track)
						updateLeafConnectee(connectionGraph=connectionGraph, my=my, track=track)
				incrementIndexDimension(my=my)
			if allDimensionsAreUnconstrained(my=my):
				insertUnconstrainedLeaf(gapsWhere=gapsWhere, my=my)
			initializeIndexMiniGap(my=my)
			while loopingToActiveGapCeiling(my=my):
				filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
				incrementIndexMiniGap(my=my)
		if thereIsAnActiveLeaf(my=my):
			placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
		if my[indexMy.gap1ndex.value] > 0:
			return

def countParallel(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]]
					,  foldGroups: ndarray[tuple[int]		   , dtype[integer[Any]]]
					,   gapsWhere: ndarray[tuple[int]		   , dtype[integer[Any]]]
					,		   my: ndarray[tuple[int]		   , dtype[integer[Any]]]
					,		track: ndarray[tuple[int, int]	   , dtype[integer[Any]]]
				) -> None:

	gapsWherePARALLEL = gapsWhere.copy()
	myPARALLEL = my.copy()
	trackPARALLEL = track.copy()

	taskDivisionsPrange = myPARALLEL[indexMy.taskDivisions.value]

	for indexSherpa in prange(taskDivisionsPrange):
		groupsOfFolds: int = 0

		gapsWhere = gapsWherePARALLEL.copy()
		my = myPARALLEL.copy()
		track = trackPARALLEL.copy()

		my[indexMy.taskIndex.value] = indexSherpa

		while activeLeafGreaterThan0(my=my):
			if activeLeafIsTheFirstLeaf(my=my) or leafBelowSentinelIs1(track=track):
				if activeLeafGreaterThanLeavesTotal(foldGroups=foldGroups, my=my):
					groupsOfFolds += 1
				else:
					initializeVariablesToFindGaps(my=my, track=track)
					while loopUpToDimensionsTotal(my=my):
						if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
							decrementDimensionsUnconstrained(my=my)
						else:
							initializeLeafConnectee(connectionGraph=connectionGraph, my=my)
							while loopingLeavesConnectedToActiveLeaf(my=my):
								if thisIsMyTaskIndex(my=my):
									countGaps(gapsWhere=gapsWhere, my=my, track=track)
								updateLeafConnectee(connectionGraph=connectionGraph, my=my, track=track)
						incrementIndexDimension(my=my)
					initializeIndexMiniGap(my=my)
					while loopingToActiveGapCeiling(my=my):
						filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
						incrementIndexMiniGap(my=my)
			while noGapsHere(my=my, track=track):
				backtrack(my=my, track=track)
			if thereIsAnActiveLeaf(my=my):
				placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
		foldGroups[my[indexMy.taskIndex.value]] = groupsOfFolds

def countSequential( connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]]
						, foldGroups: ndarray[tuple[int]		  , dtype[integer[Any]]]
						,  gapsWhere: ndarray[tuple[int]		  , dtype[integer[Any]]]
						,		  my: ndarray[tuple[int]		  , dtype[integer[Any]]]
						,	   track: ndarray[tuple[int, int]	  , dtype[integer[Any]]]
					) -> None:

	groupsOfFolds: int = 0

	while activeLeafGreaterThan0(my=my):
		if activeLeafIsTheFirstLeaf(my=my) or leafBelowSentinelIs1(track=track):
			if activeLeafGreaterThanLeavesTotal(foldGroups=foldGroups, my=my):
				groupsOfFolds += 1
			else:
				initializeVariablesToFindGaps(my=my, track=track)
				while loopUpToDimensionsTotal(my=my):
					initializeLeafConnectee(connectionGraph=connectionGraph, my=my)
					if activeLeafConnectedToItself(my=my):
						decrementDimensionsUnconstrained(my=my)
					else:
						while loopingLeavesConnectedToActiveLeaf(my=my):
							countGaps(gapsWhere=gapsWhere, my=my, track=track)
							updateLeafConnectee(connectionGraph=connectionGraph, my=my, track=track)
					incrementIndexDimension(my=my)
				initializeIndexMiniGap(my=my)
				while loopingToActiveGapCeiling(my=my):
					filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
					incrementIndexMiniGap(my=my)
		while noGapsHere(my=my, track=track):
			backtrack(my=my, track=track)
		if thereIsAnActiveLeaf(my=my):
			placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
	foldGroups[my[indexMy.taskIndex.value]] = groupsOfFolds

def doTheNeedful(connectionGraph: ndarray[tuple[int, int, int], dtype[integer[Any]]]
					, foldGroups: ndarray[tuple[int]		  , dtype[integer[Any]]]
					,  gapsWhere: ndarray[tuple[int]		  , dtype[integer[Any]]]
					,   mapShape: ndarray[tuple[int]		  , dtype[integer[Any]]]
					,		  my: ndarray[tuple[int]		  , dtype[integer[Any]]]
					,	   track: ndarray[tuple[int, int]	  , dtype[integer[Any]]]
					) -> None:

	countInitialize(connectionGraph, gapsWhere, my, track)

	if my[indexMy.taskDivisions.value] > 0:
		countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
	else:
		countSequential(connectionGraph, foldGroups, gapsWhere, my, track)
