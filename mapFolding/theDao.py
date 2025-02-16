from mapFolding import indexMy, indexTrack
from numba import prange
from numpy import dtype, integer, ndarray
from typing import Any, Tuple

def activeGapIncrement(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> None:
	# `.value` is not necessary for this module or most modules. But, this module is transformed into Numba "jitted" functions, and Numba won't use `Enum` for an ndarray index without `.value`.
	my[indexMy.gap1ndex.value] += 1

def activeLeafGreaterThan0Condition(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value]

def activeLeafGreaterThanLeavesTotalCondition(foldGroups: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] > foldGroups[-1]

def activeLeafIsTheFirstLeafCondition(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] <= 1

def allDimensionsAreUnconstrained(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return not my[indexMy.dimensionsUnconstrained.value]

def backtrack(my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
	my[indexMy.leaf1ndex.value] -= 1
	track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
	track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]

def backtrackCondition(my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]

def gap1ndexCeilingIncrement(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.gap1ndexCeiling.value] += 1

def countGaps(gapsWhere: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
	gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
	if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
		gap1ndexCeilingIncrement(my=my)
	track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1

def dimensionsUnconstrainedCondition(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]

def dimensionsUnconstrainedDecrement(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.dimensionsUnconstrained.value] -= 1

def filterCommonGaps(gapsWhere: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
	gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
	if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == my[indexMy.dimensionsUnconstrained.value]:
		activeGapIncrement(my=my)
	track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] = 0

def findGapsInitializeVariables(my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
	my[indexMy.dimensionsUnconstrained.value] = my[indexMy.dimensionsTotal.value]
	my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
	my[indexMy.indexDimension.value] = 0

def indexDimensionIncrement(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.indexDimension.value] += 1

def indexMiniGapIncrement(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.indexMiniGap.value] += 1

def indexMiniGapInitialization(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]

def insertUnconstrainedLeaf(gapsWhere: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.indexLeaf.value] = 0
	while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
		gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.indexLeaf.value]
		my[indexMy.gap1ndexCeiling.value] += 1
		my[indexMy.indexLeaf.value] += 1

def leafBelowSentinelIs1Condition(track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> Any:
	return track[indexTrack.leafBelow.value, 0] == 1

def leafConnecteeInitialization(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]]) -> None:
	my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]

def leafConnecteeUpdate(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
	my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]

def activeLeafConnectedToItself(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leafConnectee.value] == my[indexMy.leaf1ndex.value]

def loopingLeavesConnectedToActiveLeaf(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]

def loopUpToDimensionsTotal(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.indexDimension.value] < my[indexMy.dimensionsTotal.value]

def loopingToActiveGapCeiling(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]

def placeLeaf(gapsWhere: ndarray[Tuple[int], dtype[integer[Any]]], my: ndarray[Tuple[int], dtype[integer[Any]]], track: ndarray[Tuple[int, int], dtype[integer[Any]]]) -> None:
	my[indexMy.gap1ndex.value] -= 1
	track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
	track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
	track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
	track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
	track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
	my[indexMy.leaf1ndex.value] += 1

def placeLeafCondition(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value]

def thereAreComputationDivisionsYouMightSkip(my: ndarray[Tuple[int], dtype[integer[Any]]]) -> Any:
	return my[indexMy.leaf1ndex.value] != my[indexMy.taskDivisions.value] or my[indexMy.leafConnectee.value] % my[indexMy.taskDivisions.value] == my[indexMy.taskIndex.value]

def countInitialize(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]]
						, gapsWhere: ndarray[Tuple[int]			 , dtype[integer[Any]]]
						,		 my: ndarray[Tuple[int]			 , dtype[integer[Any]]]
						,	  track: ndarray[Tuple[int, int]	 , dtype[integer[Any]]]
					) -> None:

	while activeLeafGreaterThan0Condition(my=my):
		if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
			findGapsInitializeVariables(my=my, track=track)
			while loopUpToDimensionsTotal(my=my):
				if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
					dimensionsUnconstrainedDecrement(my=my)
				else:
					leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
					while loopingLeavesConnectedToActiveLeaf(my=my):
						countGaps(gapsWhere=gapsWhere, my=my, track=track)
						leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
				indexDimensionIncrement(my=my)
			if allDimensionsAreUnconstrained(my=my):
				insertUnconstrainedLeaf(gapsWhere=gapsWhere, my=my)
			indexMiniGapInitialization(my=my)
			while loopingToActiveGapCeiling(my=my):
				filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
				indexMiniGapIncrement(my=my)
		if placeLeafCondition(my=my):
			placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
		if my[indexMy.gap1ndex.value] > 0:
			return

def countParallel(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]]
					,  foldGroups: ndarray[Tuple[int]		   , dtype[integer[Any]]]
					,   gapsWhere: ndarray[Tuple[int]		   , dtype[integer[Any]]]
					,		   my: ndarray[Tuple[int]		   , dtype[integer[Any]]]
					,		track: ndarray[Tuple[int, int]	   , dtype[integer[Any]]]
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

		while activeLeafGreaterThan0Condition(my=my):
			if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
				if activeLeafGreaterThanLeavesTotalCondition(foldGroups=foldGroups, my=my):
					groupsOfFolds += 1
				else:
					findGapsInitializeVariables(my=my, track=track)
					while loopUpToDimensionsTotal(my=my):
						if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
							dimensionsUnconstrainedDecrement(my=my)
						else:
							leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
							while loopingLeavesConnectedToActiveLeaf(my=my):
								if thereAreComputationDivisionsYouMightSkip(my=my):
									countGaps(gapsWhere=gapsWhere, my=my, track=track)
								leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
						indexDimensionIncrement(my=my)
					indexMiniGapInitialization(my=my)
					while loopingToActiveGapCeiling(my=my):
						filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
						indexMiniGapIncrement(my=my)
			while backtrackCondition(my=my, track=track):
				backtrack(my=my, track=track)
			if placeLeafCondition(my=my):
				placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
		foldGroups[my[indexMy.taskIndex.value]] = groupsOfFolds

def countSequential( connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]]
						, foldGroups: ndarray[Tuple[int]		  , dtype[integer[Any]]]
						,  gapsWhere: ndarray[Tuple[int]		  , dtype[integer[Any]]]
						,		  my: ndarray[Tuple[int]		  , dtype[integer[Any]]]
						,	   track: ndarray[Tuple[int, int]	  , dtype[integer[Any]]]
					) -> None:

	groupsOfFolds: int = 0

	while activeLeafGreaterThan0Condition(my=my):
		if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
			if activeLeafGreaterThanLeavesTotalCondition(foldGroups=foldGroups, my=my):
				groupsOfFolds += 1
			else:
				findGapsInitializeVariables(my=my, track=track)
				while loopUpToDimensionsTotal(my=my):
					leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
					if activeLeafConnectedToItself(my=my):
						dimensionsUnconstrainedDecrement(my=my)
					else:
						while loopingLeavesConnectedToActiveLeaf(my=my):
							countGaps(gapsWhere=gapsWhere, my=my, track=track)
							leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
					indexDimensionIncrement(my=my)
				indexMiniGapInitialization(my=my)
				while loopingToActiveGapCeiling(my=my):
					filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
					indexMiniGapIncrement(my=my)
		while backtrackCondition(my=my, track=track):
			backtrack(my=my, track=track)
		if placeLeafCondition(my=my):
			placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
	foldGroups[my[indexMy.taskIndex.value]] = groupsOfFolds

def doTheNeedful(connectionGraph: ndarray[Tuple[int, int, int], dtype[integer[Any]]]
					, foldGroups: ndarray[Tuple[int]		  , dtype[integer[Any]]]
					,  gapsWhere: ndarray[Tuple[int]		  , dtype[integer[Any]]]
					,   mapShape: ndarray[Tuple[int]		  , dtype[integer[Any]]]
					,		  my: ndarray[Tuple[int]		  , dtype[integer[Any]]]
					,	   track: ndarray[Tuple[int, int]	  , dtype[integer[Any]]]
					) -> None:

	countInitialize(connectionGraph, gapsWhere, my, track)

	if my[indexMy.taskDivisions.value] > 0:
		countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
	else:
		countSequential(connectionGraph, foldGroups, gapsWhere, my, track)
