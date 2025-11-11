# ruff: noqa ERA001
from collections.abc import Callable
from gmpy2 import bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from mapFolding.algorithms.patternFinder import (
	bitBiggest, bitSecond, distanceFromPowerOf2, getDictionaryAddends4Next, getDictionaryAddends4Prior,
	getDictionaryLeafRanges, getDictionaryPileToLeaves, getExcludedAddendIndices, getExcludedIndexLeaves,
	multiplicityOfPrimeFactor2, numeralOfLengthInBase)
from mapFolding.algorithms.pinning2DnAnnex import addItUp, exclude_rBefore_k
from mapFolding.dataBaskets import EliminationState
from mapFolding.tests.verify import printStatisticsPermutations, verifyPinning2Dn
from math import log2, prod
from pprint import pprint

def pinByFormula(state: EliminationState, maximumListPinnedLeaves: int = 5000) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (state.mapShape[0] == 2)):
		return state

	def appendPinnedLeavesAtPile(pinnedLeavesWorkbench: dict[int, int], listIndexLeavesAtPile: list[int], pile: int, mapShape: tuple[int, ...]) -> None:
		for indexLeaf in listIndexLeavesAtPile:
			if indexLeaf < sumsProductsOfDimensions[0]:
				continue

			if indexLeaf > sumsProductsOfDimensions[-1]:
				continue

			if indexLeaf in pinnedLeavesWorkbench.values():
				continue

			if pile not in list(dictionaryLeafRanges[indexLeaf]):
				continue

			if exclude_rBefore_k(pile, indexLeaf, pinnedLeavesWorkbench, mapShape):
				continue

			pileToPin: int = pile
			pinnedLeaves: dict[int, int] = pinnedLeavesWorkbench.copy()

			if indexLeaf in [0b000011, ordinal([0,1],'0',0)]:
				pinnedLeaves[pileToPin] = indexLeaf
				pileToPin += 1
				indexLeaf += dictionaryAddends4Next[indexLeaf][0]

			if indexLeaf in [0b000010, ordinal([1,1],'0',0)]:
				pinnedLeaves[pileToPin] = indexLeaf
				pileToPin -= 1
				indexLeaf += dictionaryAddends4Prior[indexLeaf][0]

			pinnedLeaves[pileToPin] = indexLeaf
			state.listPinnedLeaves.append(pinnedLeaves.copy())

	def nextPinnedLeavesWorkbench(state: EliminationState) -> dict[int, int] | None:
		pinnedLeavesWorkbench: dict[int, int] | None = None
		for pile in pileProcessingOrder:
			if pile == queueStopBefore:
				break
			if not all(pile in pinnedLeaves for pinnedLeaves in state.listPinnedLeaves):
				pinnedLeavesWorkbench = next((pinnedLeaves.copy() for pinnedLeaves in state.listPinnedLeaves if pile not in pinnedLeaves))
				state.listPinnedLeaves.remove(pinnedLeavesWorkbench)
				break
		return pinnedLeavesWorkbench

	def whereNext(pilingsPinned: list[int]) -> int:
		return next(pile for pile in pileProcessingOrder if pile not in pilingsPinned)

	ordinal: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state.dimensionsTotal, base=state.mapShape[0])
	"""Prototype."""

	productsOfDimensions:		list[int] = [prod(state.mapShape[0:dimension]) for dimension in range(state.dimensionsTotal + 1)]
	sumsProductsOfDimensions:	list[int] = [sum(productsOfDimensions[0:dimension]) for dimension in range(state.dimensionsTotal + 1)]

	dictionaryAddends4Next:		dict[int, list[int]]	= getDictionaryAddends4Next(state)
	dictionaryAddends4Prior:	dict[int, list[int]]	= getDictionaryAddends4Prior(state)
	dictionaryLeafRanges:		dict[int, range]		= getDictionaryLeafRanges(state)
	dictionaryPileToLeaves:		dict[int, list[int]]	= getDictionaryPileToLeaves(state)

	state.listPinnedLeaves = state.listPinnedLeaves or [{0b000000: 0b000000}]

	pileProcessingOrder: list[int] = [0b000000, 0b000001, 0b000010, ordinal([1,1],'1',1), ordinal([1,1],'1',0), 0b000011, ordinal([1,1],'1',[0,1]), 0b000100]
	# pileProcessingOrder.extend([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

	pileProcessingOrder.extend([ordinal([0,1],'1',1), ordinal([0,1],'0',1), ordinal([1,1],'0',1)])
	queueStopBefore: int = ordinal([0,1],'0',1)

	oneTime: int = 0

	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (pinnedLeavesWorkbench := nextPinnedLeavesWorkbench(state)):
		pile: int = whereNext(list(pinnedLeavesWorkbench.keys()))
		listIndexLeavesAtPile: list[int] = []
		listAddendIndicesExcluded: list[int] = []
		listRemoveIndexLeaves: list[int] = []

		if pile == 0b000000:
			listIndexLeavesAtPile.append(0b000000)
		if pile == 0b000001:
			listIndexLeavesAtPile.append(0b000001)
		if pile == 0b000010:
			listIndexLeavesAtPile = addItUp(dictionaryAddends4Next, pinnedLeavesWorkbench[pile - 1], listAddendIndicesExcluded)
		if pile == ordinal([1,1],'1',1):
			listIndexLeavesAtPile.append(ordinal([1,0],'0',0))
		if pile == ordinal([1,1],'1',0):
			indexLeafAt__10: int = pinnedLeavesWorkbench[0b000010]
			if indexLeafAt__10.bit_length() < state.dimensionsTotal:
				listAddendIndicesExcluded.extend([*range(0b000001, indexLeafAt__10.bit_length())])
			listIndexLeavesAtPile = addItUp(dictionaryAddends4Prior, pinnedLeavesWorkbench[pile + 1], listAddendIndicesExcluded)
		if pile == 0b000011:
			listAddendIndicesExcluded.append(0)
			indexLeafAtPileLess1: int = pinnedLeavesWorkbench[pile - 1]
			indexLeafAt11ones0: int = pinnedLeavesWorkbench[ordinal([1,1],'1',0)]
			if is_even(indexLeafAt11ones0) and (indexLeafAtPileLess1 == ordinal([1,0],'0',1)):
				listAddendIndicesExcluded.extend([*range(multiplicityOfPrimeFactor2(pinnedLeavesWorkbench[ordinal([1,1],'1',0)]) + 1, state.dimensionsTotal)])
			listIndexLeavesAtPile = addItUp(dictionaryAddends4Next, indexLeafAtPileLess1, listAddendIndicesExcluded)
		if pile == ordinal([1,1],'1',[0,1]):
			indexLeafAtPilePlus1: int = pinnedLeavesWorkbench[pile + 1]
			if indexLeafAtPilePlus1 < ordinal([1,1],'0',0):
				listAddendIndicesExcluded.append(-1)
			indexLeafAt__10 = pinnedLeavesWorkbench[0b000010]
			if (indexLeafAtPilePlus1 == ordinal([1,0],'0',1)) and (indexLeafAt__10 != 0b000011):
				listAddendIndicesExcluded.extend([*range(0, indexLeafAt__10.bit_length() - 2)])
			listIndexLeavesAtPile = addItUp(dictionaryAddends4Prior, indexLeafAtPilePlus1, listAddendIndicesExcluded)
		if pile == 0b000100:
			indexLeafAtPileLess1 = pinnedLeavesWorkbench[pile - 1]
			if is_odd(indexLeafAtPileLess1):
				listAddendIndicesExcluded.extend([*range(indexLeafAtPileLess1.bit_length() - 1, 5), distanceFromPowerOf2(indexLeafAtPileLess1 - 0b000011).bit_count()])
			indexLeafAt11ones0 = pinnedLeavesWorkbench[ordinal([1,1],'1',0)]
			if is_even(indexLeafAtPileLess1) and is_even(indexLeafAt11ones0):
				listAddendIndicesExcluded.extend([*range(multiplicityOfPrimeFactor2(distanceFromPowerOf2(indexLeafAt11ones0)) - 0b000010, (state.dimensionsTotal - 3))])
			if is_odd(indexLeafAtPileLess1):
				listAddendIndicesExcluded.append((int(log2(distanceFromPowerOf2(indexLeafAt11ones0))) + 4) % 5)
			indexLeafAt11ones01: int = pinnedLeavesWorkbench[ordinal([1,1],'1',[0,1])]
			if is_even(indexLeafAtPileLess1) and indexLeafAt11ones01:
				listAddendIndicesExcluded.extend([*range((state.dimensionsTotal - 3))][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - distanceFromPowerOf2(indexLeafAt11ones01 - (indexLeafAt11ones01.bit_count() - is_even(indexLeafAt11ones01))).bit_count()) % (state.dimensionsTotal - 2) - is_even(indexLeafAt11ones01): None])
			indexLeafAt__10 = pinnedLeavesWorkbench[0b000010]
			if (indexLeafAt__10 == ordinal([1,0],'0',1)):
				listAddendIndicesExcluded.extend([(int(log2(distanceFromPowerOf2(indexLeafAt11ones0))) + 4) % 5, multiplicityOfPrimeFactor2(indexLeafAt11ones01) - 1])
			if (indexLeafAt__10 == ordinal([1,0],'0',1)) and (indexLeafAt11ones01 > ordinal([1,0],'0',1)):
				listAddendIndicesExcluded.extend([*range(int(indexLeafAt11ones01 - 2**(indexLeafAt11ones01.bit_length() - 1)).bit_length() - 1, state.dimensionsTotal - 2)])
			if ((indexLeafAt__10 == ordinal([1,0],'0',1)) and (0 < indexLeafAtPileLess1 - indexLeafAt__10 <= 2**(state.dimensionsTotal - 4)) and (0 < (indexLeafAt11ones0 - indexLeafAtPileLess1) <= 2**(state.dimensionsTotal - 3))):
				listAddendIndicesExcluded.extend([distanceFromPowerOf2(indexLeafAtPileLess1 - 0b11).bit_count(), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
			listIndexLeavesAtPile = addItUp(dictionaryAddends4Next, indexLeafAtPileLess1, listAddendIndicesExcluded)


		if pile == ordinal([0,1],'1',1):
			listRemoveIndexLeaves = []

			pileExcluder: int = 0b000010
			indexLeafAtPileExcluder: int = pinnedLeavesWorkbench[pileExcluder]
			for d in range(state.dimensionsTotal):
				if d < state.dimensionsTotal - 2:
					indexLeaf: int = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([0b000010, ordinal([1,0],'0',0) + indexLeafAtPileExcluder])
				if 0 < d < state.dimensionsTotal - 2:
					indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([0b000010 + indexLeafAtPileExcluder])
				if d == 1:
					indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([ordinal([1,0],'0',0) + indexLeafAtPileExcluder + 1])
				if d == state.dimensionsTotal - 2:
					indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([ordinal([0,1],'0',0), ordinal([0,1],'0',0) + indexLeafAtPileExcluder])
			del pileExcluder

			pileExcluder = ordinal([1,1],'1',0)
			indexLeafAtPileExcluder = pinnedLeavesWorkbench[pileExcluder]
			for d in range(state.dimensionsTotal):
				if d == 0:
					indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([0b000010])
				if d < state.dimensionsTotal - 2:
					indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([ordinal([0,1],'0',0) + indexLeafAtPileExcluder])
				if 0 < d < state.dimensionsTotal - 2:
					indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([2**d, ordinal([0,1],'0',0) + indexLeafAtPileExcluder - (2**d - 0b000001)])
				if 0 < d < state.dimensionsTotal - 3:
					indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([0b000001 + indexLeafAtPileExcluder])
				if 0 < d < state.dimensionsTotal - 1:
					indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
					if indexLeaf == indexLeafAtPileExcluder:
						listRemoveIndexLeaves.extend([ordinal([0,1],'0',0)])
			del pileExcluder

			"""indexLeafAt__11
			(2,) * 5:
			13      False   [2, 4, 7, 11, 13, 19, 21, 25, 28]       [13, 4, 2, 28, 25, 11, 19]
			25      False   [8, 25, 28]     [25, 8, 25, 8]

			(2,) * 6:
			13      False   [2, 4, 7, 11, 13, 35, 37, 41, 44]       [13, 4, 2, 44, 41, 11, 35]
			21      False   [2, 4, 8, 19, 21, 25, 28, 35, 49, 52, 56]       [21, 4, 2, 52, 49, 19, 35]
			25      False   [2, 8, 19, 25, 41, 49, 56, 59]  [25, 8, 2, 56, 49]
			49      False   [16, 49, 56]    [49, 16, 49, 16]
			"""
			pileExcluder = 0b000011
			indexLeafAtPileExcluder = pinnedLeavesWorkbench[pileExcluder]

			if is_odd(indexLeafAtPileExcluder):
				listRemoveIndexLeaves.extend([indexLeafAtPileExcluder, productsOfDimensions[raiseIfNone(bitSecond(indexLeafAtPileExcluder))]])

				if indexLeafAtPileExcluder < ordinal([1,0],'0',0):
					comebackOffset: int = sumsProductsOfDimensions[distanceFromPowerOf2(indexLeafAtPileExcluder - 0b000011).bit_count() + 1]
					listRemoveIndexLeaves.extend([
						0b000010
						, indexLeafAtPileExcluder + ordinal([0,1],'1',1)
						, indexLeafAtPileExcluder + ordinal([0,1],'1',1) - comebackOffset
					])
					if distanceFromPowerOf2(indexLeafAtPileExcluder - 0b000011).bit_count() == 1:
						listRemoveIndexLeaves.extend([
							productsOfDimensions[bitBiggest(indexLeafAtPileExcluder)] + comebackOffset
							, ordinal([1,0],'0',0) + comebackOffset
						])

				if ordinal([1,0],'0',0) < indexLeafAtPileExcluder:
					listRemoveIndexLeaves.extend([ordinal([1,1],'0',1), productsOfDimensions[bitBiggest(indexLeafAtPileExcluder) - 1]])
			del pileExcluder

			"""ordinal([1,1],'1',[0,1])
			38: [2, 4, 16, 21, 35, 37, 38, 49, 50], # NOTE missing 21
			42: [2, 4, 14, 16, 35, 37, 38, 41, 42, 49, 50], # NOTE missing 4, 14, 37, 38, 41
			"""

			pileExcluder = ordinal([1,1],'1',[0,1])
			indexLeafAtPileExcluder = pinnedLeavesWorkbench[pileExcluder]
			if ordinal([1,0],'0',0) < indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([ordinal([1,1],'0',1), indexLeafAtPileExcluder])

				if is_even(indexLeafAtPileExcluder):
					listRemoveIndexLeaves.extend([ordinal([0,1],'0',0)])
					bit = 1
					if bit_test(indexLeafAtPileExcluder, bit):
						listRemoveIndexLeaves.extend([2**bit, ordinal([1,0],'0',0) + 2**bit + 0b000001])
						listRemoveIndexLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
					bit = 2
					if bit_test(indexLeafAtPileExcluder, bit):
						listRemoveIndexLeaves.extend([2**bit, ordinal([1,0],'0',0) + 2**bit + 0b000001])
						if 1 < multiplicityOfPrimeFactor2(indexLeafAtPileExcluder):
							listRemoveIndexLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
					bit = 3
					if bit_test(indexLeafAtPileExcluder, bit) and (1 < multiplicityOfPrimeFactor2(indexLeafAtPileExcluder)):
						listRemoveIndexLeaves.extend([2**bit])
						listRemoveIndexLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
						if multiplicityOfPrimeFactor2(indexLeafAtPileExcluder) < state.dimensionsTotal - 3:
							listRemoveIndexLeaves.extend([ordinal([1,0],'0',0) + 2**bit + 0b000001])

				if is_odd(indexLeafAtPileExcluder):
					listRemoveIndexLeaves.extend([0b000010])

					sheepOrGoat = distanceFromPowerOf2(indexLeafAtPileExcluder - 0b000011).bit_count()
					if 0 < sheepOrGoat < state.dimensionsTotal - 3:
						comebackOffset = 2**bitBiggest(indexLeafAtPileExcluder) - 2
						listRemoveIndexLeaves.extend([indexLeafAtPileExcluder - comebackOffset])
					if 0 < sheepOrGoat < state.dimensionsTotal - 4:
						comebackOffset = 2**raiseIfNone(bitSecond(indexLeafAtPileExcluder)) - 2
						listRemoveIndexLeaves.extend([indexLeafAtPileExcluder - comebackOffset])

			pileExcluder = 0b000100
			indexLeafAtPileExcluder = pinnedLeavesWorkbench[pileExcluder]

			if is_even(indexLeafAtPileExcluder):
				listRemoveIndexLeaves.extend([0b000010, indexLeafAtPileExcluder + 1, ordinal([1,0],'0',0) + 0b000011])
			if is_odd(indexLeafAtPileExcluder):
				listRemoveIndexLeaves.extend([indexLeafAtPileExcluder - 1])
				if ordinal([0,1],'0',0) < indexLeafAtPileExcluder < ordinal([1,0],'0',0):
					listRemoveIndexLeaves.extend([ordinal([0,1],'0',0) + 0b000011, ordinal([1,1],'0',1)])
				if ordinal([1,0],'0',0) < indexLeafAtPileExcluder:
					listRemoveIndexLeaves.extend([ordinal([0,1],'0',0), ordinal([1,1],'0',1)])
					bit = 1
					if bit_test(indexLeafAtPileExcluder, bit):
						listRemoveIndexLeaves.extend([2**bit, ordinal([1,0],'0',0) + 2**bit + 0b000001])
					bit = 2
					if bit_test(indexLeafAtPileExcluder, bit):
						listRemoveIndexLeaves.extend([ordinal([1,0],'0',0) + 2**bit + 0b000001])
					bit = 3
					if bit_test(indexLeafAtPileExcluder, bit):
						listRemoveIndexLeaves.extend([ordinal([1,0],'0',0) + 2**bit + 0b000001])
					bit = 4
					if bit_test(indexLeafAtPileExcluder, bit) and (indexLeafAtPileExcluder.bit_length() > 5):
						listRemoveIndexLeaves.extend([ordinal([1,1,1],'0',0)])
			del pileExcluder

			indexLeafAt__10 = pinnedLeavesWorkbench[0b000010]
			indexLeafAt11ones0 = pinnedLeavesWorkbench[ordinal([1,1],'1',0)]
			indexLeafAt__11 = pinnedLeavesWorkbench[0b000011]
			indexLeafAt11ones01 = pinnedLeavesWorkbench[ordinal([1,1],'1',[0,1])]

			if (indexLeafAt__11 != ordinal([1,1],'0',1)) and (indexLeafAt11ones0 == ordinal([1,1],'0',0)):
				listRemoveIndexLeaves.append(0b000010)
			if (indexLeafAt11ones01 != ordinal([1,0],'0',1) + dictionaryAddends4Prior[ordinal([1,0],'0',1)][0]) and (indexLeafAt__10 == 0b000011):
				listRemoveIndexLeaves.append(ordinal([0,1],'0',0))

			tuplePilesExcluders = (0b000010, ordinal([1,1],'1',0))
			(indexLeavesAtPilesExcluders) = (indexLeafAt__10, indexLeafAt11ones0)
			if (indexLeafAt__10 == ordinal([0,0,1],'0',1)) and (indexLeafAt11ones0 == ordinal([1,1],'0',0)):
				listRemoveIndexLeaves.extend([ordinal([0,0,1],'0',0), ordinal([1,1,1],'0',0)])
			if indexLeafAt__10 == ordinal([1,0],'0',1):
				listRemoveIndexLeaves.extend([ordinal([0,1],'0',0), indexLeafAt11ones0 + 0b000001])
			if indexLeafAt__10.bit_length() < state.dimensionsTotal - 2:
				listRemoveIndexLeaves.extend([0b000010, indexLeafAt11ones0 + 0b000010])

			tuplePilesExcluders = (0b000010, 0b000011)
			(indexLeavesAtPilesExcluders) = (indexLeafAt__10, indexLeafAt__11)

			tuplePilesExcluders = (ordinal([1,1],'1',0), ordinal([1,1],'1',[0,1]))
			(indexLeavesAtPilesExcluders) = (indexLeafAt11ones0, indexLeafAt11ones01)

			tuplePilesExcluders = (0b000011, ordinal([1,1],'1',[0,1]))
			(indexLeavesAtPilesExcluders) = (indexLeafAt__11, indexLeafAt11ones01)

			# listRemoveIndexLeaves: list[int] = []

			dictionaryExcludedIndexLeaves = getExcludedIndexLeaves(state, pile, tuplePilesExcluders)
			if oneTime < 1:
				oneTime += 1
				# pprint(dictionaryExcludedIndexLeaves, width=140)

			listExcludedIndexLeavesGoal = dictionaryExcludedIndexLeaves[((indexLeavesAtPilesExcluders))]

			# print(indexLeavesAtPilesExcluders, sorted(set(listExcludedIndexLeavesGoal).difference(set(listRemoveIndexLeaves)))
			# 	, listExcludedIndexLeavesGoal == sorted(set(listRemoveIndexLeaves)), sorted(set(listRemoveIndexLeaves)), listExcludedIndexLeavesGoal, sep='\t')

			listIndexLeavesAtPile = sorted(set(dictionaryPileToLeaves[pile]).difference(set(listRemoveIndexLeaves)))

		appendPinnedLeavesAtPile(pinnedLeavesWorkbench, listIndexLeavesAtPile, pile, state.mapShape)

	return state

if __name__ == '__main__':
	state = EliminationState((2,) * 5)
	state: EliminationState = pinByFormula(state)

	# pprint(state.listPinnedLeaves)
	print(f"{len(state.listPinnedLeaves)=}")

	printStatisticsPermutations(state)
	verifyPinning2Dn(state)

