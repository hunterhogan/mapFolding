from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from mapFolding._e import (
	dimensionNearestTail, dimensionNearest首, dimensionSecondNearest首, getDictionaryPileRanges, getLeaf,
	getLeavesCreaseBack, ptount, 一, 三, 二, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e.dataBaskets import EliminationState

# ======= Domain-based subroutines for analyzing a specific `pile`. =======

def pinPile首零Less零AfterFourthOrder(state: EliminationState) -> list[int]:
	leaf: int = -1
	leafAt一: int = raiseIfNone(getLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - 一))
	leafAt一零: int = raiseIfNone(getLeaf(state.leavesPinned, 一+零))
	leafAt首Less一零: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - (一+零)))
	leafAt二: int = raiseIfNone(getLeaf(state.leavesPinned, 二))

	dictionaryPileToLeaves: dict[int, tuple[int, ...]] = getDictionaryPileRanges(state)
	listRemoveLeaves: list[int] = []

	pileExcluder: int = 一
	for dimension in range(state.dimensionsTotal):
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt一:
				listRemoveLeaves.extend([一, 首零(state.dimensionsTotal) + leafAt一])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt一:
				listRemoveLeaves.extend([一 + leafAt一])
		if dimension == 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt一:
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + leafAt一 + 零])
		if dimension == state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt一:
				listRemoveLeaves.extend([首一(state.dimensionsTotal), 首一(state.dimensionsTotal) + leafAt一])
	del pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - 一
	for dimension in range(state.dimensionsTotal):
		if dimension == 0:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([一])
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAt首Less一])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([int(bit_flip(0, dimension)), 首一(state.dimensionsTotal) + leafAt首Less一 - (int(bit_flip(0, dimension)) - 零)])
		if 0 < dimension < state.dimensionsTotal - 3:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([零 + leafAt首Less一])
		if 0 < dimension < state.dimensionsTotal - 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([首一(state.dimensionsTotal)])
	del pileExcluder

	pileExcluder = 一+零
	if leafAt一零 == 三+二+零:
		listRemoveLeaves.extend([二+一+零, 首零(state.dimensionsTotal)+二+零])
	if leafAt一零 == 首一(state.dimensionsTotal)+二+零:
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首一二(state.dimensionsTotal)+零, 首零一二(state.dimensionsTotal)])
	if leafAt一零 == 首一二(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零二(state.dimensionsTotal)+零])
	if leafAt一零 == 首零一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	if is_odd(leafAt一零):
		listRemoveLeaves.extend([leafAt一零, state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt一零))]])
		if leafAt一零 < 首零(state.dimensionsTotal):
			comebackOffset: int = state.sumsOfProductsOfDimensions[ptount(leafAt一零) + 1]
			listRemoveLeaves.extend([
				一
				, leafAt一零 + 首零(state.dimensionsTotal)-零
				, leafAt一零 + 首零(state.dimensionsTotal)-零 - comebackOffset
			])
			if ptount(leafAt一零) == 1:
				listRemoveLeaves.extend([
					state.productsOfDimensions[dimensionNearest首(leafAt一零)] + comebackOffset
					, 首零(state.dimensionsTotal) + comebackOffset
				])
		if 首零(state.dimensionsTotal) < leafAt一零:
			listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, state.productsOfDimensions[dimensionNearest首(leafAt一零) - 1]])
	del pileExcluder

	pileExcluder = state.leavesTotal - (一+零)
	if 首零(state.dimensionsTotal) < leafAt首Less一零:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, leafAt首Less一零])
		if is_even(leafAt首Less一零):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			bit = 1
			if bit_test(leafAt首Less一零, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 2
			if bit_test(leafAt首Less一零, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				if 1 < dimensionNearestTail(leafAt首Less一零):
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 3
			if bit_test(leafAt首Less一零, bit):
				if 1 < dimensionNearestTail(leafAt首Less一零):
					listRemoveLeaves.extend([2**bit])
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
				if dimensionNearestTail(leafAt首Less一零) < bit:
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])

			sheepOrGoat = 0
			shepherdOfDimensions: int = int(bit_flip(0, state.dimensionsTotal - 5))
			if (leafAt首Less一零//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveLeaves.extend([二])
				sheepOrGoat = ptount(leafAt首Less一零//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset = int(bit_flip(0, dimensionNearest首(leafAt首Less一零))) - 二
					listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = int(bit_flip(0, raiseIfNone(dimensionSecondNearest首(leafAt首Less一零)))) - 二
					listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])

		if is_odd(leafAt首Less一零):
			listRemoveLeaves.extend([一])
			if leafAt首Less一零 & bit_mask(4) == 0b001001:
				listRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAt首Less一零)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = int(bit_flip(0, dimensionNearest首(leafAt首Less一零))) - 一
				listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = int(bit_flip(0, raiseIfNone(dimensionSecondNearest首(leafAt首Less一零)))) - 一
				listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])

	pileExcluder = 二
	if is_even(leafAt二):
		listRemoveLeaves.extend([一, leafAt二 + 零, 首零(state.dimensionsTotal)+一+零])
	if is_odd(leafAt二):
		listRemoveLeaves.extend([leafAt二 - 零])
		if 首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零一(state.dimensionsTotal)+零])
		if 首零(state.dimensionsTotal) < leafAt二:
			listRemoveLeaves.extend([首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)+零])
			bit = 1
			if bit_test(leafAt二, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 2
			if bit_test(leafAt二, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 3
			if bit_test(leafAt二, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 4
			if bit_test(leafAt二, bit) and (leafAt二.bit_length() > 5):
				listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	del pileExcluder

	if (leafAt一零 != 首零一(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.append(一)
	if (leafAt首Less一零 != next(getLeavesCreaseBack(state, 首零(state.dimensionsTotal)+零))) and (leafAt一 == 一+零):
		listRemoveLeaves.append(首一(state.dimensionsTotal))
	if (leafAt一 == 首二(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less一 + 零])
	if leafAt一.bit_length() < state.dimensionsTotal - 2:
		listRemoveLeaves.extend([一, leafAt首Less一 + 一])

	return sorted(set(dictionaryPileToLeaves[state.pile]).difference(set(listRemoveLeaves)))

