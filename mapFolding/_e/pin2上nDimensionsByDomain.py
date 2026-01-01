from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from mapFolding._e import (
	dimensionIndex, dimensionNearestTail, dimensionNearest首, dimensionSecondNearest首, getDictionaryPileRanges, getLeaf,
	getLeavesCreaseBack, ptount, Z0Z_0NearestTail, 一, 三, 二, 四, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e.dataBaskets import EliminationState

# ======= Domain-based subroutines for analyzing a specific `pile`. =======

def pinPile首零Less零AfterFourthOrder(state: EliminationState) -> list[int]:
	leaf: int = -1
	leafAt一: int = raiseIfNone(getLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - 一))
	leafAt一零: int = raiseIfNone(getLeaf(state.leavesPinned, 一+零))
	leafAt首Less一零: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - (一+零)))
	leafAt二: int = raiseIfNone(getLeaf(state.leavesPinned, 二))
	leafAt首Less二: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - (二)))

	dictionaryPileToLeaves: dict[int, tuple[int, ...]] = getDictionaryPileRanges(state)
	listRemoveLeaves: list[int] = []

# ======== use leafAt一 to exclude a `leaf` from `pile` ===================
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

# ======== use leafAt首Less一 to exclude a `leaf` from `pile` ===================
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

# ======== use leafAt一零 to exclude a `leaf` from `pile` ===================
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

# ======== use leafAt首Less一零 to exclude a `leaf` from `pile` ===================
	if 首零(state.dimensionsTotal) < leafAt首Less一零:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, leafAt首Less一零])
		if is_even(leafAt首Less一零):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			dimension: int = 一
			if bit_test(leafAt首Less一零, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])
				listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2])])
			dimension = 二
			if bit_test(leafAt首Less一零, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])
				if 1 < dimensionNearestTail(leafAt首Less一零):
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2])])
			dimension = 三
			if bit_test(leafAt首Less一零, dimensionIndex(dimension)):
				if 1 < dimensionNearestTail(leafAt首Less一零):
					listRemoveLeaves.extend([dimension])
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2])])
				if dimensionNearestTail(leafAt首Less一零) < dimensionIndex(dimension):
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])

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

# ======== use leafAt二 to exclude a `leaf` from `pile` ===================
# {3: [2],
# 5: [2, 4, 7, 35, 37],
# 6: [2, 4, 7, 35, 37, 38],
# 9: [2, 8, 11, 19, 25, 41, 49, 59],
# 10: [2, 4, 7, 11, 13, 14, 35, 41, 42, 44],
# 15: [2, 4, 7, 11, 13, 14, 37, 38, 41],
# 17: [8, 16, 19, 41, 49, 56],
# 18: [2, 4, 7, 19, 22, 35, 49, 50],
# 23: [2, 4, 19, 21, 22, 37, 38, 49, 50, 52],
# 27: [2, 11, 19, 25, 26, 41, 42, 49, 56, 59],
# 29: [2, 7, 8, 13, 19, 21, 25, 28, 31, 35, 41, 44, 49, 52, 56, 61],
# 34: [2, 16, 35],
# 39: [2, 4, 16, 35, 37, 38, 49],
# 43: [2, 16, 35, 41, 42, 49, 59],
# 45: [16, 37, 41, 44, 47, 49, 61],
# 51: [2, 16, 35, 49, 50, 56],
# 53: [16, 37, 49, 52, 55, 56],
# 57: [16, 25, 41, 49, 56, 59]}
	listRemoveLeaves = []

	if is_even(leafAt二):
		listRemoveLeaves.extend([一, leafAt二 + 零, 首零(state.dimensionsTotal)+一+零])
	if is_odd(leafAt二):
		listRemoveLeaves.extend([leafAt二 - 零])
		if 首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零一(state.dimensionsTotal)+零])
		if 首零(state.dimensionsTotal) < leafAt二:
			listRemoveLeaves.extend([首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)+零])
			dimension = 一
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])
			dimension = 二
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])
			dimension = 三
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])
			dimension = 四
			if bit_test(leafAt二, dimensionIndex(dimension)) and (leafAt二.bit_length() > 5):
				listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])

	# print(leafAt二, sorted(set(listRemoveLeaves)))

# (0, 32, 48, 56, 60, 62, 63)
# (0, 16, 24, 28, 30, 31)

# ruff: noqa
# 17, 18, 20, 24, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60
# 010001 17: [2, 16],
# 010010 18: [2, 8, 16, 19, 26, 41, 49, 50, 56],
# 010100 20: [4, 8, 16, 21, 28, 44, 49, 52, 56],
# 011000 24: [8, 16, 25, 49, 56],
# 100010 34: [2, 4, 7, 35, 49, 50],
# 100100 36: [2, 4, 16, 35, 37, 38, 49, 50, 52],
# 100111 39: [2, 4, 7, 35, 37, 38, 49, 55],
# 101000 40: [8, 16, 25, 41, 49, 56],
# 101011 43: [2, 35, 42, 49, 59],
# 101101 45: [2, 11, 35, 37, 41, 44, 49, 61],
# 101110 46: [2, 4, 8, 16, 31, 35, 37, 38, 42, 44, 47, 49, 50, 62],
# 110000 48: [16, 49],
# 110011 51: [2, 19, 35, 49, 50],
# 110101 53: [2, 19, 35, 37, 49, 52],
# 110110 54: [2, 4, 16, 21, 35, 37, 38, 49, 50, 52, 55],
# 111001 57: [2, 41, 49, 56],
# 111010 58: [2, 16, 35, 42, 49, 50, 56, 59],
# 111100 60: [4, 8, 16, 37, 44, 49, 52, 56, 61]

# Surplus:
# 18: 41
# 20: 44
# 34: 4,7
# 36: 2, 35, 38
# 39: 7
# 40: 25
# 45: 11, 35
# 46: 8, 31
# 51: 19
# 53: 19, 35
# 54: 21
# 60: 8


# ======== use leafAt首Less二 to exclude a `leaf` from `pile` ===================
	dimensionHead: int = dimensionNearest首(leafAt首Less二)
	dimensionTail: int = dimensionNearestTail(leafAt首Less二)

	if leafAt首Less二 != 首一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零])

	if is_odd(leafAt首Less二):
		listRemoveLeaves.extend([一, leafAt首Less二 - 零, leafAt首Less二 - state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt首Less二))]])
	if is_even(leafAt首Less二):
		listRemoveLeaves.extend([leafAt首Less二 + 零, state.productsOfDimensions[dimensionTail], leafAt首Less二 - state.productsOfDimensions[dimensionTail]])

		if leafAt首Less二 != 首零(state.dimensionsTotal)+一:
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])

		if leafAt首Less二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less二 + state.productsOfDimensions[dimensionNearest首(leafAt首Less二) + 1]])
			dimension = 三
			if not bit_test(leafAt首Less二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension, leafAt首Less二 + dimension, state.sumsOfProductsOfDimensionsNearest首[dimensionIndex(dimension)]])

	if dimensionTail == 3:
		listRemoveLeaves.extend([state.sumsOfProductsOfDimensionsNearest首[3]])

	dimension = 二
	if bit_test(leafAt首Less二, dimensionIndex(dimension)):
		listRemoveLeaves.extend([leafAt首Less二 - dimension])
		if (is_even(leafAt首Less二)) or (is_odd(leafAt首Less二) and (dimensionIndex(dimension) < Z0Z_0NearestTail(state, leafAt首Less二))):
			listRemoveLeaves.extend([dimension])

	dimension = 三
	if bit_test(leafAt首Less二, dimensionIndex(dimension)):
		listRemoveLeaves.extend([leafAt首Less二 - dimension])

	if 首零(state.dimensionsTotal) < leafAt首Less二:
		dimension = 一
		if bit_test(leafAt首Less二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])
		dimension = 二
		if bit_test(leafAt首Less二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])
		dimension = 四
		if not bit_test(leafAt首Less二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([leafAt首Less二 + dimension])
		if bit_test(leafAt首Less二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([leafAt首Less二 - dimension])
	del dimensionHead, dimensionTail

# ======= Miscellaneous exclusions ===================
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

