from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from mapFolding import decreasing
from mapFolding._e import (
	dimensionIndex, dimensionNearestTail, dimensionNearest首, dimensionSecondNearest首, exclude, getDictionaryPileRanges,
	getLeaf, getLeavesCreaseBack, getLeavesCreaseNext, getSumsOfProductsOfDimensionsNearest首, leafInSubHyperplane, ptount,
	Z0Z_0NearestTail, 一, 三, 二, 五, 四, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.Z0Z_analysisPython.workBenchPatternFinder import getExcludedLeaves
from more_itertools import last
from operator import getitem

# ======= Domain-based subroutines for analyzing a specific `pile`. =======

def pinPile首零Less零AfterFourthOrder(state: EliminationState) -> list[int]:
	"""All fourth-order piles must be pinned or you will get an error."""
	leafAt一:			int = raiseIfNone(getLeaf(state.leavesPinned, 			一))
	leafAt首Less一:		int = raiseIfNone(getLeaf(state.leavesPinned, state.首 - 一))
	leafAt一零:			int = raiseIfNone(getLeaf(state.leavesPinned, 			(一+零)))
	leafAt首Less一零:	int = raiseIfNone(getLeaf(state.leavesPinned, state.首 - (一+零)))
	leafAt二:			int = raiseIfNone(getLeaf(state.leavesPinned, 			二))
	leafAt首Less二:		int = raiseIfNone(getLeaf(state.leavesPinned, state.首 - (二)))

	dictionaryPileRanges: dict[int, tuple[int, ...]] = getDictionaryPileRanges(state)
	listRemoveLeaves: list[int] = []

# ======== use `leafAt一` to exclude a `leaf` from `pile` ===================

	pileExcluder: int = 一
	for dimension, leaf in enumerate(dictionaryPileRanges[pileExcluder]):
		if leaf == leafAt一:
			if dimension < state.dimensionsTotal - 2:
				listRemoveLeaves.extend([一, 首零(state.dimensionsTotal) + leafAt一])
			if 0 < dimension < state.dimensionsTotal - 2:
				listRemoveLeaves.extend([一 + leafAt一])
			if dimension == 1:
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + leafAt一 + 零])
			if dimension == state.dimensionsTotal - 2:
				listRemoveLeaves.extend([首一(state.dimensionsTotal), 首一(state.dimensionsTotal) + leafAt一])
	del pileExcluder

# ------- Use information from other piles to select which leaves to exclude. -------
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less一 + 零])
	if dimensionNearest首(leafAt一) < state.dimensionsTotal - 3:
		listRemoveLeaves.extend([一, leafAt首Less一 + 一])

# ======== use `leafAt首Less一` to exclude a `leaf` from `pile` ===================

	pileExcluder = state.首 - 一
	for dimension, leaf in enumerate(dictionaryPileRanges[pileExcluder]):
		if leaf == leafAt首Less一:
			if dimension == 0:
				listRemoveLeaves.extend([一])
			if dimension < state.dimensionsTotal - 2:
				listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAt首Less一])
			if 0 < dimension < state.dimensionsTotal - 2:
				listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimension), 首一(state.dimensionsTotal) + leafAt首Less一 - getitem(state.sumsOfProductsOfDimensions, dimension)])
			if 0 < dimension < state.dimensionsTotal - 3:
				listRemoveLeaves.extend([零 + leafAt首Less一])
			if 0 < dimension < state.dimensionsTotal - 1:
				listRemoveLeaves.extend([首一(state.dimensionsTotal)])
	del pileExcluder

# ------- Use information from other piles to decide whether to exclude some leaves. -------
	if (leafAt一 == 首二(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])

# ======== use `leafAt一零` to exclude a `leaf` from `pile` ===================
# NOTE a leaf in pile一零 does not have leafCrease in the pile-range of pile首零Less零, but `leafInSubHyperplane(leafAt一零)` does
# have leafCrease in the pile-range of pile首零Less零. `ptount` uses leafInSubHyperplane. I wrote this code block long before I
# understood this.

# NOTE this section relies on the exclusions in `leafAt一` and `leafAt首Less一` to exclude some leaves.

	listRemoveLeaves.extend([leafAt一零])
	if leafAt一零 == 三+二+零:
		listRemoveLeaves.extend([二+一+零, 首零(state.dimensionsTotal)+二+零])
	if leafAt一零 == 首一(state.dimensionsTotal)+二+零:
		listRemoveLeaves.extend([首二(state.dimensionsTotal), leafAt一零 + getitem(state.productsOfDimensions, raiseIfNone(dimensionSecondNearest首(leafAt一零))), leafAt一零 + getitem(state.sumsOfProductsOfDimensions, raiseIfNone(dimensionSecondNearest首(leafAt一零)) + 1), 首零一二(state.dimensionsTotal)])
	if leafAt一零 == 首一二(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal)+(一+零), last(getLeavesCreaseBack(state, leafInSubHyperplane(leafAt一零)))])
	if leafAt一零 == 首零一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	if is_odd(leafAt一零):
		dimensionHeadSecond: int = raiseIfNone(dimensionSecondNearest首(leafAt一零))
		indexBy首Second: int = dimensionHeadSecond * decreasing + decreasing # Are you confused and/or annoyed by this? Blame Python. (Or figure out a better formula.)
		listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionHeadSecond)])
		if leafAt一零 < 首零(state.dimensionsTotal):
			sumsOfProductsOfDimensionsNearest首InSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, state.dimensionsTotal, state.dimensionsTotal - 1)
			listRemoveLeaves.extend([一, leafAt一零 + getitem(state.sumsOfProductsOfDimensions, (state.dimensionsTotal - 1)), leafAt一零 + getitem(sumsOfProductsOfDimensionsNearest首InSubHyperplane, indexBy首Second)])
			if dimensionHeadSecond == 2:
				listRemoveLeaves.extend([getitem(state.sumsOfProductsOfDimensions, dimensionHeadSecond) + getitem(state.productsOfDimensions, dimensionNearest首(leafAt一零)), getitem(state.sumsOfProductsOfDimensions, dimensionHeadSecond) + 首零(state.dimensionsTotal)])
			if dimensionHeadSecond == 3:
				listRemoveLeaves.extend([一 + leafAt一零 + getitem(state.productsOfDimensions, (state.dimensionsTotal - 1))])
		if 首零(state.dimensionsTotal) < leafAt一零:
			listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, getitem(state.productsOfDimensions, (dimensionNearest首(leafAt一零) - 1))])

# ======== use `leafAt首Less一零` to exclude a `leaf` from `pile` ===================
# NOTE a leaf in pile首Less一零 does not have leafCrease in the pile-range of pile首零Less零, but `leafInSubHyperplane(leafAt首
# Less一零)` does have leafCrease in the pile-range of pile首零Less零. `ptount` uses leafInSubHyperplane. I wrote this code block
# long before I understood this.

# NOTE This section could be "modernized" to be more similar to `leafAt一零`, which used to have `comebackOffset`, too.

	listRemoveLeaves.extend([leafAt首Less一零])

	if 首零(state.dimensionsTotal) < leafAt首Less一零:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零])
		if is_even(leafAt首Less一零):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			dimension: int = 一
			if bit_test(leafAt首Less一零, dimensionIndex(dimension)):
				listRemoveLeaves.extend([ dimension, 首零(state.dimensionsTotal) + dimension + 零, state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2]), leafAt首Less一零 - dimension - getitem(state.sumsOfProductsOfDimensions, (dimensionIndex(dimension) + 1)), ])
			dimension = 二
			if bit_test(leafAt首Less一零, dimensionIndex(dimension)):
				listRemoveLeaves.extend([ dimension, 首零(state.dimensionsTotal) + dimension + 零 ])
				if 1 < dimensionNearestTail(leafAt首Less一零):
					listRemoveLeaves.extend([ state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2]), ])
				else: # NOTE IDK and IDC why this works, but it does.
					listRemoveLeaves.extend([getitem(tuple(getLeavesCreaseBack(state, leafInSubHyperplane(leafAt首Less一零))), dimensionIndex(dimension)) - 零])
			dimension = 三
			if bit_test(leafAt首Less一零, dimensionIndex(dimension)):
				if 1 < dimensionNearestTail(leafAt首Less一零):
					listRemoveLeaves.extend([dimension])
					listRemoveLeaves.extend([state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2])])
				if dimensionNearestTail(leafAt首Less一零) < dimensionIndex(dimension):
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])
			sheepOrGoat = 0
			shepherdOfDimensions: int = int(bit_flip(0, state.dimensionsTotal - 5))
			if (leafAt首Less一零//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveLeaves.extend([二])
				sheepOrGoat: int = ptount(leafAt首Less一零//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset: int = state.productsOfDimensions[dimensionNearest首(leafAt首Less一零)] - 二
					listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt首Less一零))] - 二
					listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])
		if is_odd(leafAt首Less一零):
			listRemoveLeaves.extend([一])
			if leafAt首Less一零 & bit_mask(4) == 0b001001:
				listRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAt首Less一零)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = state.productsOfDimensions[dimensionNearest首(leafAt首Less一零)] - 一
				listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt首Less一零))] - 一
				listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])

# ------- Use information from other piles to decide whether to exclude some leaves. -------
	if (leafAt一 == 一+零) and (leafAt首Less一零 != next(getLeavesCreaseBack(state, 首零(state.dimensionsTotal)+零))):
		listRemoveLeaves.append(首一(state.dimensionsTotal))

# NOTE Above this line, all exclusions based on only one leaf in a pile are covered. 😊
# ======== use leafAt二 to exclude a `leaf` from `pile` ===================
# NOTE Below this line, abandon all hope, the who code here. 😈
	listRemoveLeaves = []
	dimensionHead: int = dimensionNearest首(leafAt二)
# 000011  3  [2]
# 000101  5  [2, 4, 7, 35, 37]
# 000110  6  [2, 4, 7, 35, 37, 38]
# 001001  9  [2, 8, 11, 19, 25, 41, 49, 59]
# 001010  10 [2, 4, 7, 11, 13, 14, 35, 41, 42, 44]
# 001111  15 [2, 4, 7, 11, 13, 14, 37, 38, 41]

# 010001  17 [			8, 16,	19, 41, 						49, 		56]
# 010010  18 [2, 4, 7, 			19, 22, 35, 					49, 50]
# 010111  23 [2, 4, 			19, 21, 22, 37, 38, 			49, 50, 52]
# 011011  27 [2, 		11,		19, 25, 26, 41, 42, 			49, 		56, 59]
# 011101  29 [2, 	7, 8, 13, 	19, 21, 25, 28, 31, 35, 41, 44, 49, 	52, 56, 	61]

# 100010  34 [2, *16, 35]

# 100111  39 [2, 4, 16, 	35, 37, 38, 		49]
# 101011  43 [2, 	16, 	35, 		41, 42, 49, *59]
# 101101  45 [		16, 		37, 	41, 44, 47, 49, 61]
# 110011  51 [2, 	16, 	35, 			49, 50, *56]
# 110101  53 [		16, 		37, 		49, 52, 55, 56]
# 111001  57 [		16, 25, 			41, 49, 56, 59]

	creaseNextAt二: tuple[int, ...] = tuple(getLeavesCreaseNext(state, leafAt二))
	listIndicesCreaseNextToKeep: list[int] = []

	if (二 < leafAt二 < 首一(state.dimensionsTotal)-零):
		listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal)])

		dimension = 一
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				leafAt二 + 首零(state.dimensionsTotal) + dimension
			])

		if not bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				leafAt二 + 首零(state.dimensionsTotal) - dimension
			])

		if is_odd(leafAt二):
			dimension = 三
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([
					leafAt二 + 首零(state.dimensionsTotal) + dimension
				])

				dimension = 四
				if not bit_test(leafAt二, dimensionIndex(dimension)):
					listRemoveLeaves.extend([
						leafAt二 + 首零(state.dimensionsTotal) - dimension
					])

	if ((首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal)) and raiseIfNone(dimensionSecondNearest首(leafAt二)) != 2):
		listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal)])
		if is_odd(leafAt二):
			dimension = 二
			if not bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([
					leafAt二 + 首零(state.dimensionsTotal) - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension)),
				])

			dimension = 三
			if not bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([
					leafAt二 + 首零(state.dimensionsTotal) - dimension,
					leafAt二 + 首零(state.dimensionsTotal) + getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension)),
				])

			dimension = 四
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([
					leafAt二 - dimension
				])


	if is_even(leafAt二):
		listIndicesCreaseNextToKeep.extend(range(state.dimensionsTotal - dimensionHead + 1, state.dimensionsTotal - 1))

		dimension = 一
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				dimension,
				首零(state.dimensionsTotal) + dimension + 零
			])
			# print(leafAt二.__format__('06b'), leafAt二, listRemoveLeaves)

		dimension = 二
		if not bit_test(leafAt二, dimensionIndex(dimension)):
			listIndicesCreaseNextToKeep.append(creaseNextAt二.index(state.productsOfDimensions[dimensionHead]))

		listRemoveLeaves.extend([
				leafAt二 + 零,
				leafAt二 + 首零(state.dimensionsTotal),
				leafAt二 + state.sumsOfProductsOfDimensions[state.dimensionsTotal-1],
				state.productsOfDimensions[dimensionHead] + (一+零),
			])

		if leafAt二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([
				二
			])

	if is_odd(leafAt二):
		# --- all odd ---
		if dimensionNearestTail(leafAt二 - 1) == 1:
			listRemoveLeaves.extend([
				一
			])

		if leafInSubHyperplane(leafAt二) == state.sumsOfProductsOfDimensions[3]:
			listRemoveLeaves.extend([
				二
			])

		dimension = 零
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				dimension,
				leafAt二 - dimension,
				首零(state.dimensionsTotal) + dimension + 零
			])

		dimension = 二
		if not bit_test(leafAt二, dimensionIndex(dimension)):
			listIndicesCreaseNextToKeep.append(dimensionIndex(dimension))

		dimension = 三
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				leafAt二 - dimension,
				首零(state.dimensionsTotal) + dimension + 零
			])
		if not bit_test(leafAt二, dimensionIndex(dimension)):
			listIndicesCreaseNextToKeep.append(dimensionIndex(dimension))

			dimension = 四
			if not bit_test(leafAt二, dimensionIndex(dimension)):
				listIndicesCreaseNextToKeep.append(dimensionIndex(dimension))

		dimension = 四
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				首零(state.dimensionsTotal) + dimension + 零,
			])

		dimension = 五
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				首一(state.dimensionsTotal),
				首零一(state.dimensionsTotal)+零
			])

		# --- small ---
		if leafAt二 < 首一(state.dimensionsTotal):
			listRemoveLeaves.extend([
				一
			])

		# --- medium ---
		if 首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([
				leafAt二 + state.sumsOfProductsOfDimensions[state.dimensionsTotal-2],
				首一(state.dimensionsTotal)+(一+零),
			])

		#  --- large ---
		if 首零(state.dimensionsTotal) < leafAt二:
			dimension = 二
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([
					leafAt二 - dimension,
					首零(state.dimensionsTotal) + dimension + 零
				])

			dimension = 四
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([
					dimension,
					leafAt二 - dimension,
					首零(state.dimensionsTotal) + dimension + 零,
					首零一二(state.dimensionsTotal)
				])

	listRemoveLeaves.extend(exclude(creaseNextAt二, listIndicesCreaseNextToKeep))

	excludedLeaves: list[int] = getExcludedLeaves(state, pileTarget=31, groupByLeavesAtPiles=(pileExcluder := 二,))[leafAtPileExcluder := raiseIfNone(getLeaf(state.leavesPinned, pileExcluder))]
	if surplus := sorted(set(excludedLeaves).difference(listRemoveLeaves)):
		print(leafAtPileExcluder.__format__('06b'), leafAtPileExcluder, surplus)
		# print(leafAtPileExcluder, [ss - leafAtPileExcluder for ss in surplus])

# ======== use leafAt首Less二 to exclude a `leaf` from `pile` ===================
	# listRemoveLeaves = []
	dimensionHead: int = dimensionNearest首(leafAt首Less二)
	dimensionTail: int = dimensionNearestTail(leafAt首Less二)
	creaseBackAt首Less二: tuple[int, ...] = tuple(getLeavesCreaseBack(state, leafAt首Less二))
	creaseNextAt首Less二: tuple[int, ...] = tuple(getLeavesCreaseNext(state, leafAt首Less二))
	listIndicesCreaseNextToKeep: list[int] = []

	# --- only 17 allows 49 ---
	if leafAt首Less二 != 首一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零])

	# --- odd and even ---

	dimension = 二
	if bit_test(leafAt首Less二, dimensionIndex(dimension)):
		listRemoveLeaves.extend([leafAt首Less二 - dimension])
		if (is_even(leafAt首Less二)
		or (is_odd(leafAt首Less二) and (dimensionIndex(dimension) < Z0Z_0NearestTail(state, leafAt首Less二)))):
			listRemoveLeaves.extend([dimension])

	dimension = 三
	if bit_test(leafAt首Less二, dimensionIndex(dimension)):
		listRemoveLeaves.extend([leafAt首Less二 - dimension])

	if dimensionTail == 3:
		listRemoveLeaves.extend([state.sumsOfProductsOfDimensionsNearest首[3]])

	# --- large ---
	if 首零(state.dimensionsTotal) < leafAt首Less二:
		dimension = 一
		if bit_test(leafAt首Less二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				dimension,
				首零(state.dimensionsTotal) + dimension + 零
			])

		dimension = 二
		if bit_test(leafAt首Less二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				首零(state.dimensionsTotal) + dimension + 零
			])

		dimension = 四
		if bit_test(leafAt首Less二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				leafAt首Less二 - dimension
			])
		if not bit_test(leafAt首Less二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([
				leafAt首Less二 + dimension
			])

	if is_odd(leafAt首Less二):
		listRemoveLeaves.extend([
			一,
			leafAt首Less二 - 零,
			leafAt首Less二 - state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt首Less二))]
		])

	if is_even(leafAt首Less二):
		listRemoveLeaves.extend([
			leafAt首Less二 + 零,
			state.productsOfDimensions[dimensionTail],
			leafAt首Less二 - state.productsOfDimensions[dimensionTail]
		])

		if leafAt首Less二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([
				首一(state.dimensionsTotal),
				leafAt首Less二 + state.productsOfDimensions[dimensionNearest首(leafAt首Less二) + 1]
			])

			dimension = 三
			if not bit_test(leafAt首Less二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([
					dimension,
					leafAt首Less二 + dimension,
					state.sumsOfProductsOfDimensionsNearest首[dimensionIndex(dimension)]
				])

		if leafAt首Less二 != 首零(state.dimensionsTotal)+一:
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])

	del dimensionHead, dimensionTail

	listExcluded = getExcludedLeaves(state, pileTarget=31, groupByLeavesAtPiles=(60,))[leafAt首Less二]
	surplus = sorted(set(listExcluded).difference(listRemoveLeaves))
	surplusInNext = sorted(set(surplus).intersection(creaseNextAt首Less二))
	surplusInBack = sorted(set(surplus).intersection(creaseBackAt首Less二))
	# print(leafAt首Less二.__format__('06b'), leafAt首Less二, creaseBackAt首Less二, creaseNextAt首Less二, sep='\t')
	# if surplus:
	# 	print(leafAt首Less二, [ss-leafAt首Less二 for ss in surplus], sep='\t')

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
# 110011 51: [2, 19, 35, 49, 50],

# 101011 43: [2, 35, 42, 49, 59],
# 101101 45: [2, 11, 35, 37, 41, 44, 49, 61],
# 101110 46: [2, 4, 8, 16, 31, 35, 37, 38, 42, 44, 47, 49, 50, 62],
# 110000 48: [16, 49],
# 110101 53: [2, 19, 35, 37, 49, 52],
# 110110 54: [2, 4, 16, 21, 35, 37, 38, 49, 50, 52, 55],
# 111001 57: [2, 41, 49, 56],
# 111010 58: [2, 16, 35, 42, 49, 50, 56, 59],
# 111100 60: [4, 8, 16, 37, 44, 49, 52, 56, 61]

	return sorted(set(dictionaryPileRanges[state.pile]).difference(set(listRemoveLeaves)))

