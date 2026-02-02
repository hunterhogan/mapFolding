from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from mapFolding import decreasing, zeroIndexed
from mapFolding._e import (
	dimensionIndex, dimensionNearestTail, dimensionNearest首, dimensionsConsecutiveAtTail, dimensionSecondNearest首,
	DOTgetPileIfLeaf, getDictionaryPileRanges, getLeavesCreaseAnte, getLeavesCreasePost,
	getSumsOfProductsOfDimensionsNearest首, howManyDimensionsHaveOddParity, leafInSubHyperplane, ptount, 一, 三, 二, 五, 四, 零,
	首一, 首一二, 首二, 首零, 首零一, 首零一二)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import exclude, notLeafOriginOrLeaf零
from more_itertools import last
from operator import getitem, neg

#======== Domain-based subroutines for analyzing a specific `pile`. =======

def pinPile零Ante首零AfterDepth4(state: EliminationState) -> list[int]:
	"""All fourth-order piles must be pinned or you will get an error.

	Unless I've made a mistake:

	I have made a rule to exclude a leaf from pile 零Ante首零
		for all piles in the first four orders (pile <= 4 or pile >= 首 - 4),
			for all leaves in each pile's pile-range
				that exclude a leaf from pile 零Ante首零.

	If I were to figure out the last few cases, it would remove 23 surplus dictionaries.
		10 of 23 dictionaries
			if leafAt二 == 15:
				listRemoveLeaves.extend([38])
		2 of 23 dictionaries
			if leafAt二 == 9:
				listRemoveLeaves.extend([19])
		2 of 23 dictionaries
				listRemoveLeaves.extend([59])
		3 of 23 dictionaries
			if leafAt二 == 23:
				listRemoveLeaves.extend([50])
		4 of 23 dictionaries
			if leafAt二 == 29:
				listRemoveLeaves.extend([7])
		2 of 23 dictionaries
				listRemoveLeaves.extend([35])

	But I would still have 1312 surplus dictionaries.

	Therefore, if I continue to pin pile 零Ante首零, I should probably focus on different strategies.
	"""
	leafAt一:			int = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace,	   一))
	leafAt一Ante首:		int = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(一)+state.首))
	leafAt一零:			int = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace,	  (一+零)))
	leafAt零一Ante首:	int = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(零+一)+state.首))
	leafAt二:			int = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace,	   二))
	leafAt二Ante首:		int = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(二)+state.首))

	dictionaryPileRanges: dict[int, tuple[int, ...]] = getDictionaryPileRanges(state)
	listRemoveLeaves: list[int] = []

#========= use `leafAt一` to exclude a `leaf` from `pile` ===================

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

#-------- Use information from other piles to select which leaves to exclude. -------
	if leafAt一 == (零)+首零(state.dimensionsTotal):
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt一Ante首 + 零])
	if dimensionNearest首(leafAt一) < state.dimensionsTotal - 3:
		listRemoveLeaves.extend([一, leafAt一Ante首 + 一])

#========= use `leafAt一Ante首` to exclude a `leaf` from `pile` ===================

	pileExcluder = neg(一)+state.首
	for dimension, leaf in enumerate(dictionaryPileRanges[pileExcluder]):
		if leaf == leafAt一Ante首:
			if dimension == 0:
				listRemoveLeaves.extend([一])
			if dimension < state.dimensionsTotal - 2:
				listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAt一Ante首])
			if 0 < dimension < state.dimensionsTotal - 2:
				listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimension), 首一(state.dimensionsTotal) + leafAt一Ante首 - getitem(state.sumsOfProductsOfDimensions, dimension)])
			if 0 < dimension < state.dimensionsTotal - 3:
				listRemoveLeaves.extend([零 + leafAt一Ante首])
			if 0 < dimension < state.dimensionsTotal - 1:
				listRemoveLeaves.extend([首一(state.dimensionsTotal)])
	del pileExcluder

#-------- Use information from other piles to decide whether to exclude some leaves. -------
	if (leafAt一 == (零)+首二(state.dimensionsTotal)) and (leafAt一Ante首 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])

#========= use `leafAt一零` to exclude a `leaf` from `pile` ===================
# NOTE a leaf in pile一零 does not have leafCrease in the pile-range of pile零Ante首零, but `leafInSubHyperplane(leafAt一零)` does
# have leafCrease in the pile-range of pile零Ante首零. `ptount` uses leafInSubHyperplane. I wrote this code block long before I
# understood this.

# NOTE this section relies on the exclusions in `leafAt一` and `leafAt一Ante首` to exclude some leaves.

	listRemoveLeaves.extend([leafAt一零])
	if leafAt一零 == 三+二+零:
		listRemoveLeaves.extend([二+一+零, (零+二)+首零(state.dimensionsTotal)])
	if leafAt一零 == (零+二)+首一(state.dimensionsTotal):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), leafAt一零 + getitem(state.productsOfDimensions, raiseIfNone(dimensionSecondNearest首(leafAt一零))), leafAt一零 + getitem(state.sumsOfProductsOfDimensions, raiseIfNone(dimensionSecondNearest首(leafAt一零)) + 1), 首零一二(state.dimensionsTotal)])
	if leafAt一零 == (零)+首一二(state.dimensionsTotal):
		listRemoveLeaves.extend([首一(state.dimensionsTotal)+(一+零), last(getLeavesCreaseAnte(state, leafInSubHyperplane(leafAt一零)))])
	if leafAt一零 == (零)+首零一(state.dimensionsTotal):
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
			listRemoveLeaves.extend([(零)+首零一(state.dimensionsTotal), getitem(state.productsOfDimensions, (dimensionNearest首(leafAt一零) - 1))])

#========= use `leafAt零一Ante首` to exclude a `leaf` from `pile` ===================
# NOTE a leaf in pile首Less一零 does not have leafCrease in the pile-range of pile零Ante首零, but `leafInSubHyperplane(leafAt首
# Less一零)` does have leafCrease in the pile-range of pile零Ante首零. `ptount` uses leafInSubHyperplane. I wrote this code block
# long before I understood this.

# NOTE This section could be "modernized" to be more similar to `leafAt一零`, which used to have `comebackOffset`, too.

	listRemoveLeaves.extend([leafAt零一Ante首])

	if 首零(state.dimensionsTotal) < leafAt零一Ante首:
		listRemoveLeaves.extend([(零)+首零一(state.dimensionsTotal)])
		if is_even(leafAt零一Ante首):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			dimension: int = 一
			if bit_test(leafAt零一Ante首, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零, state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2]), leafAt零一Ante首 - dimension - getitem(state.sumsOfProductsOfDimensions, (dimensionIndex(dimension) + 1))])
			dimension = 二
			if bit_test(leafAt零一Ante首, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])
				if 1 < dimensionNearestTail(leafAt零一Ante首):
					listRemoveLeaves.extend([state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2])])
				else: # NOTE IDK and IDC why this works, but it does.
					listRemoveLeaves.extend([getitem(tuple(getLeavesCreaseAnte(state, leafInSubHyperplane(leafAt零一Ante首))), dimensionIndex(dimension)) - 零])
			dimension = 三
			if bit_test(leafAt零一Ante首, dimensionIndex(dimension)):
				if 1 < dimensionNearestTail(leafAt零一Ante首):
					listRemoveLeaves.extend([dimension])
					listRemoveLeaves.extend([state.首 - sum(state.productsOfDimensions[dimensionIndex(dimension): state.dimensionsTotal - 2])])
				if dimensionNearestTail(leafAt零一Ante首) < dimensionIndex(dimension):
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])
			sheepOrGoat = 0
			shepherdOfDimensions: int = int(bit_flip(0, state.dimensionsTotal - 5))
			if (leafAt零一Ante首//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveLeaves.extend([二])
				sheepOrGoat: int = ptount(leafAt零一Ante首//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset: int = state.productsOfDimensions[dimensionNearest首(leafAt零一Ante首)] - 二
					listRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt零一Ante首))] - 二
					listRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
		if is_odd(leafAt零一Ante首):
			listRemoveLeaves.extend([一])
			if leafAt零一Ante首 & bit_mask(4) == 0b001001:
				listRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAt零一Ante首)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = state.productsOfDimensions[dimensionNearest首(leafAt零一Ante首)] - 一
				listRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt零一Ante首))] - 一
				listRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])

#-------- Use information from other piles to decide whether to exclude some leaves. -------
	if (leafAt一 == 一+零) and (leafAt零一Ante首 != next(getLeavesCreaseAnte(state, (零)+首零(state.dimensionsTotal)))):
		listRemoveLeaves.append(首一(state.dimensionsTotal))

# NOTE Above this line, all exclusions based on only one leaf in a pile are covered. 😊
#========= use leafAt二 to exclude a `leaf` from `pile` ===================
# NOTE Below this line, abandon all hope, the who code here. 😈

	dimensionHead: int = dimensionNearest首(leafAt二)
	creasePostAt二: tuple[int, ...] = tuple(getLeavesCreasePost(state, leafAt二))
	listIndicesCreasePostToKeep: list[int] = []

	if (二 < leafAt二 < neg(零)+首一(state.dimensionsTotal)):
		listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal)])

		dimension = 一
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) + dimension])

		if not bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) - dimension])

		if is_odd(leafAt二):
			dimension = 三
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) + dimension])

				dimension = 四
				if not bit_test(leafAt二, dimensionIndex(dimension)):
					listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) - dimension])

	if ((首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal)) and raiseIfNone(dimensionSecondNearest首(leafAt二)) != 2):
		listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal)])

		if is_odd(leafAt二):
			dimension = 二
			if not bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])

			dimension = 三
			if not bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([leafAt二 + 首零(state.dimensionsTotal) - dimension, leafAt二 + 首零(state.dimensionsTotal) + getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])

			dimension = 四
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([leafAt二 - dimension])

	if is_even(leafAt二):
		listIndicesCreasePostToKeep.extend(range(state.dimensionsTotal - dimensionHead + 1, (state.dimensionsTotal - zeroIndexed)))

		listRemoveLeaves.extend([
				leafAt二 + 零, leafAt二 + 首零(state.dimensionsTotal), leafAt二 + getitem(state.sumsOfProductsOfDimensions, (state.dimensionsTotal-1)), getitem(state.productsOfDimensions, dimensionHead) + (一+零)])

		dimension = 一
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])

		dimension = 二
		if not bit_test(leafAt二, dimensionIndex(dimension)):
			listIndicesCreasePostToKeep.append(creasePostAt二.index(state.productsOfDimensions[dimensionHead]))

		if leafAt二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionIndex(二)), getitem(state.sumsOfProductsOfDimensions, (dimensionIndex(二) + 1))])

		dimension = 四
		if (not bit_test(leafAt二, dimensionIndex(dimension))) and (首零(state.dimensionsTotal) < leafAt二):
			listRemoveLeaves.extend([getitem(state.productsOfDimensions, dimensionIndex(dimension))])

# NOTE 1) I am sure this concept has validity. 2) I am sure there is a more accurate computation for it.
		zerosAtThe首 = 2
		if state.dimensionsTotal - zeroIndexed - dimensionHead == zerosAtThe首:
			sumsOfProductsOfDimensionsNearest首InSubSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, state.dimensionsTotal, state.dimensionsTotal - zerosAtThe首)
			addendForUnknownReasons: int = -1
			leavesWeDontWant: list[int] = [aLeaf + addendForUnknownReasons for aLeaf in filter(notLeafOriginOrLeaf零, sumsOfProductsOfDimensionsNearest首InSubSubHyperplane)]
			listRemoveLeaves.extend(leavesWeDontWant)

	if is_odd(leafAt二):

		if dimensionNearestTail(leafAt二 - 1) == 1:
			listRemoveLeaves.extend([一])

		if leafInSubHyperplane(leafAt二) == state.sumsOfProductsOfDimensions[3]:
			listRemoveLeaves.extend([二])

		dimension = 零
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([dimension, leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零])

		dimension = 二
		if not bit_test(leafAt二, dimensionIndex(dimension)):
			listIndicesCreasePostToKeep.append(dimensionIndex(dimension))

		if bit_test(leafAt二, dimensionIndex(dimension)) and bit_test(leafAt二, dimensionIndex(一)):
			listRemoveLeaves.extend([leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零])

		dimension = 三
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零])

		if not bit_test(leafAt二, dimensionIndex(dimension)):
			listIndicesCreasePostToKeep.append(dimensionIndex(dimension))

			dimension = 四
			if not bit_test(leafAt二, dimensionIndex(dimension)):
				listIndicesCreasePostToKeep.append(dimensionIndex(dimension))

		dimension = 四
		if bit_test(leafAt二, dimensionIndex(dimension)):

			dimensionBonus: int = 零
			if bit_test(leafAt二, dimensionIndex(dimensionBonus)):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + dimensionBonus])

			dimensionBonus = 二
			if bit_test(leafAt二, dimensionIndex(dimensionBonus)):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + dimensionBonus])

			dimensionBonus = 三
			if bit_test(leafAt二, dimensionIndex(dimensionBonus)):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + dimensionBonus])

		dimension = 五
		if bit_test(leafAt二, dimensionIndex(dimension)):
			listRemoveLeaves.extend([首一(state.dimensionsTotal),(零)+首零一(state.dimensionsTotal)])

		# --- small ---
		if leafAt二 < 首一(state.dimensionsTotal):
			listRemoveLeaves.extend([一])

		# --- medium ---
		if 首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([leafAt二 + getitem(state.sumsOfProductsOfDimensions, (state.dimensionsTotal - 2)), 首一(state.dimensionsTotal)+(一+零)])

		#  --- large ---
		if 首零(state.dimensionsTotal) < leafAt二:
			dimension = 二
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零])

			dimension = 四
			if bit_test(leafAt二, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension, leafAt二 - dimension, 首零(state.dimensionsTotal) + dimension + 零, 首零一二(state.dimensionsTotal)])

				if bit_test(leafAt二, dimensionIndex(三)):
					listRemoveLeaves.extend([leafAt二 - 五])

	listRemoveLeaves.extend(exclude(creasePostAt二, listIndicesCreasePostToKeep))

#========= use leafAt首Less二 to exclude a `leaf` from `pile` ===================

	dimensionHead: int = dimensionNearest首(leafAt二Ante首)
	dimensionTail: int = dimensionNearestTail(leafAt二Ante首)

	#-------- I DON'T KNOW AND I DON'T CARE WHY THIS WORKS AS LONG AS IT WORKS -------
	if (leafAt二Ante首 - 1) in getitem(dictionaryPileRanges, (neg(二)+state.首)):
		dimension = 三
		if not bit_test(leafAt二Ante首, dimensionIndex(dimension)):

			enumerateFrom1: int = zeroIndexed
			for bitToTest, leafToRemove in enumerate(tuple(getLeavesCreaseAnte(state, (leafAt二Ante首 - 1))), start=enumerateFrom1):
				if bit_test(leafAt二Ante首, bitToTest):
					listRemoveLeaves.extend([leafToRemove])

				if dimensionHead < bitToTest:
					listRemoveLeaves.extend([leafToRemove])

	theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead: int = 1
	if bit_test(leafAt二Ante首, theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead):
		creaseAnteAt二Ante首: tuple[int, ...] = tuple(getLeavesCreaseAnte(state, leafAt二Ante首))

		largestPossibleLengthOfListOfCreases: int = state.dimensionsTotal - 1
		if len(creaseAnteAt二Ante首) == largestPossibleLengthOfListOfCreases:

			voodooAddend: int = 2
			if not bit_test(leafAt二Ante首, voodooAddend + theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead):
				voodooMath: int = creaseAnteAt二Ante首[largestPossibleLengthOfListOfCreases - zeroIndexed]

				listRemoveLeaves.extend([voodooMath])
	# /voodooMath

	# --- only 17 allows 49 ---

	if leafAt二Ante首 !=(零)+首一(state.dimensionsTotal):
		listRemoveLeaves.extend([(零)+首零一(state.dimensionsTotal)])

	# --- odd and even ---

	if howManyDimensionsHaveOddParity(leafAt二Ante首) == 1:
		listRemoveLeaves.extend([leafInSubHyperplane(leafAt二Ante首)])

	dimension = 二
	if bit_test(leafAt二Ante首, dimensionIndex(dimension)):
		listRemoveLeaves.extend([leafAt二Ante首 - dimension])

		if (is_even(leafAt二Ante首)
		or (is_odd(leafAt二Ante首) and (dimensionIndex(dimension) < dimensionsConsecutiveAtTail(state, leafAt二Ante首)))):
			listRemoveLeaves.extend([dimension])

	dimension = 三
	if bit_test(leafAt二Ante首, dimensionIndex(dimension)):
		listRemoveLeaves.extend([leafAt二Ante首 - dimension])

		dimension = 四
		if is_even(leafAt二Ante首) and (not bit_test(leafAt二Ante首, dimensionIndex(dimension))):
			listRemoveLeaves.extend([leafAt二Ante首 - getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])

	if dimensionTail == 3:
		listRemoveLeaves.extend([getitem(state.sumsOfProductsOfDimensionsNearest首, dimensionTail)])

	# --- large ---

	if 首零(state.dimensionsTotal) < leafAt二Ante首:

		dimension = 一
		if bit_test(leafAt二Ante首, dimensionIndex(dimension)):
			listRemoveLeaves.extend([dimension, 首零(state.dimensionsTotal) + dimension + 零])

		if is_odd(leafAt二Ante首) and (not bit_test(leafAt二Ante首, dimensionIndex(dimension))):
			listRemoveLeaves.extend([leafAt二Ante首 - 首零(state.dimensionsTotal) - dimension])

			dimension = 二
			if bit_test(leafAt二Ante首, dimensionIndex(dimension)):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + getitem(state.sumsOfProductsOfDimensions, dimensionIndex(dimension))])

		dimension = 二
		if bit_test(leafAt二Ante首, dimensionIndex(dimension)):
			listRemoveLeaves.extend([首零(state.dimensionsTotal) + dimension + 零])

			dimension = 三
			if is_even(leafAt二Ante首) and bit_test(leafAt二Ante首, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension])

		dimension = 四
		if bit_test(leafAt二Ante首, dimensionIndex(dimension)):
			listRemoveLeaves.extend([leafAt二Ante首 - dimension])

		if not bit_test(leafAt二Ante首, dimensionIndex(dimension)):
			listRemoveLeaves.extend([leafAt二Ante首 + dimension])

	if is_odd(leafAt二Ante首):
		dimension = 零 # This is redundant but it might help expose patterns.
		if bit_test(leafAt二Ante首, dimensionIndex(dimension)):
			listRemoveLeaves.extend([一, leafAt二Ante首 - dimension, leafAt二Ante首 - getitem(state.productsOfDimensions, raiseIfNone(dimensionSecondNearest首(leafAt二Ante首)))])

	if is_even(leafAt二Ante首):
		dimension = 零 # This is redundant but it might help expose patterns.
		if not bit_test(leafAt二Ante首, dimensionIndex(dimension)):
			listRemoveLeaves.extend([leafAt二Ante首 + dimension, state.productsOfDimensions[dimensionTail], leafAt二Ante首 - state.productsOfDimensions[dimensionTail]])

		dimension = 二
		if bit_test(leafAt二Ante首, dimensionIndex(dimension)):
			listRemoveLeaves.extend([dimension])

			if 首零(state.dimensionsTotal) < leafAt二Ante首 < 首零一二(state.dimensionsTotal):
				listRemoveLeaves.extend([leafAt二Ante首 + dimensionTail])

				if dimensionTail == 2:
					addendIDC: int = (state.首 - leafAt二Ante首) // 2
					listRemoveLeaves.extend([addendIDC + leafAt二Ante首])

			if leafAt二Ante首 < 首零(state.dimensionsTotal):
				listRemoveLeaves.extend([leafAt二Ante首 + state.sumsOfProductsOfDimensions[dimensionTail], state.首 - leafAt二Ante首])

		if leafAt二Ante首 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt二Ante首 + state.productsOfDimensions[dimensionNearest首(leafAt二Ante首) + 1]])

			dimension = 三
			if not bit_test(leafAt二Ante首, dimensionIndex(dimension)):
				listRemoveLeaves.extend([dimension, leafAt二Ante首 + dimension, state.sumsOfProductsOfDimensionsNearest首[dimensionIndex(dimension)]])

		if leafAt二Ante首 !=(一)+首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])

	del dimensionHead, dimensionTail

	return sorted(set(dictionaryPileRanges[state.pile]).difference(set(listRemoveLeaves)))

