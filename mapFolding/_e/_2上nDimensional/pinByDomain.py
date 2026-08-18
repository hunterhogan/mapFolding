from __future__ import annotations

from gmpy2 import bit_flip, bit_mask, bit_test as isBit1吗, is_even as isEven吗, is_odd as isOdd吗
from hunterMakesPy import decreasing, raiseIfNone, zeroIndexed
from mapFolding._e import getIteratorOfLeaves, getMapShape首ProductsSums
from mapFolding._e._2上nDimensional import (
	dimensionIndex, getLeavesCreaseAnte, getLeavesCreasePost, leafInSubHyperplane, moreThanLeaf零吗, ptount, 一, 三, 二, 五, 四, 工dimensionTail,
	工dimension首一, 工dimension首零, 工totalDimensionsOdd, 工totalDimensionsTail, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二)
from mapFolding._e.pileOptions import getDictionaryChoicesLeaf
from more_itertools import last
from operator import getitem, neg
from typing import TYPE_CHECKING
from Z0Z_tools import exclude

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import StateElimination
	from mapFolding._e.theTypes import ChoicesLeaf, Leaf, Pile

#======== Domain-based subroutines for analyzing a specific `pile`. =======

def pinPile零Ante首零AfterDepth4(state: StateElimination) -> list[int]:
	"""All fourth-order piles must be pinned or you will get an error.

	Unless I've made a mistake:

	I have made a rule to exclude a leaf from pile 零Ante首零
		for all piles in the first four orders (pile <= 4 or pile >= 首 - 4),
			for all leaves in each pile's pile-range
				that exclude a leaf from pile 零Ante首零.

	If I were to figure out the last few cases, it would remove 23 surplus dictionaries.
		10 of 23 dictionaries
			if leafAt二 == 15:
				boxOfRemoveLeaves.extend([38])
		2 of 23 dictionaries
			if leafAt二 == 9:
				boxOfRemoveLeaves.extend([19])
		2 of 23 dictionaries
				boxOfRemoveLeaves.extend([59])
		3 of 23 dictionaries
			if leafAt二 == 23:
				boxOfRemoveLeaves.extend([50])
		4 of 23 dictionaries
			if leafAt二 == 29:
				boxOfRemoveLeaves.extend([7])
		2 of 23 dictionaries
				boxOfRemoveLeaves.extend([35])

	But I would still have 1312 surplus dictionaries.

	Therefore, if I continue to pin pile 零Ante首零, I should probably focus on different strategies.

	Returns
	-------
	boxOfRemoveLeaves : list[int]
		A list of leaves to exclude from pile 零Ante首零.
	"""
	leafAt一:			Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
	leafAt一Ante首:		Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))
	leafAt一零:			Leaf = raiseIfNone(state.permutationSpace.getLeaf(一 + 零))
	leafAt零一Ante首:	Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(零 + 一) + state.首))
	leafAt二:			Leaf = raiseIfNone(state.permutationSpace.getLeaf(二))
	leafAt二Ante首:		Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(二) + state.首))

	dictionaryChoicesLeaf: dict[Pile, ChoicesLeaf] = getDictionaryChoicesLeaf(state)
	boxOfRemoveLeaves: list[int] = []

#========= use `leafAt一` to exclude a `leaf` from `pile` ===================

	pileExcluder: Pile = 一
	for dimension, leaf in enumerate(getIteratorOfLeaves(dictionaryChoicesLeaf[pileExcluder])):
		if leaf == leafAt一:
			if dimension < state.totalDimensions - 2:
				boxOfRemoveLeaves.extend([一, 首零(state.totalDimensions) + leafAt一])
			if 0 < dimension < state.totalDimensions - 2:
				boxOfRemoveLeaves.extend([一 + leafAt一])
			if dimension == 1:
				boxOfRemoveLeaves.extend([首零(state.totalDimensions) + leafAt一 + 零])
			if dimension == state.totalDimensions - 2:
				boxOfRemoveLeaves.extend([首一(state.totalDimensions), 首一(state.totalDimensions) + leafAt一])
	del pileExcluder

#-------- Use information from other piles to select which leaves to exclude. -------
	if leafAt一 == (零) + 首零(state.totalDimensions):
		boxOfRemoveLeaves.extend([首一(state.totalDimensions), leafAt一Ante首 + 零])
	if 工dimension首零(leafAt一) < state.totalDimensions - 3:
		boxOfRemoveLeaves.extend([一, leafAt一Ante首 + 一])

#========= use `leafAt一Ante首` to exclude a `leaf` from `pile` ===================

	pileExcluder = neg(一) + state.首
	for dimension, leaf in enumerate(getIteratorOfLeaves(dictionaryChoicesLeaf[pileExcluder])):
		if leaf == leafAt一Ante首:
			if dimension == 0:
				boxOfRemoveLeaves.extend([一])
			if dimension < state.totalDimensions - 2:
				boxOfRemoveLeaves.extend([首一(state.totalDimensions) + leafAt一Ante首])
			if 0 < dimension < state.totalDimensions - 2:
				boxOfRemoveLeaves.extend([getitem(state.mapShapeProducts, dimension), 首一(state.totalDimensions) + leafAt一Ante首 - getitem(state.mapShapeProductsSums, dimension)])
			if 0 < dimension < state.totalDimensions - 3:
				boxOfRemoveLeaves.extend([零 + leafAt一Ante首])
			if 0 < dimension < state.totalDimensions - 1:
				boxOfRemoveLeaves.extend([首一(state.totalDimensions)])
	del pileExcluder

#-------- Use information from other piles to decide whether to exclude some leaves. -------
	if (leafAt一 == (零) + 首二(state.totalDimensions)) and (leafAt一Ante首 == 首零一(state.totalDimensions)):
		boxOfRemoveLeaves.extend([首二(state.totalDimensions), 首零一二(state.totalDimensions)])

#========= use `leafAt一零` to exclude a `leaf` from `pile` ===================
# DEVELOPMENT a leaf in pile一零 does not have leafCrease in the pile-range of pile零Ante首零, but `leafInSubHyperplane(leafAt一零)` does
# have leafCrease in the pile-range of pile零Ante首零. `ptount` uses leafInSubHyperplane. I wrote this code block long before I
# understood this.

# DEVELOPMENT this section relies on the exclusions in `leafAt一` and `leafAt一Ante首` to exclude some leaves.

	boxOfRemoveLeaves.extend([leafAt一零])
	if leafAt一零 == 三 + 二 + 零:
		boxOfRemoveLeaves.extend([二 + 一 + 零, (零 + 二) + 首零(state.totalDimensions)])
	if leafAt一零 == (零 + 二) + 首一(state.totalDimensions):
		boxOfRemoveLeaves.extend([首二(state.totalDimensions), leafAt一零 + getitem(state.mapShapeProducts, raiseIfNone(工dimension首一(leafAt一零))), leafAt一零 + getitem(state.mapShapeProductsSums, raiseIfNone(工dimension首一(leafAt一零)) + 1), 首零一二(state.totalDimensions)])
	if leafAt一零 == (零) + 首一二(state.totalDimensions):
		boxOfRemoveLeaves.extend([首一(state.totalDimensions) + (一 + 零), last(getLeavesCreaseAnte(state, leafInSubHyperplane(leafAt一零)))])
	if leafAt一零 == (零) + 首零一(state.totalDimensions):
		boxOfRemoveLeaves.extend([首零一二(state.totalDimensions)])
	if isOdd吗(leafAt一零):
		dimensionHeadSecond: int = raiseIfNone(工dimension首一(leafAt一零))
		次By首Second: int = dimensionHeadSecond * decreasing + decreasing  # Are you confused and/or annoyed by this? Blame Python. (Or figure out a better formula.)
		boxOfRemoveLeaves.extend([getitem(state.mapShapeProducts, dimensionHeadSecond)])
		if leafAt一零 < 首零(state.totalDimensions):
			mapShape首ProductsSumsInSubHyperplane: tuple[int, ...] = getMapShape首ProductsSums(state.mapShapeProducts, state.totalDimensions, state.totalDimensions - 1)
			boxOfRemoveLeaves.extend([一, leafAt一零 + getitem(state.mapShapeProductsSums, (state.totalDimensions - 1)), leafAt一零 + getitem(mapShape首ProductsSumsInSubHyperplane, 次By首Second)])
			if dimensionHeadSecond == 2:
				boxOfRemoveLeaves.extend([getitem(state.mapShapeProductsSums, dimensionHeadSecond) + getitem(state.mapShapeProducts, 工dimension首零(leafAt一零)), getitem(state.mapShapeProductsSums, dimensionHeadSecond) + 首零(state.totalDimensions)])
			if dimensionHeadSecond == 3:
				boxOfRemoveLeaves.extend([一 + leafAt一零 + getitem(state.mapShapeProducts, (state.totalDimensions - 1))])
		if 首零(state.totalDimensions) < leafAt一零:
			boxOfRemoveLeaves.extend([(零) + 首零一(state.totalDimensions), getitem(state.mapShapeProducts, (工dimension首零(leafAt一零) - 1))])

#========= use `leafAt零一Ante首` to exclude a `leaf` from `pile` ===================
# DEVELOPMENT a leaf in pile首Less一零 does not have leafCrease in the pile-range of pile零Ante首零, but `leafInSubHyperplane(leafAt首
# Less一零)` does have leafCrease in the pile-range of pile零Ante首零. `ptount` uses leafInSubHyperplane. I wrote this code block
# long before I understood this.

# DEVELOPMENT This section could be "modernized" to be more similar to `leafAt一零`, which used to have `comebackOffset`, too.

	boxOfRemoveLeaves.extend([leafAt零一Ante首])

	if 首零(state.totalDimensions) < leafAt零一Ante首:
		boxOfRemoveLeaves.extend([(零) + 首零一(state.totalDimensions)])
		if isEven吗(leafAt零一Ante首):
			boxOfRemoveLeaves.extend([首一(state.totalDimensions)])
			dimension: int = 一
			if isBit1吗(leafAt零一Ante首, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([dimension, 首零(state.totalDimensions) + dimension + 零, state.首 - sum(state.mapShapeProducts[dimensionIndex(dimension): state.totalDimensions - 2]), leafAt零一Ante首 - dimension - getitem(state.mapShapeProductsSums, (dimensionIndex(dimension) + 1))])
			dimension = 二
			if isBit1吗(leafAt零一Ante首, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([dimension, 首零(state.totalDimensions) + dimension + 零])
				if 1 < 工dimensionTail(leafAt零一Ante首):
					boxOfRemoveLeaves.extend([state.首 - sum(state.mapShapeProducts[dimensionIndex(dimension): state.totalDimensions - 2])])
				else:  # DEVELOPMENT IDK and IDC why this works, but it does.
					boxOfRemoveLeaves.extend([getitem(tuple(getLeavesCreaseAnte(state, leafInSubHyperplane(leafAt零一Ante首))), dimensionIndex(dimension)) - 零])
			dimension = 三
			if isBit1吗(leafAt零一Ante首, dimensionIndex(dimension)):
				if 1 < 工dimensionTail(leafAt零一Ante首):
					boxOfRemoveLeaves.extend([dimension])
					boxOfRemoveLeaves.extend([state.首 - sum(state.mapShapeProducts[dimensionIndex(dimension): state.totalDimensions - 2])])
				if 工dimensionTail(leafAt零一Ante首) < dimensionIndex(dimension):
					boxOfRemoveLeaves.extend([首零(state.totalDimensions) + dimension + 零])
			sheepOrGoat = 0
			shepherdOfDimensions: int = int(bit_flip(0, state.totalDimensions - 5))
			if (leafAt零一Ante首 // shepherdOfDimensions) & bit_mask(5) == 0b10101:
				boxOfRemoveLeaves.extend([二])
				sheepOrGoat: int = ptount(leafAt零一Ante首 // shepherdOfDimensions)
				if 0 < sheepOrGoat < state.totalDimensions - 3:
					comebackOffset: int = state.mapShapeProducts[工dimension首零(leafAt零一Ante首)] - 二
					boxOfRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
				if 0 < sheepOrGoat < state.totalDimensions - 4:
					comebackOffset = state.mapShapeProducts[raiseIfNone(工dimension首一(leafAt零一Ante首))] - 二
					boxOfRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
		if isOdd吗(leafAt零一Ante首):
			boxOfRemoveLeaves.extend([一])
			if leafAt零一Ante首 & bit_mask(4) == 0b001001:
				boxOfRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAt零一Ante首)
			if 0 < sheepOrGoat < state.totalDimensions - 3:
				comebackOffset = state.mapShapeProducts[工dimension首零(leafAt零一Ante首)] - 一
				boxOfRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])
			if 0 < sheepOrGoat < state.totalDimensions - 4:
				comebackOffset = state.mapShapeProducts[raiseIfNone(工dimension首一(leafAt零一Ante首))] - 一
				boxOfRemoveLeaves.extend([leafAt零一Ante首 - comebackOffset])

#-------- Use information from other piles to decide whether to exclude some leaves. -------
	if (leafAt一 == 一 + 零) and (leafAt零一Ante首 != next(getLeavesCreaseAnte(state, (零) + 首零(state.totalDimensions)))):
		boxOfRemoveLeaves.append(首一(state.totalDimensions))

# DEVELOPMENT Above this line, all exclusions based on only one leaf in a pile are covered. 😊
#========= use leafAt二 to exclude a `leaf` from `pile` ===================
# DEVELOPMENT Below this line, abandon all hope, the who code here. 😈

	dimensionHead: int = 工dimension首零(leafAt二)
	creasePostAt二: tuple[int, ...] = tuple(getLeavesCreasePost(state, leafAt二))
	boxOfIndicesCreasePostToKeep: list[int] = []

	if (二 < leafAt二 < neg(零) + 首一(state.totalDimensions)):
		boxOfRemoveLeaves.extend([leafAt二 + 首零(state.totalDimensions)])

		dimension = 一
		if isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([leafAt二 + 首零(state.totalDimensions) + dimension])

		if not isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([leafAt二 + 首零(state.totalDimensions) - dimension])

		if isOdd吗(leafAt二):
			dimension = 三
			if isBit1吗(leafAt二, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([leafAt二 + 首零(state.totalDimensions) + dimension])

				dimension = 四
				if not isBit1吗(leafAt二, dimensionIndex(dimension)):
					boxOfRemoveLeaves.extend([leafAt二 + 首零(state.totalDimensions) - dimension])

	if ((首一(state.totalDimensions) < leafAt二 < 首零(state.totalDimensions)) and raiseIfNone(工dimension首一(leafAt二)) != 2):
		boxOfRemoveLeaves.extend([leafAt二 + 首零(state.totalDimensions)])

		if isOdd吗(leafAt二):
			dimension = 二
			if not isBit1吗(leafAt二, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([leafAt二 + 首零(state.totalDimensions) - getitem(state.mapShapeProductsSums, dimensionIndex(dimension))])

			dimension = 三
			if not isBit1吗(leafAt二, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([leafAt二 + 首零(state.totalDimensions) - dimension, leafAt二 + 首零(state.totalDimensions) + getitem(state.mapShapeProductsSums, dimensionIndex(dimension))])

			dimension = 四
			if isBit1吗(leafAt二, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([leafAt二 - dimension])

	if isEven吗(leafAt二):
		boxOfIndicesCreasePostToKeep.extend(range(state.totalDimensions - dimensionHead + 1, (state.totalDimensions - zeroIndexed)))

		boxOfRemoveLeaves.extend([
				leafAt二 + 零, leafAt二 + 首零(state.totalDimensions), leafAt二 + getitem(state.mapShapeProductsSums, (state.totalDimensions - 1)), getitem(state.mapShapeProducts, dimensionHead) + (一 + 零)])

		dimension = 一
		if isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([dimension, 首零(state.totalDimensions) + dimension + 零])

		dimension = 二
		if not isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfIndicesCreasePostToKeep.append(creasePostAt二.index(state.mapShapeProducts[dimensionHead]))

		if leafAt二 < 首零(state.totalDimensions):
			boxOfRemoveLeaves.extend([getitem(state.mapShapeProducts, dimensionIndex(二)), getitem(state.mapShapeProductsSums, (dimensionIndex(二) + 1))])

		dimension = 四
		if (not isBit1吗(leafAt二, dimensionIndex(dimension))) and (首零(state.totalDimensions) < leafAt二):
			boxOfRemoveLeaves.extend([getitem(state.mapShapeProducts, dimensionIndex(dimension))])

		# DEVELOPMENT 1) I am sure this concept has validity. 2) I am sure there is a more accurate computation for it.
		zerosAtThe首 = 2
		if state.totalDimensions - zeroIndexed - dimensionHead == zerosAtThe首:
			mapShape首ProductsSumsInSubSubHyperplane: tuple[int, ...] = getMapShape首ProductsSums(state.mapShapeProducts, state.totalDimensions, state.totalDimensions - zerosAtThe首)
			addendForUnknownReasons: int = -1
			leavesWeDontWant: list[int] = [aLeaf + addendForUnknownReasons for aLeaf in filter(moreThanLeaf零吗, mapShape首ProductsSumsInSubSubHyperplane)]
			boxOfRemoveLeaves.extend(leavesWeDontWant)

	if isOdd吗(leafAt二):

		if 工dimensionTail(leafAt二 - 1) == 1:
			boxOfRemoveLeaves.extend([一])

		if leafInSubHyperplane(leafAt二) == state.mapShapeProductsSums[3]:
			boxOfRemoveLeaves.extend([二])

		dimension = 零
		if isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([dimension, leafAt二 - dimension, 首零(state.totalDimensions) + dimension + 零])

		dimension = 二
		if not isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfIndicesCreasePostToKeep.append(dimensionIndex(dimension))

		if isBit1吗(leafAt二, dimensionIndex(dimension)) and isBit1吗(leafAt二, dimensionIndex(一)):
			boxOfRemoveLeaves.extend([leafAt二 - dimension, 首零(state.totalDimensions) + dimension + 零])

		dimension = 三
		if isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([leafAt二 - dimension, 首零(state.totalDimensions) + dimension + 零])

		if not isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfIndicesCreasePostToKeep.append(dimensionIndex(dimension))

			dimension = 四
			if not isBit1吗(leafAt二, dimensionIndex(dimension)):
				boxOfIndicesCreasePostToKeep.append(dimensionIndex(dimension))

		dimension = 四
		if isBit1吗(leafAt二, dimensionIndex(dimension)):

			dimensionBonus: int = 零
			if isBit1吗(leafAt二, dimensionIndex(dimensionBonus)):
				boxOfRemoveLeaves.extend([首零(state.totalDimensions) + dimension + dimensionBonus])

			dimensionBonus = 二
			if isBit1吗(leafAt二, dimensionIndex(dimensionBonus)):
				boxOfRemoveLeaves.extend([首零(state.totalDimensions) + dimension + dimensionBonus])

			dimensionBonus = 三
			if isBit1吗(leafAt二, dimensionIndex(dimensionBonus)):
				boxOfRemoveLeaves.extend([首零(state.totalDimensions) + dimension + dimensionBonus])

		dimension = 五
		if isBit1吗(leafAt二, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([首一(state.totalDimensions), (零) + 首零一(state.totalDimensions)])

		#--- small ---
		if leafAt二 < 首一(state.totalDimensions):
			boxOfRemoveLeaves.extend([一])

		#--- medium ---
		if 首一(state.totalDimensions) < leafAt二 < 首零(state.totalDimensions):
			boxOfRemoveLeaves.extend([leafAt二 + getitem(state.mapShapeProductsSums, (state.totalDimensions - 2)), 首一(state.totalDimensions) + (一 + 零)])

		#--- large ---
		if 首零(state.totalDimensions) < leafAt二:
			dimension = 二
			if isBit1吗(leafAt二, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([leafAt二 - dimension, 首零(state.totalDimensions) + dimension + 零])

			dimension = 四
			if isBit1吗(leafAt二, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([dimension, leafAt二 - dimension, 首零(state.totalDimensions) + dimension + 零, 首零一二(state.totalDimensions)])

				if isBit1吗(leafAt二, dimensionIndex(三)):
					boxOfRemoveLeaves.extend([leafAt二 - 五])

	boxOfRemoveLeaves.extend(exclude(creasePostAt二, boxOfIndicesCreasePostToKeep))

#========= use leafAt首Less二 to exclude a `leaf` from `pile` ===================

	dimensionHead: int = 工dimension首零(leafAt二Ante首)
	dimensionTail: int = 工dimensionTail(leafAt二Ante首)

	#-------- I DON'T KNOW AND I DON'T CARE WHY THIS WORKS AS LONG AS IT WORKS -------
	if isBit1吗(getitem(dictionaryChoicesLeaf, (neg(二) + state.首)), leafAt二Ante首 - 1):
		dimension = 三
		if not isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):

			enumerateFrom1: int = zeroIndexed
			for bitToTest, leafToRemove in enumerate(tuple(getLeavesCreaseAnte(state, (leafAt二Ante首 - 1))), start=enumerateFrom1):
				if isBit1吗(leafAt二Ante首, bitToTest):
					boxOfRemoveLeaves.extend([leafToRemove])

				if dimensionHead < bitToTest:
					boxOfRemoveLeaves.extend([leafToRemove])

	theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead: int = 1
	if isBit1吗(leafAt二Ante首, theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead):
		creaseAnteAt二Ante首: tuple[int, ...] = tuple(getLeavesCreaseAnte(state, leafAt二Ante首))

		largestPossibleLengthOfListOfCreases: int = state.totalDimensions - 1
		if len(creaseAnteAt二Ante首) == largestPossibleLengthOfListOfCreases:

			voodooAddend: int = 2
			if not isBit1吗(leafAt二Ante首, voodooAddend + theLastPossibleIndexOfCreaseAnteIfCountingFromTheHead):
				voodooMath: int = creaseAnteAt二Ante首[largestPossibleLengthOfListOfCreases - zeroIndexed]

				boxOfRemoveLeaves.extend([voodooMath])
	# /voodooMath

	#--- only 17 allows 49 ---

	if leafAt二Ante首 != (零) + 首一(state.totalDimensions):
		boxOfRemoveLeaves.extend([(零) + 首零一(state.totalDimensions)])

	#--- odd and even ---

	if 工totalDimensionsOdd(leafAt二Ante首) == 1:
		boxOfRemoveLeaves.extend([leafInSubHyperplane(leafAt二Ante首)])

	dimension = 二
	if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
		boxOfRemoveLeaves.extend([leafAt二Ante首 - dimension])

		if (isEven吗(leafAt二Ante首)
		or (isOdd吗(leafAt二Ante首) and (dimensionIndex(dimension) < 工totalDimensionsTail(state, leafAt二Ante首)))):
			boxOfRemoveLeaves.extend([dimension])

	dimension = 三
	if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
		boxOfRemoveLeaves.extend([leafAt二Ante首 - dimension])

		dimension = 四
		if isEven吗(leafAt二Ante首) and (not isBit1吗(leafAt二Ante首, dimensionIndex(dimension))):
			boxOfRemoveLeaves.extend([leafAt二Ante首 - getitem(state.mapShapeProductsSums, dimensionIndex(dimension))])

	if dimensionTail == 3:
		boxOfRemoveLeaves.extend([getitem(state.mapShape首ProductsSums, dimensionTail)])

	#--- large ---

	if 首零(state.totalDimensions) < leafAt二Ante首:

		dimension = 一
		if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([dimension, 首零(state.totalDimensions) + dimension + 零])

		if isOdd吗(leafAt二Ante首) and (not isBit1吗(leafAt二Ante首, dimensionIndex(dimension))):
			boxOfRemoveLeaves.extend([leafAt二Ante首 - 首零(state.totalDimensions) - dimension])

			dimension = 二
			if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([首零(state.totalDimensions) + getitem(state.mapShapeProductsSums, dimensionIndex(dimension))])

		dimension = 二
		if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([首零(state.totalDimensions) + dimension + 零])

			dimension = 三
			if isEven吗(leafAt二Ante首) and isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([dimension])

		dimension = 四
		if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([leafAt二Ante首 - dimension])

		if not isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([leafAt二Ante首 + dimension])

	if isOdd吗(leafAt二Ante首):
		dimension = 零  # This is redundant but it might help expose patterns.
		if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([一, leafAt二Ante首 - dimension, leafAt二Ante首 - getitem(state.mapShapeProducts, raiseIfNone(工dimension首一(leafAt二Ante首)))])

	if isEven吗(leafAt二Ante首):
		dimension = 零  # This is redundant but it might help expose patterns.
		if not isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([leafAt二Ante首 + dimension, state.mapShapeProducts[dimensionTail], leafAt二Ante首 - state.mapShapeProducts[dimensionTail]])

		dimension = 二
		if isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
			boxOfRemoveLeaves.extend([dimension])

			if 首零(state.totalDimensions) < leafAt二Ante首 < 首零一二(state.totalDimensions):
				boxOfRemoveLeaves.extend([leafAt二Ante首 + dimensionTail])

				if dimensionTail == 2:
					addendIDC: int = (state.首 - leafAt二Ante首) // 2
					boxOfRemoveLeaves.extend([addendIDC + leafAt二Ante首])

			if leafAt二Ante首 < 首零(state.totalDimensions):
				boxOfRemoveLeaves.extend([leafAt二Ante首 + state.mapShapeProductsSums[dimensionTail], state.首 - leafAt二Ante首])

		if leafAt二Ante首 < 首零(state.totalDimensions):
			boxOfRemoveLeaves.extend([首一(state.totalDimensions), leafAt二Ante首 + state.mapShapeProducts[工dimension首零(leafAt二Ante首) + 1]])

			dimension = 三
			if not isBit1吗(leafAt二Ante首, dimensionIndex(dimension)):
				boxOfRemoveLeaves.extend([dimension, leafAt二Ante首 + dimension, state.mapShape首ProductsSums[dimensionIndex(dimension)]])

		if leafAt二Ante首 != (一) + 首零(state.totalDimensions):
			boxOfRemoveLeaves.extend([首一(state.totalDimensions)])

	del dimensionHead, dimensionTail

	return sorted(set(getIteratorOfLeaves(dictionaryChoicesLeaf[state.pile])).difference(set(boxOfRemoveLeaves)))
