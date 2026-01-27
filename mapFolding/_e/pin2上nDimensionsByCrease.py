from collections.abc import Iterator
from gmpy2 import bit_flip, is_even as isEven吗, is_odd as isOdd吗
from hunterMakesPy import raiseIfNone
from mapFolding import inclusive
from mapFolding._e import (
	dimensionIndex, dimensionNearestTail, dimensionNearest首, DOTgetPileIfLeaf, getLeavesCreaseAnte, getLeavesCreasePost,
	Leaf, leafInSubHyperplane, ptount, 一, 三, 二, 五, 四, 零, 首一, 首零, 首零一)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import exclude
from operator import add, neg, sub
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

#======== crease-based subroutines for analyzing a specific `pile`. =======
def _getLeavesCrease(state: EliminationState, leaf: Leaf) -> tuple[Leaf, ...]:
	if 0 < leaf:
		return tuple(getLeavesCreaseAnte(state, abs(leaf)))
	return tuple(getLeavesCreasePost(state, abs(leaf)))

#-------- Depth 2 ------------------------------------
def pinPile一ByCrease(state: EliminationState) -> Iterator[Leaf]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一Ante首: Leaf | None = DOTgetPileIfLeaf(state.permutationSpace, neg(一)+state.首)

	if leafAt一Ante首 and (0 < dimensionNearestTail(leafAt一Ante首)):
		listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAt一Ante首) - 零, state.dimensionsTotal - 一)])
	return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile一Ante首ByCrease(state: EliminationState) -> Iterator[Leaf]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: Leaf | None = DOTgetPileIfLeaf(state.permutationSpace, 一)

	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listCreaseIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

#-------- Depth 3 ------------------------------------
def pinPile一零ByCrease(state: EliminationState) -> Iterator[Leaf]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, 一))
	leafAt一Ante首: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(一)+state.首))

	if 1 < len(tupleLeavesCrease):
		listCreaseIndicesExcluded.append(0)
	if isEven吗(leafAt一Ante首) and (leafAt一 == (零)+首零(state.dimensionsTotal)):
		listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAt一Ante首) + 零, state.dimensionsTotal)])
	return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile零一Ante首ByCrease(state: EliminationState) -> Iterator[Leaf]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, 一))
	leafAt一Ante首: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(一)+state.首))

	if leafAt一Ante首 < 首零一(state.dimensionsTotal):
		listCreaseIndicesExcluded.append(-1)
	if (leafAt一Ante首 == (零)+首零(state.dimensionsTotal)) and (leafAt一 != 一+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 零)])
	return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

#-------- Depth 4 ------------------------------------
def pinPile二ByCrease(state: EliminationState) -> Iterator[Leaf]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, 一))
	leafAt一Ante首: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(一)+state.首))
	leafAt一零: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, 一+零))
	leafAt零一Ante首: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(零+一)+state.首))

	if isOdd吗(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一零), 5), ptount(leafAt一零)])
		listCreaseIndicesExcluded.append((dimensionIndex(leafInSubHyperplane(leafAt一Ante首)) + 4) % 5)
	if isEven吗(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafInSubHyperplane(leafAt零一Ante首 - (leafAt零一Ante首.bit_count() - isEven吗(leafAt零一Ante首))).bit_count()) % (state.dimensionsTotal - 2) - isEven吗(leafAt零一Ante首): None])
		if isEven吗(leafAt一Ante首):
			listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafInSubHyperplane(leafAt一Ante首)) - 一, (state.dimensionsTotal - 3))])
	if leafAt一 == (零)+首零(state.dimensionsTotal):
		listCreaseIndicesExcluded.extend([(dimensionIndex(leafInSubHyperplane(leafAt一Ante首)) + 4) % 5, dimensionNearestTail(leafAt零一Ante首) - 1])
		if (零)+首零(state.dimensionsTotal) < leafAt零一Ante首:
			listCreaseIndicesExcluded.extend([*range(int(leafAt零一Ante首 - int(bit_flip(0, dimensionNearest首(leafAt零一Ante首)))).bit_length() - 1, state.dimensionsTotal - 2)])
		if ((0 < leafAt一零 - leafAt一 <= bit_flip(0, state.dimensionsTotal - 4)) and (0 < (leafAt一Ante首 - leafAt一零) <= bit_flip(0, state.dimensionsTotal - 3))):
			listCreaseIndicesExcluded.extend([ptount(leafAt一零), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

def pinPile二Ante首ByCrease(state: EliminationState) -> Iterator[Leaf]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, 一))
	leafAt一Ante首: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(一)+state.首))
	leafAt一零: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, 一+零))
	leafAt零一Ante首: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, neg(零+一)+state.首))
	leafAt二: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, 二))

	addendDimension首零: int = leafAt零一Ante首 - leafAt一Ante首
	addendDimension一零: int = leafAt二 - leafAt一零
	addendDimension一: int = 			 leafAt一零 - leafAt一
	addendDimension零: int =						 leafAt一 - 零

# ruff: noqa: SIM102

	if ((addendDimension一零 in [一, 二, 三, 四])
		or ((addendDimension一零 == 五) and (addendDimension首零 != 一))
		or (addendDimension一 in [二, 三])
		or ((addendDimension一 == 一) and not (addendDimension零 == addendDimension首零 and addendDimension一零 < 0))
	):
		if leafAt零一Ante首 == 首一(state.dimensionsTotal):
			if addendDimension零 == 三:
				listCreaseIndicesExcluded.append(dimensionIndex(二))
			if addendDimension零 == 五:
				if addendDimension一 == 二:
					listCreaseIndicesExcluded.append(dimensionIndex(二))
				if addendDimension一 == 三:
					listCreaseIndicesExcluded.append(dimensionIndex(三))
			if addendDimension一零 == 三:
				listCreaseIndicesExcluded.append(dimensionIndex(二))

		if 0 < (dimensionTail := dimensionNearestTail(leafAt零一Ante首)) < 5:
			listCreaseIndicesExcluded.extend(list(range(dimensionTail % 4)) or [dimensionIndex(一)])

		if addendDimension首零 == neg(五):
			listCreaseIndicesExcluded.append(dimensionIndex(一))
		if addendDimension首零 == 一:
			listCreaseIndicesExcluded.append(dimensionIndex(二))
		if addendDimension首零 == 四:
			if addendDimension零 == 三:
				listCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
			if addendDimension一 == 一:
				if addendDimension一零 == 三:
					listCreaseIndicesExcluded.append(dimensionIndex(二))

		if addendDimension零 == 一:
			listCreaseIndicesExcluded.append(dimensionIndex(一))
			if addendDimension一零 == 三:
				listCreaseIndicesExcluded.extend([*range(dimensionIndex(二), dimensionIndex(三) + inclusive)])
			if addendDimension一零 == 四:
				listCreaseIndicesExcluded.extend([*range(dimensionIndex(三), dimensionIndex(四) + inclusive)])
		if addendDimension零 == 二:
			listCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
		if addendDimension零 == 三:
			listCreaseIndicesExcluded.append(dimensionIndex(三))

		if addendDimension一 == 二:
			listCreaseIndicesExcluded.append(dimensionIndex(一))
		if addendDimension一 == 三:
			listCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
		if addendDimension一 == 四:
			listCreaseIndicesExcluded.append(dimensionIndex(一))
			if addendDimension一零 == 三:
				listCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(三) + inclusive)])

		if addendDimension一零 == 一:
			listCreaseIndicesExcluded.append(dimensionIndex(一))
		if addendDimension一零 == 二:
			listCreaseIndicesExcluded.append(dimensionIndex(二))
		if addendDimension一零 == 三:
			listCreaseIndicesExcluded.append(dimensionIndex(三))
		if addendDimension一零 == 五:
			listCreaseIndicesExcluded.append(dimensionIndex(一))

	return exclude(tupleLeavesCrease, listCreaseIndicesExcluded)

