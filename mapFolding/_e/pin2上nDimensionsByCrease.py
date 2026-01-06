from gmpy2 import bit_flip, is_even, is_odd
from hunterMakesPy import raiseIfNone
from mapFolding import inclusive
from mapFolding._e import (
	dimensionIndex, dimensionNearestTail, dimensionNearest首, DOTgetPileIfLeaf, exclude, getLeavesCreaseBack,
	getLeavesCreaseNext, leafInSubHyperplane, ptount, 一, 三, 二, 五, 四, 零, 首一, 首零, 首零一)
from mapFolding._e.dataBaskets import EliminationState
from operator import add, neg, sub
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

# ======= crease-based subroutines for analyzing a specific `pile`. =======
def _getLeavesCrease(state: EliminationState, leaf: int) -> tuple[int, ...]:
	if 0 < leaf:
		return tuple(getLeavesCreaseBack(state, abs(leaf)))
	return tuple(getLeavesCreaseNext(state, abs(leaf)))

# Second order
def pinPile一Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt首Less一: int | None = DOTgetPileIfLeaf(state.leavesPinned, state.leavesTotal - 一)

	if leafAt首Less一 and (0 < dimensionNearestTail(leafAt首Less一)):
		listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAt首Less一) - 零, state.dimensionsTotal - 一)])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int | None = DOTgetPileIfLeaf(state.leavesPinned, 一)

	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listCreaseIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

# Third order
def pinPile一零Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, state.leavesTotal - 一))

	if 1 < len(tupleLeavesCrease):
		listCreaseIndicesExcluded.append(0)
	if is_even(leafAt首Less一) and (leafAt一 == 首零(state.dimensionsTotal)+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAt首Less一) + 零, state.dimensionsTotal)])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一零Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, state.leavesTotal - 一))

	if leafAt首Less一 < 首零一(state.dimensionsTotal):
		listCreaseIndicesExcluded.append(-1)
	if (leafAt首Less一 == 首零(state.dimensionsTotal)+零) and (leafAt一 != 一+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 零)])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

# Fourth order
def pinPile二Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, state.leavesTotal - 一))
	leafAt一零: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, 一+零))
	leafAt首Less一零: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, state.leavesTotal - (一+零)))

	if is_odd(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一零), 5), ptount(leafAt一零)])
		listCreaseIndicesExcluded.append((dimensionIndex(leafInSubHyperplane(leafAt首Less一)) + 4) % 5)
	if is_even(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafInSubHyperplane(leafAt首Less一零 - (leafAt首Less一零.bit_count() - is_even(leafAt首Less一零))).bit_count()) % (state.dimensionsTotal - 2) - is_even(leafAt首Less一零): None])
		if is_even(leafAt首Less一):
			listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafInSubHyperplane(leafAt首Less一)) - 一, (state.dimensionsTotal - 3))])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listCreaseIndicesExcluded.extend([(dimensionIndex(leafInSubHyperplane(leafAt首Less一)) + 4) % 5, dimensionNearestTail(leafAt首Less一零) - 1])
		if 首零(state.dimensionsTotal)+零 < leafAt首Less一零:
			listCreaseIndicesExcluded.extend([*range(int(leafAt首Less一零 - int(bit_flip(0, dimensionNearest首(leafAt首Less一零)))).bit_length() - 1, state.dimensionsTotal - 2)])
		if ((0 < leafAt一零 - leafAt一 <= bit_flip(0, state.dimensionsTotal - 4)) and (0 < (leafAt首Less一 - leafAt一零) <= bit_flip(0, state.dimensionsTotal - 3))):
			listCreaseIndicesExcluded.extend([ptount(leafAt一零), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less二Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, state.leavesTotal - 一))
	leafAt一零: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, 一+零))
	leafAt首Less一零: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, state.leavesTotal - (一+零)))
	leafAt二: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, 二))

	addendDimension首零: int = leafAt首Less一零 - leafAt首Less一
	addendDimension一零: int = leafAt二 - leafAt一零
	addendDimension一: int = 			 leafAt一零 - leafAt一
	addendDimension零: int =						 leafAt一 - 零

# ruff: noqa: SIM102

	if ((addendDimension一零 in [一, 二, 三, 四])
		or ((addendDimension一零 == 五) and (addendDimension首零 != 一))
		or (addendDimension一 in [二, 三])
		or ((addendDimension一 == 一) and not (addendDimension零 == addendDimension首零 and addendDimension一零 < 0))
	):
		if leafAt首Less一零 == 首一(state.dimensionsTotal):
			if addendDimension零 == 三:
				listCreaseIndicesExcluded.append(dimensionIndex(二))
			if addendDimension零 == 五:
				if addendDimension一 == 二:
					listCreaseIndicesExcluded.append(dimensionIndex(二))
				if addendDimension一 == 三:
					listCreaseIndicesExcluded.append(dimensionIndex(三))
			if addendDimension一零 == 三:
				listCreaseIndicesExcluded.append(dimensionIndex(二))

		if 0 < (dimensionTail := dimensionNearestTail(leafAt首Less一零)) < 5:
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

	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

