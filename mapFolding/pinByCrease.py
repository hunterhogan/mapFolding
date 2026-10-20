"""Select leaves for end piles using crease restrictions.

(AI generated docstring)

You can retrieve possible leaves for a target pile after pinning the required neighboring piles. You
inspect the workbench permutation space and receive leaf choices without assigning a leaf to a pile.

Contents
--------
Functions
	Depth 2
		pinPile一Ante首ByCrease
			Iterate over crease neighbors for pile two positions before the end.
		pinPile一ByCrease
			Iterate over crease neighbors for pile two.
	Depth 3
		pinPile一零ByCrease
			Iterate over crease neighbors for pile three.
		pinPile零一Ante首ByCrease
			Iterate over crease neighbors for pile three positions before the end.
	Depth 4
		pinPile二Ante首ByCrease
			Iterate over crease neighbors for pile four positions before the end.
		pinPile二ByCrease
			Iterate over crease neighbors for pile four.
"""
# ruff: file-ignore[collapsible-if, too-many-boolean-expressions]
from __future__ import annotations

from gmpy2 import bit_flip, is_even as isEven吗, is_odd as isOdd吗
from hunterMakesPy import inclusive, raiseIfNone
from mapFolding.semiotics import dimensionIndex, 一, 三, 二, 五, 四, 零, 首一, 首零, 首零一
from mapFolding.creases import getLeavesCreaseAnte, getLeavesCreasePost
from mapFolding.measure import leafInSubHyperplane, ptount, 工dimensionTail, 工dimension首零
from operator import add, neg, sub
from typing import TYPE_CHECKING
from Z0Z_tools import exclude

if TYPE_CHECKING:
	from collections.abc import Iterator
	from hunterMakesPy import CallableFunction
	from mapFolding.dataBaskets import StateElimination
	from mapFolding.theTypes import Leaf

#======== crease-based subroutines for analyzing a specific `pile`. =======
def _getLeavesCrease(state: StateElimination, leaf: Leaf) -> tuple[Leaf, ...]:
	if 0 < leaf:
		return tuple(getLeavesCreaseAnte(state, abs(leaf)))
	return tuple(getLeavesCreasePost(state, abs(leaf)))

#-------- Depth 2 ------------------------------------
def pinPile一ByCrease(state: StateElimination) -> Iterator[Leaf]:
	"""Iterate over crease neighbors for pile two.

	(AI generated docstring)

	You can obtain possible leaves for the target pile on the workbench by excluding crease neighbors
	using leaves already pinned near the ends of the pile order. You receive leaf choices without
	modifying `state`.

	Parameters
	----------
	state : StateElimination
		Workbench state for a map whose axis lengths are all `2`. Set `state.pile` to
		the target pile and provide the pinned leaves in `state.permutationSpace`.

	Returns
	-------
	leavesToPin : Iterator[Leaf]
		Iterator over leaf indices after exclusions, in the original
		crease order.

	Pinned Piles
	------------
	Pile `state.pile - 1` must contain a pinned leaf.
	No additional pinned pile is required; pile `state.首 - 2` supplies an optional restriction.

	A required pile without a pinned leaf causes `ValueError`.

	You select leaves with `mapFolding.creases.getLeavesCreasePost` [1]. Follow the
	pinning order used by `mapFolding.pinIt.pinPilesAtEnds` [2].

	References
	----------
	[1] `mapFolding.creases.getLeavesCreasePost`

	[2] `mapFolding.pinIt.pinPilesAtEnds`
	"""
	direction: CallableFunction[[int, int], int] = sub

	boxOfCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f"I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一Ante首: Leaf | None = state.permutationSpace.getLeaf(neg(一) + state.首)

	if leafAt一Ante首 and (0 < 工dimensionTail(leafAt一Ante首)):
		boxOfCreaseIndicesExcluded.extend([*range(工dimensionTail(leafAt一Ante首) - 零, state.totalDimensions - 一)])
	return exclude(tupleLeavesCrease, boxOfCreaseIndicesExcluded)

def pinPile一Ante首ByCrease(state: StateElimination) -> Iterator[Leaf]:
	"""Iterate over crease neighbors for pile two positions before the end.

	(AI generated docstring)

	You can obtain possible leaves for the target pile on the workbench by excluding crease neighbors
	using leaves already pinned near the ends of the pile order. You receive leaf choices without
	modifying `state`.

	Parameters
	----------
	state : StateElimination
		Workbench state for a map whose axis lengths are all `2`. Set `state.pile` to
		the target pile and provide the pinned leaves in `state.permutationSpace`.

	Returns
	-------
	leavesToPin : Iterator[Leaf]
		Iterator over leaf indices after exclusions, in the original
		crease order.

	Pinned Piles
	------------
	Pile `state.pile + 1` must contain a pinned leaf.
	No additional pinned pile is required; pile `2` supplies an optional restriction.

	A required pile without a pinned leaf causes `ValueError`.

	You select leaves with `mapFolding.creases.getLeavesCreaseAnte` [1]. Follow the
	pinning order used by `mapFolding.pinIt.pinPilesAtEnds` [2].

	References
	----------
	[1] `mapFolding.creases.getLeavesCreaseAnte`

	[2] `mapFolding.pinIt.pinPilesAtEnds`
	"""
	direction: CallableFunction[[int, int], int] = add

	boxOfCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f"I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: Leaf | None = state.permutationSpace.getLeaf(一)

	if leafAt一 and (leafAt一.bit_length() < state.totalDimensions):
		boxOfCreaseIndicesExcluded.extend([*range(零, 工dimension首零(leafAt一) + inclusive)])
	return exclude(tupleLeavesCrease, boxOfCreaseIndicesExcluded)

#-------- Depth 3 ------------------------------------
def pinPile一零ByCrease(state: StateElimination) -> Iterator[Leaf]:
	"""Iterate over crease neighbors for pile three.

	(AI generated docstring)

	You can obtain possible leaves for the target pile on the workbench by excluding crease neighbors
	using leaves already pinned near the ends of the pile order. You receive leaf choices without
	modifying `state`.

	Parameters
	----------
	state : StateElimination
		Workbench state for a map whose axis lengths are all `2`. Set `state.pile` to
		the target pile and provide the pinned leaves in `state.permutationSpace`.

	Returns
	-------
	leavesToPin : Iterator[Leaf]
		Iterator over leaf indices after exclusions, in the original
		crease order.

	Pinned Piles
	------------
	Pile `state.pile - 1` must contain a pinned leaf.
	Piles `2` and `state.首 - 2` must also be pinned.

	A required pile without a pinned leaf causes `ValueError`.

	You select leaves with `mapFolding.creases.getLeavesCreasePost` [1]. Follow the
	pinning order used by `mapFolding.pinIt.pinPilesAtEnds` [2].

	References
	----------
	[1] `mapFolding.creases.getLeavesCreasePost`

	[2] `mapFolding.pinIt.pinPilesAtEnds`
	"""
	direction: CallableFunction[[int, int], int] = sub

	boxOfCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f"I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: 		Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
	leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))

	if 1 < len(tupleLeavesCrease):
		boxOfCreaseIndicesExcluded.append(0)
	if isEven吗(leafAt一Ante首) and (leafAt一 == (零) + 首零(state.totalDimensions)):
		boxOfCreaseIndicesExcluded.extend([*range(工dimensionTail(leafAt一Ante首) + 零, state.totalDimensions)])
	return exclude(tupleLeavesCrease, boxOfCreaseIndicesExcluded)

def pinPile零一Ante首ByCrease(state: StateElimination) -> Iterator[Leaf]:
	"""Iterate over crease neighbors for pile three positions before the end.

	(AI generated docstring)

	You can obtain possible leaves for the target pile on the workbench by excluding crease neighbors
	using leaves already pinned near the ends of the pile order. You receive leaf choices without
	modifying `state`.

	Parameters
	----------
	state : StateElimination
		Workbench state for a map whose axis lengths are all `2`. Set `state.pile` to
		the target pile and provide the pinned leaves in `state.permutationSpace`.

	Returns
	-------
	leavesToPin : Iterator[Leaf]
		Iterator over leaf indices after exclusions, in the original
		crease order.

	Pinned Piles
	------------
	Pile `state.pile + 1` must contain a pinned leaf.
	Piles `2` and `state.首 - 2` must also be pinned.

	A required pile without a pinned leaf causes `ValueError`.

	You select leaves with `mapFolding.creases.getLeavesCreaseAnte` [1]. Follow the
	pinning order used by `mapFolding.pinIt.pinPilesAtEnds` [2].

	References
	----------
	[1] `mapFolding.creases.getLeavesCreaseAnte`

	[2] `mapFolding.pinIt.pinPilesAtEnds`
	"""
	direction: CallableFunction[[int, int], int] = add

	boxOfCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f"I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: 		Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
	leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))

	if leafAt一Ante首 < 首零一(state.totalDimensions):
		boxOfCreaseIndicesExcluded.append(-1)
	if (leafAt一Ante首 == (零) + 首零(state.totalDimensions)) and (leafAt一 != 一 + 零):
		boxOfCreaseIndicesExcluded.extend([*range(工dimension首零(leafAt一) - 零)])
	return exclude(tupleLeavesCrease, boxOfCreaseIndicesExcluded)

#-------- Depth 4 ------------------------------------
def pinPile二ByCrease(state: StateElimination) -> Iterator[Leaf]:
	"""Iterate over crease neighbors for pile four.

	(AI generated docstring)

	You can obtain possible leaves for the target pile on the workbench by excluding crease neighbors
	using leaves already pinned near the ends of the pile order. You receive leaf choices without
	modifying `state`.

	Parameters
	----------
	state : StateElimination
		Workbench state for a map whose axis lengths are all `2`. Set `state.pile` to
		the target pile and provide the pinned leaves in `state.permutationSpace`.

	Returns
	-------
	leavesToPin : Iterator[Leaf]
		Iterator over leaf indices after exclusions, in the original
		crease order.

	Pinned Piles
	------------
	Pile `state.pile - 1` must contain a pinned leaf.
	Piles `2`, `3`, `state.首 - 2`, and `state.首 - 3` must also be pinned.

	A required pile without a pinned leaf causes `ValueError`.

	You select leaves with `mapFolding.creases.getLeavesCreasePost` [1]. Follow the
	pinning order used by `mapFolding.pinIt.pinPilesAtEnds` [2].

	References
	----------
	[1] `mapFolding.creases.getLeavesCreasePost`

	[2] `mapFolding.pinIt.pinPilesAtEnds`
	"""
	direction: CallableFunction[[int, int], int] = sub

	boxOfCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f"I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: 		Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
	leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))
	leafAt一零: 	  Leaf = raiseIfNone(state.permutationSpace.getLeaf(一 + 零))
	leafAt零一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(零 + 一) + state.首))

	if isOdd吗(leafAt一零):
		boxOfCreaseIndicesExcluded.extend([*range(工dimension首零(leafAt一零), 5), ptount(leafAt一零)])
		boxOfCreaseIndicesExcluded.append((dimensionIndex(leafInSubHyperplane(leafAt一Ante首)) + 4) % 5)
	if isEven吗(leafAt一零):
		boxOfCreaseIndicesExcluded.extend([*range(state.totalDimensions - 3)][(state.totalDimensions - 3) - ((state.totalDimensions - 2) - leafInSubHyperplane(leafAt零一Ante首 - (leafAt零一Ante首.bit_count() - isEven吗(leafAt零一Ante首))).bit_count()) % (state.totalDimensions - 2) - isEven吗(leafAt零一Ante首): None])
		if isEven吗(leafAt一Ante首):
			boxOfCreaseIndicesExcluded.extend([*range(工dimensionTail(leafInSubHyperplane(leafAt一Ante首)) - 一, (state.totalDimensions - 3))])
	if leafAt一 == (零) + 首零(state.totalDimensions):
		boxOfCreaseIndicesExcluded.extend([(dimensionIndex(leafInSubHyperplane(leafAt一Ante首)) + 4) % 5, 工dimensionTail(leafAt零一Ante首) - 1])
		if (零) + 首零(state.totalDimensions) < leafAt零一Ante首:
			boxOfCreaseIndicesExcluded.extend([*range(int(leafAt零一Ante首 - int(bit_flip(0, 工dimension首零(leafAt零一Ante首)))).bit_length() - 1, state.totalDimensions - 2)])
		if ((0 < leafAt一零 - leafAt一 <= bit_flip(0, state.totalDimensions - 4)) and (0 < (leafAt一Ante首 - leafAt一零) <= bit_flip(0, state.totalDimensions - 3))):
			boxOfCreaseIndicesExcluded.extend([ptount(leafAt一零), state.totalDimensions - 3, state.totalDimensions - 4])
	return exclude(tupleLeavesCrease, boxOfCreaseIndicesExcluded)

def pinPile二Ante首ByCrease(state: StateElimination) -> Iterator[Leaf]:
	"""Iterate over crease neighbors for pile four positions before the end.

	(AI generated docstring)

	You can obtain possible leaves for the target pile on the workbench by excluding crease neighbors
	using leaves already pinned near the ends of the pile order. You receive leaf choices without
	modifying `state`.

	Parameters
	----------
	state : StateElimination
		Workbench state for a map whose axis lengths are all `2`. Set `state.pile` to
		the target pile and provide the pinned leaves in `state.permutationSpace`.

	Returns
	-------
	leavesToPin : Iterator[Leaf]
		Iterator over leaf indices after exclusions, in the original
		crease order.

	Pinned Piles
	------------
	Pile `state.pile + 1` must contain a pinned leaf.
	Piles `2`, `3`, `4`, `state.首 - 2`, and `state.首 - 3` must also be pinned.

	A required pile without a pinned leaf causes `ValueError`.

	You select leaves with `mapFolding.creases.getLeavesCreaseAnte` [1]. Follow the
	pinning order used by `mapFolding.pinIt.pinPilesAtEnds` [2].

	References
	----------
	[1] `mapFolding.creases.getLeavesCreaseAnte`

	[2] `mapFolding.pinIt.pinPilesAtEnds`
	"""
	direction: CallableFunction[[int, int], int] = add

	boxOfCreaseIndicesExcluded: list[int] = []
	leafRoot: Leaf = raiseIfNone(state.permutationSpace.getLeaf(direction(state.pile, 1)), f"I could not find an `int` type `Leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[Leaf, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: 		Leaf = raiseIfNone(state.permutationSpace.getLeaf(一))
	leafAt一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(一) + state.首))
	leafAt一零: 	  Leaf = raiseIfNone(state.permutationSpace.getLeaf(一 + 零))
	leafAt零一Ante首: Leaf = raiseIfNone(state.permutationSpace.getLeaf(neg(零 + 一) + state.首))
	leafAt二: 		Leaf = raiseIfNone(state.permutationSpace.getLeaf(二))

	addendDimension首零: int = leafAt零一Ante首 - leafAt一Ante首

	addendDimension一零: int = leafAt二 - leafAt一零
	addendDimension一:	 int = leafAt一零 - leafAt一
	addendDimension零:	 int = leafAt一 - 零

	if ((addendDimension一零 in {一, 二, 三, 四})
		or ((addendDimension一零 == 五) and (addendDimension首零 != 一))
		or (addendDimension一 in {二, 三})
		or ((addendDimension一 == 一) and not (addendDimension零 == addendDimension首零 and addendDimension一零 < 0))
	):
		if leafAt零一Ante首 == 首一(state.totalDimensions):
			if addendDimension零 == 三:
				boxOfCreaseIndicesExcluded.append(dimensionIndex(二))
			if addendDimension零 == 五:
				if addendDimension一 == 二:
					boxOfCreaseIndicesExcluded.append(dimensionIndex(二))
				if addendDimension一 == 三:
					boxOfCreaseIndicesExcluded.append(dimensionIndex(三))
			if addendDimension一零 == 三:
				boxOfCreaseIndicesExcluded.append(dimensionIndex(二))

		if 0 < (dimensionTail := 工dimensionTail(leafAt零一Ante首)) < 5:
			boxOfCreaseIndicesExcluded.extend(list(range(dimensionTail % 4)) or [dimensionIndex(一)])

		if addendDimension首零 == neg(五):
			boxOfCreaseIndicesExcluded.append(dimensionIndex(一))
		if addendDimension首零 == 一:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(二))
		if addendDimension首零 == 四:
			if addendDimension零 == 三:
				boxOfCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
			if addendDimension一 == 一:
				if addendDimension一零 == 三:
					boxOfCreaseIndicesExcluded.append(dimensionIndex(二))

		if addendDimension零 == 一:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(一))
			if addendDimension一零 == 三:
				boxOfCreaseIndicesExcluded.extend([*range(dimensionIndex(二), dimensionIndex(三) + inclusive)])
			if addendDimension一零 == 四:
				boxOfCreaseIndicesExcluded.extend([*range(dimensionIndex(三), dimensionIndex(四) + inclusive)])
		if addendDimension零 == 二:
			boxOfCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
		if addendDimension零 == 三:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(三))

		if addendDimension一 == 二:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(一))
		if addendDimension一 == 三:
			boxOfCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(二) + inclusive)])
		if addendDimension一 == 四:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(一))
			if addendDimension一零 == 三:
				boxOfCreaseIndicesExcluded.extend([*range(dimensionIndex(一), dimensionIndex(三) + inclusive)])

		if addendDimension一零 == 一:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(一))
		if addendDimension一零 == 二:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(二))
		if addendDimension一零 == 三:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(三))
		if addendDimension一零 == 五:
			boxOfCreaseIndicesExcluded.append(dimensionIndex(一))

	return exclude(tupleLeavesCrease, boxOfCreaseIndicesExcluded)
