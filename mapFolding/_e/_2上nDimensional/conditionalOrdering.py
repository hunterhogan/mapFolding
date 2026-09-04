from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from functools import cache
from gmpy2 import is_even as isEven吗, is_odd as isOdd吗
from hunterMakesPy import decreasing, inclusive
from mapFolding._e import getLookupDomainsLeaves, getMapShape首ProductsSums
from mapFolding._e._2上nDimensional import leafInSubHyperplane, 一, 工dimensionTail, 工dimension首零, 工totalDimensionsOdd, 零, 首一, 首零, 首零一
from mapFolding._e.dataBaskets import StateElimination
from mapFolding.beDRY import mapShapeIs2上nDimensions
from operator import neg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.theTypes import Leaf, Pile

# IMPROVEMENT getDictionaryConditionalLeafPredecessors development
def getLeafPredecessors(state: StateElimination) -> dict[Leaf, dict[Pile, list[Leaf]]]:
	"""leaf: pile: [conditional `leafPredecessor`].

	Some leaves are always preceded by one or more leaves. Most leaves, however, are preceded by one or more other leaves only if
	the leaf is in a specific pile.
	"""
	dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = {}
	if mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToRideThis=6):
		dictionaryConditionalLeafPredecessors = _getDictionaryConditionalLeafPredecessors(state.mapShape)
	return dictionaryConditionalLeafPredecessors
@cache
def _getDictionaryConditionalLeafPredecessors(mapShape: tuple[int, ...]) -> dict[Leaf, dict[Pile, list[Leaf]]]:
	"""Prototype."""
	state = StateElimination(mapShape)
	dictionaryDomains: dict[Leaf, range] = getLookupDomainsLeaves(state)

	dictionaryPrecedence: dict[Leaf, dict[Pile, list[Leaf]]] = {}

#======== piles at the beginning of the leaf's domain ================
	for dimension in range(3, state.totalDimensions + inclusive):
		for countDown in range(dimension - 2 + decreasing, decreasing, decreasing):
			for leaf in range(state.mapShapeProducts[dimension] - sum(state.mapShapeProducts[countDown:dimension - 2]), state.totalLeaves, state.mapShapeProducts[dimension - 1]):
				dictionaryPrecedence[leaf] = {aPile: [state.mapShapeProducts[工dimension首零(leaf)] + state.mapShapeProducts[工dimensionTail(leaf)]]
							for aPile in list(dictionaryDomains[leaf])[0: getMapShape首ProductsSums(state.mapShapeProducts, dimensionFrom首=dimension - 1)[dimension - 2 - countDown] // 2]}

#-------- The beginning of domain首一Plus零 --------------------------------
	leaf = (零) + 首一(state.totalDimensions)
	dictionaryPrecedence[leaf] = {aPile: [2 * state.mapShapeProducts[工dimension首零(leaf)] + state.mapShapeProducts[工dimensionTail(leaf)]
										, 3 * state.mapShapeProducts[工dimension首零(leaf)] + state.mapShapeProducts[工dimensionTail(leaf)]]
							for aPile in list(dictionaryDomains[leaf])[1:2]}
	del leaf

#======== leaf首零一Plus零: conditional `leafPredecessor` in all piles of its domain ===========
	leaf: Leaf = (零) + 首零一(state.totalDimensions)
	boxOfPiles = list(dictionaryDomains[leaf])
	dictionaryPrecedence[leaf] = {aPile: [] for aPile in list(dictionaryDomains[leaf])}
	mapShape首ProductsSums: tuple[int, ...] = getMapShape首ProductsSums(state.mapShapeProducts)
	mapShape首ProductsSumsInSubHyperplane: tuple[int, ...] = getMapShape首ProductsSums(state.mapShapeProducts, dimensionFrom首=state.totalDimensions - 1)
	pileStepAbsolute = 2

	for aPile in boxOfPiles[boxOfPiles.index(一 + 零): boxOfPiles.index(neg(零) + 首零(state.totalDimensions)) + inclusive]:
		dictionaryPrecedence[leaf][aPile].append((零) + 首零(state.totalDimensions))

	for 次Universal in range(state.totalDimensions - 2):
		leafPredecessorTheFirst: int = state.mapShapeProductsSums[次Universal + 2]
		leavesPredecessorInThisSeries: int = state.mapShapeProducts[工totalDimensionsOdd(leafPredecessorTheFirst)]
		for addend in range(leavesPredecessorInThisSeries):
			leafPredecessor = leafPredecessorTheFirst + (addend * decreasing)
			pileFirst: int = (
				mapShape首ProductsSums[次Universal]
				+ state.mapShapeProductsSums[2]
				+ state.mapShapeProducts[state.totalDimensions - (次Universal + 2)]
				- ((pileStepAbsolute * 2 * (工totalDimensionsOdd(leafPredecessor) - 1 + isEven吗(leafPredecessor)))
					* (1 + (2 == (工totalDimensionsOdd(leafPredecessor) + isEven吗(leafPredecessor)) == 工dimension首零(leafPredecessor)))
				)
			)
			for aPile in boxOfPiles[boxOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor)

			leafPredecessor首零: int = leafPredecessor + 首零(state.totalDimensions)
			if (leafInSubHyperplane(leafPredecessor) == 0) and isOdd吗(工dimensionTail(leafPredecessor)):
				dictionaryPrecedence[leaf][pileFirst].append(leafPredecessor首零)
			if leafPredecessor首零 == leaf:
				continue
			pileFirst = boxOfPiles[-1] - (
					pileStepAbsolute * (
					工totalDimensionsOdd(leafPredecessor首零)
					- 1
					+ isEven吗(leafPredecessor首零)
					- isOdd吗(leafPredecessor首零)
					- int(工dimensionTail(leafPredecessor首零) == state.totalDimensions - 2)
					- int(leaf < leafPredecessor首零)
				))
			for aPile in boxOfPiles[boxOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

			if (次Universal < state.totalDimensions - 4) and isOdd吗(工dimensionTail(leafPredecessor - isOdd吗(leafPredecessor))):
				pileFirst = (
					mapShape首ProductsSumsInSubHyperplane[次Universal]
					+ state.mapShapeProductsSums[2 + 1 + 次Universal]
					- (pileStepAbsolute
						* 2
						* (工totalDimensionsOdd(leafPredecessor首零) - 1
							+ isEven吗(leafPredecessor首零) * 次Universal
							- isEven吗(leafPredecessor首零) * (int(not (bool(次Universal))))
						)
					)
					+ state.mapShapeProducts[state.totalDimensions - 1
												+ addend * (int(not (bool(次Universal))))
												- (次Universal + 2)]
				)
				for aPile in boxOfPiles[boxOfPiles.index(pileFirst) + 次Universal: boxOfPiles.index(neg(零) + 首零(state.totalDimensions)) - 次Universal + inclusive]:
					dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

	del leaf, boxOfPiles, mapShape首ProductsSums, pileStepAbsolute, mapShape首ProductsSumsInSubHyperplane

#======== leaf首零Plus零: Separate logic because the distance between absolute piles is 4, not 2 ==============
# leaf has conditional `leafPredecessor` in all but the first pile of its domain
# Reminder: has UNconditional `leafPredecessor` in the first pile: leaf零
	leaf: Leaf = (零) + 首零(state.totalDimensions)
	boxOfPiles: list[Pile] = list(dictionaryDomains[leaf])[1: None]
	dictionaryPrecedence[leaf] = {aPile: [] for aPile in boxOfPiles}
	mapShape首ProductsSums: tuple[int, ...] = getMapShape首ProductsSums(state.mapShapeProducts)
	pileStepAbsolute = 4
	for 次Universal in range(state.totalDimensions - 2):
		leafPredecessorTheFirst: int = state.mapShapeProductsSums[次Universal + 2]
		leavesPredecessorInThisSeries = state.mapShapeProducts[工totalDimensionsOdd(leafPredecessorTheFirst)]
		for addend in range(leavesPredecessorInThisSeries):
			leafPredecessor: int = leafPredecessorTheFirst + (addend * decreasing)
			leafPredecessor首零: int = leafPredecessor + 首零(state.totalDimensions)
			pileFirst = mapShape首ProductsSums[次Universal] + 6 - (pileStepAbsolute * (工totalDimensionsOdd(leafPredecessor) - 1 + isEven吗(leafPredecessor)))
			for aPile in boxOfPiles[boxOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor)
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

	del leaf, boxOfPiles, mapShape首ProductsSums, pileStepAbsolute

#======== piles at the end of the leaf's domain ================
#-------- Example of special case: has conditional `leafPredecessor` two steps before the end of the domain --------------------------
	if state.totalDimensions == 6:
		leaf = 22
		sliceOfPiles = slice(0, None)
		boxOfPiles = list(dictionaryDomains[leaf])[sliceOfPiles]
		leafPredecessorPileFirstPileLast = [(15, 43, 43)]
		for leafPredecessor, pileFirst, pileLast in leafPredecessorPileFirstPileLast:
			for pile in boxOfPiles[boxOfPiles.index(pileFirst): boxOfPiles.index(pileLast) + inclusive]:
				dictionaryPrecedence[leaf].setdefault(pile, []).append(leafPredecessor)

# REMINDER Some leaves, such as 16,48, have `leafPredecessor`, such as leaves 40 and 56, with a
# larger step size.

# DEVELOPMENT There are "knock-out" leaves, such as within the domain functions, above.
# (Discontinuities in the sequence of conditional leaf predecessors.) The "knock-out" leaves have
# patterns that I have not yet discovered: look for a crease relationship.

	return dictionaryPrecedence

# IMPROVEMENT getDictionaryConditionalLeafSuccessors development
def getLeafSuccessors(state: StateElimination) -> dict[Leaf, dict[Pile, list[Leaf]]]:
	"""leaf: pile: [conditional `leafSuccessor`]."""
	return _getDictionaryConditionalLeafSuccessors(state.mapShape)
@cache
def _getDictionaryConditionalLeafSuccessors(mapShape: tuple[int, ...]) -> dict[Leaf, dict[Pile, list[Leaf]]]:
	state = StateElimination(mapShape)
	dictionaryDomains: dict[Leaf, range] = getLookupDomainsLeaves(state)

	dictionarySuccessor: dict[Leaf, dict[Pile, list[Leaf]]] = {}

	dictionaryPrecedence: dict[Leaf, dict[Pile, list[Leaf]]] = getLeafPredecessors(state)

	for leafLater, dictionaryPiles in dictionaryPrecedence.items():
		boxOfDomainLater: tuple[Pile, ...] = tuple(dictionaryDomains[leafLater])
		dictionaryPilesByPredecessor: defaultdict[Leaf, set[Pile]] = defaultdict(set)
		for pileLater, boxOfLeafPredecessors in dictionaryPiles.items():
			for leafEarlier in boxOfLeafPredecessors:
				dictionaryPilesByPredecessor[leafEarlier].add(pileLater)

		for leafEarlier, boxOfPilesRequiring in dictionaryPilesByPredecessor.items():
			boxOfDomainEarlier: tuple[Pile, ...] = tuple(dictionaryDomains[leafEarlier])
			boxOfOptionalPiles: list[Pile] = sorted(pile for pile in boxOfDomainLater if pile not in boxOfPilesRequiring)
			for pileEarlier in boxOfDomainEarlier:
				optionalLessEqualCount: int = bisect_right(boxOfOptionalPiles, pileEarlier)
				if optionalLessEqualCount == 0:
					boxOfSuccessors: list[Leaf] = dictionarySuccessor.setdefault(leafEarlier, {}).setdefault(pileEarlier, [])
					if leafLater not in boxOfSuccessors:
						boxOfSuccessors.append(leafLater)

	return dictionarySuccessor
