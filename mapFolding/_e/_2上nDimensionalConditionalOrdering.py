# ruff: noqa: SIM102 DOC201
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from functools import cache
from gmpy2 import is_even as isEven吗, is_odd as isOdd吗
from hunterMakesPy import decreasing, inclusive
from mapFolding._e import (
	dimensionNearestTail, dimensionNearest首, getDictionaryLeafDomains, getSumsOfProductsOfDimensionsNearest首, howManyDimensionsHaveOddParity,
	leafInSubHyperplane, mapShapeIs2上nDimensions, 一, 零, 首一, 首零, 首零一)
from mapFolding._e.dataBaskets import EliminationState
from operator import neg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.theTypes import Leaf, Pile

# TODO getDictionaryConditionalLeafPredecessors development
def getDictionaryConditionalLeafPredecessors(state: EliminationState) -> dict[Leaf, dict[Pile, list[Leaf]]]:
	"""leaf: pile: [conditional `leafPredecessor`].

	Some leaves are always preceded by one or more leaves. Most leaves, however, are preceded by one or more other leaves only if
	the leaf is in a specific pile.
	"""
	dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = {}
	if mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
		dictionaryConditionalLeafPredecessors = _getDictionaryConditionalLeafPredecessors(state.mapShape)
	return dictionaryConditionalLeafPredecessors
@cache
def _getDictionaryConditionalLeafPredecessors(mapShape: tuple[int, ...]) -> dict[Leaf, dict[Pile, list[Leaf]]]:
	"""Prototype."""
	state = EliminationState(mapShape)
	dictionaryDomains: dict[Leaf, range] = getDictionaryLeafDomains(state)

	dictionaryPrecedence: dict[Leaf, dict[Pile, list[Leaf]]] = {}

#======== piles at the beginning of the leaf's domain ================
	for dimension in range(3, state.dimensionsTotal + inclusive):
		for countDown in range(dimension - 2 + decreasing, decreasing, decreasing):
			for leaf in range(state.productsOfDimensions[dimension] - sum(state.productsOfDimensions[countDown:dimension - 2]), state.leavesTotal, state.productsOfDimensions[dimension - 1]):
				dictionaryPrecedence[leaf] = {aPile: [state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]]
							for aPile in list(dictionaryDomains[leaf])[0: getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, dimensionFrom首=dimension - 1)[dimension - 2 - countDown] // 2]}

#-------- The beginning of domain首一Plus零 --------------------------------
	leaf = (零) + 首一(state.dimensionsTotal)
	dictionaryPrecedence[leaf] = {aPile: [2 * state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]
										, 3 * state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]]
							for aPile in list(dictionaryDomains[leaf])[1:2]}
	del leaf

#======== leaf首零一Plus零: conditional `leafPredecessor` in all piles of its domain ===========
	leaf: Leaf = (零) + 首零一(state.dimensionsTotal)
	listOfPiles = list(dictionaryDomains[leaf])
	dictionaryPrecedence[leaf] = {aPile: [] for aPile in list(dictionaryDomains[leaf])}
	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions)
	sumsOfProductsOfDimensionsNearest首InSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, dimensionFrom首=state.dimensionsTotal - 1)
	pileStepAbsolute = 2

	for aPile in listOfPiles[listOfPiles.index(一 + 零): listOfPiles.index(neg(零) + 首零(state.dimensionsTotal)) + inclusive]:
		dictionaryPrecedence[leaf][aPile].append((零) + 首零(state.dimensionsTotal))

	for indexUniversal in range(state.dimensionsTotal - 2):
		leafPredecessorTheFirst: int = state.sumsOfProductsOfDimensions[indexUniversal + 2]
		leavesPredecessorInThisSeries: int = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
		for addend in range(leavesPredecessorInThisSeries):
			leafPredecessor = leafPredecessorTheFirst + (addend * decreasing)
			pileFirst: int = (
				sumsOfProductsOfDimensionsNearest首[indexUniversal]
				+ state.sumsOfProductsOfDimensions[2]
				+ state.productsOfDimensions[state.dimensionsTotal - (indexUniversal + 2)]
				- ((pileStepAbsolute * 2 * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + isEven吗(leafPredecessor)))
					* (1 + (2 == (howManyDimensionsHaveOddParity(leafPredecessor) + isEven吗(leafPredecessor)) == dimensionNearest首(leafPredecessor)))
				)
			)
			for aPile in listOfPiles[listOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor)

			leafPredecessor首零: int = leafPredecessor + 首零(state.dimensionsTotal)
			if (leafInSubHyperplane(leafPredecessor) == 0) and isOdd吗(dimensionNearestTail(leafPredecessor)):
				dictionaryPrecedence[leaf][pileFirst].append(leafPredecessor首零)
			if leafPredecessor首零 == leaf:
				continue
			pileFirst = listOfPiles[-1] - (
					pileStepAbsolute * (
					howManyDimensionsHaveOddParity(leafPredecessor首零)
					- 1
					+ isEven吗(leafPredecessor首零)
					- isOdd吗(leafPredecessor首零)
					- int(dimensionNearestTail(leafPredecessor首零) == state.dimensionsTotal - 2)
					- int(leaf < leafPredecessor首零)
				))
			for aPile in listOfPiles[listOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

			if indexUniversal < state.dimensionsTotal - 4:
				if isOdd吗(dimensionNearestTail(leafPredecessor - isOdd吗(leafPredecessor))):
					pileFirst = (
						sumsOfProductsOfDimensionsNearest首InSubHyperplane[indexUniversal]
						+ state.sumsOfProductsOfDimensions[2 + 1 + indexUniversal]
						- (pileStepAbsolute
							* 2
							* (howManyDimensionsHaveOddParity(leafPredecessor首零) - 1
								+ isEven吗(leafPredecessor首零) * indexUniversal
								- isEven吗(leafPredecessor首零) * (int(not (bool(indexUniversal))))
							)
						)
						+ state.productsOfDimensions[state.dimensionsTotal - 1
													+ addend * (int(not (bool(indexUniversal))))
													- (indexUniversal + 2)]
					)
					for aPile in listOfPiles[listOfPiles.index(pileFirst) + indexUniversal: listOfPiles.index(neg(零) + 首零(state.dimensionsTotal)) - indexUniversal + inclusive]:
						dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

	del leaf, listOfPiles, sumsOfProductsOfDimensionsNearest首, pileStepAbsolute, sumsOfProductsOfDimensionsNearest首InSubHyperplane

#======== leaf首零Plus零: Separate logic because the distance between absolute piles is 4, not 2 ==============
# leaf has conditional `leafPredecessor` in all but the first pile of its domain
# Reminder: has UNconditional `leafPredecessor` in the first pile: leaf零
	leaf: Leaf = (零) + 首零(state.dimensionsTotal)
	listOfPiles: list[Pile] = list(dictionaryDomains[leaf])[1: None]
	dictionaryPrecedence[leaf] = {aPile: [] for aPile in listOfPiles}
	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions)
	pileStepAbsolute = 4
	for indexUniversal in range(state.dimensionsTotal - 2):
		leafPredecessorTheFirst: int = state.sumsOfProductsOfDimensions[indexUniversal + 2]
		leavesPredecessorInThisSeries = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
		for addend in range(leavesPredecessorInThisSeries):
			leafPredecessor: int = leafPredecessorTheFirst + (addend * decreasing)
			leafPredecessor首零: int = leafPredecessor + 首零(state.dimensionsTotal)
			pileFirst = sumsOfProductsOfDimensionsNearest首[indexUniversal] + 6 - (pileStepAbsolute * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + isEven吗(leafPredecessor)))
			for aPile in listOfPiles[listOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor)
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

	del leaf, listOfPiles, sumsOfProductsOfDimensionsNearest首, pileStepAbsolute

#======== piles at the end of the leaf's domain ================
#-------- Example of special case: has conditional `leafPredecessor` two steps before the end of the domain --------------------------
	if state.dimensionsTotal == 6:
		leaf = 22
		sliceOfPiles = slice(0, None)
		listOfPiles = list(dictionaryDomains[leaf])[sliceOfPiles]
		leafPredecessorPileFirstPileLast = [(15, 43, 43)]
		for leafPredecessor, pileFirst, pileLast in leafPredecessorPileFirstPileLast:
			for pile in listOfPiles[listOfPiles.index(pileFirst): listOfPiles.index(pileLast) + inclusive]:
				dictionaryPrecedence[leaf].setdefault(pile, []).append(leafPredecessor)

# NOTE Some leaves, such as 16,48, have `leafPredecessor`, such as leaves 40 and 56, with a larger step size.
# NOTE There may be "knock-out" leaves, such as within the domain functions, above. Or I might have to find complex formulas, such
# as in `pinPile二Crease`. Or, more likely, "knock-out" leaves might be complex formulas that I have not yet discovered.

	return dictionaryPrecedence

# TODO getDictionaryConditionalLeafSuccessors development
def getDictionaryConditionalLeafSuccessors(state: EliminationState) -> dict[Leaf, dict[Pile, list[Leaf]]]:
	"""leaf: pile: [conditional `leafSuccessor`]."""
	return _getDictionaryConditionalLeafSuccessors(state.mapShape)
@cache
def _getDictionaryConditionalLeafSuccessors(mapShape: tuple[int, ...]) -> dict[Leaf, dict[Pile, list[Leaf]]]:
	state = EliminationState(mapShape)
	dictionaryDomains: dict[Leaf, range] = getDictionaryLeafDomains(state)

	dictionarySuccessor: dict[Leaf, dict[Pile, list[Leaf]]] = {}

	dictionaryPrecedence: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)

	for leafLater, dictionaryPiles in dictionaryPrecedence.items():
		tupleDomainLater: tuple[Pile, ...] = tuple(dictionaryDomains[leafLater])
		dictionaryPilesByPredecessor: defaultdict[Leaf, set[Pile]] = defaultdict(set)
		for pileLater, listLeafPredecessors in dictionaryPiles.items():
			for leafEarlier in listLeafPredecessors:
				dictionaryPilesByPredecessor[leafEarlier].add(pileLater)

		for leafEarlier, setPilesRequiring in dictionaryPilesByPredecessor.items():
			tupleDomainEarlier: tuple[Pile, ...] = tuple(dictionaryDomains[leafEarlier])
			listOptionalPiles: list[Pile] = sorted(pile for pile in tupleDomainLater if pile not in setPilesRequiring)
			for pileEarlier in tupleDomainEarlier:
				optionalLessEqualCount: int = bisect_right(listOptionalPiles, pileEarlier)
				if optionalLessEqualCount == 0:
					listSuccessors: list[Leaf] = dictionarySuccessor.setdefault(leafEarlier, {}).setdefault(pileEarlier, [])
					if leafLater not in listSuccessors:
						listSuccessors.append(leafLater)

	return dictionarySuccessor
