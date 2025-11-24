# ruff: noqa ERA001
from collections.abc import Callable, Iterator
from copy import deepcopy
from cytoolz.dicttoolz import itemfilter, keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from itertools import chain, combinations, filterfalse, repeat, starmap
from mapFolding import decreasing, exclude, inclusive, reverseLookup
from mapFolding._e import (
	dimensionNearest首, dimensionSecondNearest首, getDictionaryAddends4Next, getDictionaryAddends4Prior,
	getDictionaryPileToLeaves, getLeafDomain, getListLeavesDecrease, getListLeavesIncrease, howMany0coordinatesAtTail,
	leafInSubHyperplane, leafOrigin, pileOrigin, PinnedLeaves, ptount, 一, 三, 二, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e.pinIt import atPilePinLeaf, deconstructPinnedLeavesAtPile
from mapFolding.algorithms.iff import pinnedLeavesHasAViolation
from mapFolding.dataBaskets import EliminationState
from math import log, prod
from more_itertools import interleave_longest
from operator import add
from pprint import pprint

def listPinnedLeavesDefault(state: EliminationState) -> EliminationState:
	state.listPinnedLeaves = [{pileOrigin: leafOrigin, 零: 零, state.leavesTotal - 零: 首零(state.dimensionsTotal)}]
	return state

# ======= append `pinnedLeaves` at `pile` if qualified =======

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, pinnedLeaves: PinnedLeaves) -> bool:
	beans, cornbread = 一+零, 一
	if (beans in pinnedLeaves.values()) ^ (cornbread in pinnedLeaves.values()):
		return True
	beans, cornbread = 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)
	if (beans in pinnedLeaves.values()) ^ (cornbread in pinnedLeaves.values()):
		return True
	return False

def appendPinnedLeavesAtPile(state: EliminationState, listLeavesAtPile: list[int]) -> EliminationState:
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	leavesToPin: list[int] = list(filterfalse(disqualify, listLeavesAtPile))
	dictionaryPinnedLeaves: dict[int, PinnedLeaves] = deconstructPinnedLeavesAtPile(state.pinnedLeaves, state.pile, leavesToPin)

	beansOrCornbread: Callable[[PinnedLeaves], bool] = beansWithoutCornbread(state)

	state.listPinnedLeaves.extend(valfilter(complement(beansOrCornbread), dictionaryPinnedLeaves).values())

	for leafBeans, pinnedLeaves in valfilter(beansOrCornbread, dictionaryPinnedLeaves).items():
		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, pinnedLeaves=pinnedLeaves)
		if leafBeans in [一+零, 首一(sherpa.dimensionsTotal)]:
			leafCornbread = getListLeavesIncrease(sherpa, leafBeans).pop()
			sherpa.pile += 1
		else:
			leafCornbread = getListLeavesDecrease(sherpa, leafBeans).pop()
			sherpa.pile -= 1

		if disqualifyAppendingLeafAtPile(sherpa, leafCornbread):
			continue
		state.listPinnedLeaves.append(atPilePinLeaf(sherpa.pinnedLeaves, sherpa.pile, leafCornbread))
		del sherpa

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: int) -> bool:
		return any([
			_alreadyPinned(state, leaf)
			, _tooSmall(state, leaf)
			, _tooLarge(state, leaf)
			, _pileNotInRange(state, leaf)
			, _pileOccupied(state, leaf)
			, _productsOfDimensionsNotInOrder(state, leaf)
			, _leafDimensionOriginNotFirst(state, leaf)
			, _leading0notBeforeTrailing0(state, leaf)
			# , _disqualifyAppendingLeafAtPileDueToDictionary(state, leaf)
		])

def _alreadyPinned(state: EliminationState, leaf: int) -> bool:
	return leaf in state.pinnedLeaves.values()

def _leading0notBeforeTrailing0(state: EliminationState, leaf: int) -> bool:
	pinnedLeavesAbove: PinnedLeaves = valfilter(lambda leafPinned: leaf < leafPinned, valfilter(_notLeafOriginOrLeaf零, state.pinnedLeaves))
	pinnedLeavesBelow: PinnedLeaves = valfilter(lambda leafPinned: leafPinned < leaf, valfilter(_notLeafOriginOrLeaf零, state.pinnedLeaves))
	compareThese: Iterator[tuple[tuple[int, int], tuple[int, int]]] = chain(zip(repeat((state.pile, leaf)), pinnedLeavesAbove.items()), zip(pinnedLeavesBelow.items(), repeat((state.pile, leaf))))
	return any((pileOf_k > pileOf_r for (pileOf_k, _k), (pileOf_r, _r) in filter(_moreLeading0thanTrailing0, compareThese)))

def _leafDimensionOriginNotFirst(state: EliminationState, leaf: int) -> bool:
	"""Each product of dimensions is a 'dimension origin' leaf: a `leafDimensionOrigin` always comes before multiples of itself in the `folding`."""
	listDimensionOrigins:  list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(零, state.dimensionsTotal + inclusive)]

	dimensionOrigin: int = leaf
	pileOfDimensionOrigin: int = state.pile
	if dimensionOrigin in listDimensionOrigins:
		for pile, leafPinned in state.pinnedLeaves.items():
			if (dimensionOrigin < leafPinned) and (leafPinned % dimensionOrigin == 0) and (pile < pileOfDimensionOrigin):
				return True

	pilesOpenBeforeLeaf: set[int] = set(range(state.pile)).difference(state.pinnedLeaves.keys())
	for dimensionOrigin in listDimensionOrigins:
		if (leaf != dimensionOrigin) and (leaf % dimensionOrigin == 0):
			if dimensionOrigin in state.pinnedLeaves.values():
				pileOfDimensionOrigin = reverseLookup(state.pinnedLeaves, dimensionOrigin)
				if state.pile < pileOfDimensionOrigin:
					return True
			else:
				pilesOpenBeforeLeaf = pilesOpenBeforeLeaf.intersection(getLeafDomain(state, dimensionOrigin))
				if not pilesOpenBeforeLeaf:
					return True
				else:
					pilesOpenBeforeLeaf.remove(min(pilesOpenBeforeLeaf))

	return False

def _pileNotInRange(state: EliminationState, leaf: int) -> bool:
	return state.pile not in list(getLeafDomain(state, leaf))

def _pileOccupied(state: EliminationState, leaf: int) -> bool:
	return state.pile in state.pinnedLeaves

def _productsOfDimensionsNotInOrder(state: EliminationState, leaf: int) -> bool:
	r: int = leaf
	pileOf_r: int = state.pile
	if r in state.productsOfDimensions[一:None]:
		pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.pinnedLeaves.keys())
		dimensionIndexOf_r: int = state.productsOfDimensions.index(r)
		for k in state.productsOfDimensions[一:dimensionIndexOf_r]:
			if k in state.pinnedLeaves.values():
				pileOf_k: int = reverseLookup(state.pinnedLeaves, k)
				if pileOf_r < pileOf_k:
					return True
			else:
				pilesOpenBefore_r = pilesOpenBefore_r.intersection(getLeafDomain(state, k))
				if not pilesOpenBefore_r:
					return True
				else:
					pilesOpenBefore_r.remove(min(pilesOpenBefore_r))

	k: int = leaf
	pileOf_k = state.pile
	if k in state.productsOfDimensions[pileOrigin:-(一)]:
		pilesOpenAfter_k: set[int] = set(range(pileOf_k, state.leavesTotal)).difference(state.pinnedLeaves.keys())
		dimensionIndexOf_k: int = state.productsOfDimensions.index(k)
		for r in state.productsOfDimensions[dimensionIndexOf_k + 零: -(一)]:
			if r in state.pinnedLeaves.values():
				pileOf_r = reverseLookup(state.pinnedLeaves, r)
				if pileOf_r < pileOf_k:
					return True
			else:
				pilesOpenAfter_k = pilesOpenAfter_k.intersection(getLeafDomain(state, r))
				if not pilesOpenAfter_k:
					return True
				else:
					pilesOpenAfter_k.remove(max(pilesOpenAfter_k))

	return False

def _tooLarge(state: EliminationState, leaf: int) -> bool:
	return leaf > state.leavesTotal - 零

def _tooSmall(state: EliminationState, leaf: int) -> bool:
	return leaf < 0

# ======= Dictionaries. =======

def _disqualifyAppendingLeafAtPileDueToDictionary(state: EliminationState, leaf: int) -> bool:
		stateCopy: EliminationState = deepcopy(state)
		stateCopy.pinnedLeaves[stateCopy.pile] = leaf
		return any([
			pinnedLeavesHasAViolation(stateCopy)
		])

def disqualifyDictionary(state: EliminationState) -> bool:
		return any([
			_leafInWrongPile(state)
			, _leading0notBeforeTrailing0dictionary(state.pinnedLeaves)
			, _duplicateLeaves(state.pinnedLeaves)
			, _kBefore_rDictionary(state)
			, _noPilesOpenFor一Dictionary(state)
			, _leafDimensionOriginNotFirstDictionary(state)
			, _productsOfDimensionsNotInOrderDictionary(state)
			, pinnedLeavesHasAViolation(state)
		])

def _kBefore_rDictionary(state: EliminationState) -> bool:
	for pileOf_r, r in itemfilter(_leafInFirstPile, valfilter(lambda leafPinned: 2 < leafPinned.bit_count(), state.pinnedLeaves)).items():
		k = int(bit_flip(0, dimensionNearest首(r)).bit_flip(howMany0coordinatesAtTail(r)))
		if k in state.pinnedLeaves.values():
			pileOf_k = reverseLookup(state.pinnedLeaves, k)
			if pileOf_k > pileOf_r:
				return True
		else:
			pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.pinnedLeaves.keys()).intersection(getLeafDomain(state, k))
			if not pilesOpenBefore_r:
				return True
	return False

def _duplicateLeaves(pinnedLeaves: PinnedLeaves) -> bool:
	return len(pinnedLeaves.values()) != len(set(pinnedLeaves.values()))

def _leading0notBeforeTrailing0dictionary(pinnedLeaves: PinnedLeaves) -> bool:
	return any((pileOf_k > pileOf_r for (pileOf_k, _k), (pileOf_r, _r)
			in filter(_moreLeading0thanTrailing0, combinations(sorted(valfilter(_notLeafOriginOrLeaf零, pinnedLeaves).items()), 2))))

def _leafDimensionOriginNotFirstDictionary(state: EliminationState) -> bool:
	listDimensionOrigins:  list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(零, state.dimensionsTotal + inclusive)]

	for dimensionOrigin in listDimensionOrigins:
		leavesInDimension: PinnedLeaves = valfilter(lambda leafPinned: (leafPinned > dimensionOrigin) and (leafPinned % dimensionOrigin == 0), state.pinnedLeaves)

		if not leavesInDimension:
			continue

		pile: int = min(leavesInDimension.keys())

		if dimensionOrigin in state.pinnedLeaves.values():
			pileOfDimensionOrigin: int = reverseLookup(state.pinnedLeaves, dimensionOrigin)
			if pile < pileOfDimensionOrigin:
				return True
		else:
			pilesOpenInDimension: set[int] = set(range(pile)).difference(state.pinnedLeaves.keys()).intersection(getLeafDomain(state, dimensionOrigin))

			if not pilesOpenInDimension:
				return True

	return False

def _leafInFirstPile(pileLeaf: tuple[int, int]) -> bool:
	return pileLeaf[0] == pileLeaf[1].bit_count() + (2**(howMany0coordinatesAtTail(pileLeaf[1]) + 1) - 2)

def _leafInWrongPile(state: EliminationState) -> bool:
	def pileNotInDomain(pile: int, leaf: int) -> bool:
		return pile not in list(getLeafDomain(state, leaf))
	return any(starmap(pileNotInDomain, valfilter(_notLeafOriginOrLeaf零, state.pinnedLeaves).items()))

def _moreLeading0thanTrailing0(tupleElement: tuple[tuple[int, int], tuple[int, int]]) -> bool:
	return dimensionNearest首(tupleElement[0][1]) <= howMany0coordinatesAtTail(tupleElement[1][1])

def _notLeafOriginOrLeaf零(leaf: int) -> bool:
	return 零 < leaf

def _noPilesOpenFor一Dictionary(state: EliminationState) -> bool:
	if (一 not in state.pinnedLeaves.values() and (一+零 not in state.pinnedLeaves.values())):
		pile = min(valfilter(lambda leafPinned: leafPinned % 一 == 0, valfilter(_notLeafOriginOrLeaf零, keyfilter(lambda pilePinned: pilePinned != state.pileLast, state.pinnedLeaves))).keys(), default=None)
		if pile:
			pilesOpenInDimension: set[int] = set(range(pile)).difference(state.pinnedLeaves.keys())
			domainOf一零 = list(getLeafDomain(state, 一+零))
			if not pilesOpenInDimension.intersection(domainOf一零):
				return True
			else:
				noPilesOpenFor一 = True
				for pileOpen in pilesOpenInDimension:
					if (pileOpen + 1 in pilesOpenInDimension) and (pileOpen in domainOf一零):
						noPilesOpenFor一 = False
				return noPilesOpenFor一
	return False

def _productsOfDimensionsNotInOrderDictionary(state: EliminationState) -> bool:
	productsOfDimensions: list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(state.dimensionsTotal + inclusive)]

	for pile, leaf in state.pinnedLeaves.items():
		r: int = leaf
		pileOf_r: int = pile
		if r in productsOfDimensions[一:None]:
			pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.pinnedLeaves.keys())
			dimensionIndexOf_r: int = productsOfDimensions.index(r)
			for k in productsOfDimensions[一:dimensionIndexOf_r]:
				if k in state.pinnedLeaves.values():
					pileOf_k: int = reverseLookup(state.pinnedLeaves, k)
					if pileOf_r < pileOf_k:
						return True
				else:
					pilesOpenBefore_r = pilesOpenBefore_r.intersection(getLeafDomain(state, k))
					if not pilesOpenBefore_r:
						return True
					else:
						pilesOpenBefore_r.remove(min(pilesOpenBefore_r))

		k: int = leaf
		pileOf_k = pile
		if k in productsOfDimensions[pileOrigin:-(一)]:
			pilesOpenAfter_k: set[int] = set(range(pileOf_k, state.leavesTotal)).difference(state.pinnedLeaves.keys())
			dimensionIndexOf_k: int = productsOfDimensions.index(k)
			for r in productsOfDimensions[dimensionIndexOf_k + 零: -(一)]:
				if r in state.pinnedLeaves.values():
					pileOf_r = reverseLookup(state.pinnedLeaves, r)
					if pileOf_r < pileOf_k:
						return True
				else:
					pilesOpenAfter_k = pilesOpenAfter_k.intersection(getLeafDomain(state, r))
					if not pilesOpenAfter_k:
						return True
					else:
						pilesOpenAfter_k.remove(min(pilesOpenAfter_k))

	return False

# ======= Subroutines for analyzing a specific `pile`. =======

def addendsToListLeavesAtPile(listAddends: list[int], leafAddend: int, listIndicesExcluded: list[int]) -> list[int]:
	return list(map(add, repeat(leafAddend), exclude(listAddends, listIndicesExcluded)))

def pinPileOriginFixed(state: EliminationState) -> list[int]:
	listLeavesAtPile: list[int] = [leafOrigin]
	return listLeavesAtPile

def pinPile零Fixed(state: EliminationState) -> list[int]:
	listLeavesAtPile: list[int] = [零]
	return listLeavesAtPile

def pinPile一Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	leafAt首Less一: int | None = state.pinnedLeaves.get(state.leavesTotal - 一)
	if leafAt首Less一 and (0 < howMany0coordinatesAtTail(leafAt首Less一)):
		listAddendIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) - 1, state.dimensionsTotal - 2)])
	return addendsToListLeavesAtPile(getDictionaryAddends4Next(state)[leafAtPileLess1], leafAtPileLess1, listAddendIndicesExcluded)

def pinPile一零Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	listAddendIndicesExcluded.append(0)
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	if is_even(leafAt首Less一) and (leafAtPileLess1 == 首零(state.dimensionsTotal)+零):
		listAddendIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) + 1, state.dimensionsTotal)])
	return addendsToListLeavesAtPile(getDictionaryAddends4Next(state)[leafAtPileLess1], leafAtPileLess1, listAddendIndicesExcluded)

def pinPile二Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	if is_odd(leafAtPileLess1):
		listAddendIndicesExcluded.extend([*range(dimensionNearest首(leafAtPileLess1), 5), ptount(leafAtPileLess1)])
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	if is_even(leafAtPileLess1) and is_even(leafAt首Less一):
		listAddendIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafInSubHyperplane(leafAt首Less一)) - 一, (state.dimensionsTotal - 3))])
	if is_odd(leafAtPileLess1):
		listAddendIndicesExcluded.append((int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5)
	leafAt首Less一零: int = state.pinnedLeaves[state.leavesTotal - (一+零)]
	if is_even(leafAtPileLess1) and leafAt首Less一零:
		listAddendIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafInSubHyperplane(leafAt首Less一零 - (leafAt首Less一零.bit_count() - is_even(leafAt首Less一零))).bit_count()) % (state.dimensionsTotal - 2) - is_even(leafAt首Less一零): None])  # noqa: E501
	leafAt一: int = state.pinnedLeaves[一]
	if (leafAt一 == 首零(state.dimensionsTotal)+零):
		listAddendIndicesExcluded.extend([(int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5, howMany0coordinatesAtTail(leafAt首Less一零) - 1])
	if (leafAt一 == 首零(state.dimensionsTotal)+零) and (leafAt首Less一零 > 首零(state.dimensionsTotal)+零):
		listAddendIndicesExcluded.extend([*range(int(leafAt首Less一零 - 2**(dimensionNearest首(leafAt首Less一零))).bit_length() - 1, state.dimensionsTotal - 2)])
	if ((leafAt一 == 首零(state.dimensionsTotal)+零) and (0 < leafAtPileLess1 - leafAt一 <= 2**(state.dimensionsTotal - 4)) and (0 < (leafAt首Less一 - leafAtPileLess1) <= 2**(state.dimensionsTotal - 3))):
		listAddendIndicesExcluded.extend([ptount(leafAtPileLess1), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return addendsToListLeavesAtPile(getDictionaryAddends4Next(state)[leafAtPileLess1], leafAtPileLess1, listAddendIndicesExcluded)

def pinPile首零Less零Leaf(state: EliminationState) -> list[int]:
	leaf: int = -1
	sumsProductsOfDimensions: list[int] = [sum(state.productsOfDimensions[0:dimension]) for dimension in range(state.dimensionsTotal + inclusive)]

	dictionaryPileToLeaves: dict[int, list[int]] = getDictionaryPileToLeaves(state)
	listRemoveLeaves: list[int] = []

	pileExcluder: int = 一
	leafAtPileExcluder: int = state.pinnedLeaves[pileExcluder]
	for dimension in range(state.dimensionsTotal):
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一, 首零(state.dimensionsTotal) + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一 + leafAtPileExcluder])
		if dimension == 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + leafAtPileExcluder + 零])
		if dimension == state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal), 首一(state.dimensionsTotal) + leafAtPileExcluder])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - 一
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	for dimension in range(state.dimensionsTotal):
		if dimension == 0:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一])
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([2**dimension, 首一(state.dimensionsTotal) + leafAtPileExcluder - (2**dimension - 零)])
		if 0 < dimension < state.dimensionsTotal - 3:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([零 + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal)])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = 一+零
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	if leafAtPileExcluder == 三+二+零:
		listRemoveLeaves.extend([二+一+零, 首零(state.dimensionsTotal)+二+零])
	if leafAtPileExcluder == 首一(state.dimensionsTotal)+二+零:
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首一二(state.dimensionsTotal)+零, 首零一二(state.dimensionsTotal)])
	if leafAtPileExcluder == 首一二(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零二(state.dimensionsTotal)+零])
	if leafAtPileExcluder == 首零一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	if is_odd(leafAtPileExcluder):
		listRemoveLeaves.extend([leafAtPileExcluder, state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder))]])
		if leafAtPileExcluder < 首零(state.dimensionsTotal):
			comebackOffset: int = sumsProductsOfDimensions[ptount(leafAtPileExcluder) + 1]
			listRemoveLeaves.extend([
				一
				, leafAtPileExcluder + 首零(state.dimensionsTotal)-零
				, leafAtPileExcluder + 首零(state.dimensionsTotal)-零 - comebackOffset
			])
			if ptount(leafAtPileExcluder) == 1:
				listRemoveLeaves.extend([
					state.productsOfDimensions[dimensionNearest首(leafAtPileExcluder)] + comebackOffset
					, 首零(state.dimensionsTotal) + comebackOffset
				])
		if 首零(state.dimensionsTotal) < leafAtPileExcluder:
			listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, state.productsOfDimensions[dimensionNearest首(leafAtPileExcluder) - 1]])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - (一+零)
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	if 首零(state.dimensionsTotal) < leafAtPileExcluder:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, leafAtPileExcluder])
		if is_even(leafAtPileExcluder):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			bit = 1
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 2
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				if 1 < howMany0coordinatesAtTail(leafAtPileExcluder):
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 3
			if bit_test(leafAtPileExcluder, bit):
				if 1 < howMany0coordinatesAtTail(leafAtPileExcluder):
					listRemoveLeaves.extend([2**bit])
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
				if howMany0coordinatesAtTail(leafAtPileExcluder) < bit:
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])

			sheepOrGoat = 0
			shepherdOfDimensions: int = 2**(state.dimensionsTotal - 5)
			if (leafAtPileExcluder//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveLeaves.extend([0b000100])
				sheepOrGoat = ptount(leafAtPileExcluder//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset = 2**dimensionNearest首(leafAtPileExcluder) - 0b100
					listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = 2**raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder)) - 0b100
					listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])

		if is_odd(leafAtPileExcluder):
			listRemoveLeaves.extend([一])
			if leafAtPileExcluder & bit_mask(4) == 0b001001:
				listRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAtPileExcluder)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = 2**dimensionNearest首(leafAtPileExcluder) - 0b10
				listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = 2**raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder)) - 0b10
				listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])

	pileExcluder = 二
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]

	if is_even(leafAtPileExcluder):
		listRemoveLeaves.extend([一, leafAtPileExcluder + 1, 首零(state.dimensionsTotal)+一+零])
	if is_odd(leafAtPileExcluder):
		listRemoveLeaves.extend([leafAtPileExcluder - 1])
		if 首一(state.dimensionsTotal) < leafAtPileExcluder < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零一(state.dimensionsTotal)+零])
		if 首零(state.dimensionsTotal) < leafAtPileExcluder:
			listRemoveLeaves.extend([首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)+零])
			bit = 1
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 2
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 3
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 4
			if bit_test(leafAtPileExcluder, bit) and (leafAtPileExcluder.bit_length() > 5):
				listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	leafAt一: int = state.pinnedLeaves[一]
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	leafAt一零: int = state.pinnedLeaves[一+零]
	leafAt首Less一零: int = state.pinnedLeaves[state.leavesTotal - (一+零)]

	if (leafAt一零 != 首零一(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.append(一)
	if (leafAt首Less一零 != 首零(state.dimensionsTotal)+零 + getDictionaryAddends4Prior(state)[首零(state.dimensionsTotal)+零][0]) and (leafAt一 == 一+零):
		listRemoveLeaves.append(首一(state.dimensionsTotal))
	if (leafAt一 == 首二(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less一 + 零])
	if leafAt一.bit_length() < state.dimensionsTotal - 2:
		listRemoveLeaves.extend([一, leafAt首Less一 + 一])

	return sorted(set(dictionaryPileToLeaves[state.pile]).difference(set(listRemoveLeaves)))

def pinPile首Less一Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	leafAtPilePlus1: int = state.pinnedLeaves[state.pile + 1]
	leafAt一: int | None = state.pinnedLeaves.get(一)
	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listAddendIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return addendsToListLeavesAtPile(getDictionaryAddends4Prior(state)[leafAtPilePlus1], leafAtPilePlus1, listAddendIndicesExcluded)

def pinPile首Less一零Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	leafAtPilePlus1: int = state.pinnedLeaves[state.pile + 1]
	if leafAtPilePlus1 < 首零一(state.dimensionsTotal):
		listAddendIndicesExcluded.append(-1)
	leafAt一: int = state.pinnedLeaves[一]
	if (leafAtPilePlus1 == 首零(state.dimensionsTotal)+零) and (leafAt一 != 一+零):
		listAddendIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 1)])
	return addendsToListLeavesAtPile(getDictionaryAddends4Prior(state)[leafAtPilePlus1], leafAtPilePlus1, listAddendIndicesExcluded)

def pinPile首Less零Fixed(state: EliminationState) -> list[int]:
	listLeavesAtPile: list[int] = [首零(state.dimensionsTotal)]
	return listLeavesAtPile

# ======= Flow control ===============================================

def nextPinnedLeavesWorkbench(state: EliminationState, pileProcessingOrder: list[int] | None = None, queueStopBefore: int | None = None) -> EliminationState:
	if pileProcessingOrder is None:
		pileProcessingOrder = pileProcessingOrderDefault(state)

	# NOTE If you delete this, there will be an infinite loop and you will be sad.
	state.pinnedLeaves = {}

	for pile in pileProcessingOrder:
		if pile == queueStopBefore:
			break

		for pinnedLeaves in filterfalse(lambda dictionary: pile in dictionary, state.listPinnedLeaves):
			state.pinnedLeaves = pinnedLeaves
			state.listPinnedLeaves.remove(pinnedLeaves)
			state.pile = pile
			return state
	return state

def pileProcessingOrderDefault(state: EliminationState) -> list[int]:
	pileProcessingOrder: list[int] = [pileOrigin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])
	pileProcessingOrder.extend(interleave_longest(range(一, 首零(state.dimensionsTotal)), range(state.leavesTotal - (一+零), 首零(state.dimensionsTotal) + decreasing, decreasing)))
	return pileProcessingOrder

