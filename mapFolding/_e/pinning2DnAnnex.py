from cytoolz.dicttoolz import itemfilter, keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from gmpy2 import bit_flip, is_even, is_odd
from itertools import chain, combinations, filterfalse, repeat
from mapFolding import exclude, inclusive, reverseLookup
from mapFolding._e import (
	dictionary2d5AtPileLeafExcludedByPile, dictionary2d5LeafExcludedAtPileByPile, dictionary2d6AtPileLeafExcludedByPile,
	dictionary2d6LeafExcludedAtPileByPile, dimensionNearest首, getLeafDomain, getListLeavesDecrease, getListLeavesIncrease,
	howMany0coordinatesAtTail, leafInSubHyperplane, leafOrigin, pileOrigin, PinnedLeaves, ptount, 一, 二, 零, 首一, 首零, 首零一)
from mapFolding._e.pinIt import (
	atPilePinLeaf, deconstructPinnedLeavesAtPile, leafIsNotPinned, pileIsOpen, requireLeafPinnedAtPile)
from mapFolding.algorithms.iff import pinnedLeavesHasAViolation
from mapFolding.dataBaskets import EliminationState
from math import log, prod
from more_itertools import is_sorted
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator

#  ====== Boolean filters ======================

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, pinnedLeaves: PinnedLeaves) -> bool:
	return any((beans in pinnedLeaves.values()) ^ (cornbread in pinnedLeaves.values()) for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))))

@syntacticCurry
def _aDimensionOriginOfLeaf(leafDimensionOrigin: int, leaf: int) -> bool:
	return (leafDimensionOrigin == leafOrigin) or ((leafDimensionOrigin < leaf) and (leaf % leafDimensionOrigin == 0))

def _leafInFirstPileOfDomain(pileLeaf: tuple[int, int]) -> bool:
	return pileLeaf[0] == pileLeaf[1].bit_count() + (2**(howMany0coordinatesAtTail(pileLeaf[1]) + 1) - 2)

def _moreLeading0thanTrailing0(tupleElement: tuple[tuple[int, int], tuple[int, int]]) -> bool:
	return dimensionNearest首(tupleElement[0][1]) <= howMany0coordinatesAtTail(tupleElement[1][1])

def notLeafOriginOrLeaf零(leaf: int) -> bool:
	return 零 < leaf

# ======= append `pinnedLeaves` at `pile` if qualified =======

def appendPinnedLeavesAtPile(state: EliminationState, listLeavesAtPile: list[int]) -> EliminationState:
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	leavesToPin: list[int] = list(filterfalse(disqualify, listLeavesAtPile))

	dictionaryPinnedLeaves: dict[int, PinnedLeaves] = deconstructPinnedLeavesAtPile(state.pinnedLeaves, state.pile, leavesToPin)

	beansOrCornbread: Callable[[PinnedLeaves], bool] = beansWithoutCornbread(state)

	state.listPinnedLeaves.extend(tuple(valfilter(complement(beansOrCornbread), dictionaryPinnedLeaves).values()))

	for leafBeans, pinnedLeaves in tuple(valfilter(beansOrCornbread, dictionaryPinnedLeaves).items()):
		stateSherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, pinnedLeaves=pinnedLeaves)
		if leafBeans in [一+零, 首一(stateSherpa.dimensionsTotal)]:
			leafCornbread = getListLeavesIncrease(stateSherpa, leafBeans).pop()
			stateSherpa.pile += 1
		else:
			leafCornbread = getListLeavesDecrease(stateSherpa, leafBeans).pop()
			stateSherpa.pile -= 1

		if disqualifyAppendingLeafAtPile(stateSherpa, leafCornbread):
			continue
		state.listPinnedLeaves.append(atPilePinLeaf(stateSherpa.pinnedLeaves, stateSherpa.pile, leafCornbread))
		del stateSherpa

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: int) -> bool:
		return any([
			_pileNotInRange(state, leaf)
			, _leafDimensionOriginsNotInOrder(state, leaf)
			, _leafDimensionOriginNotFirst(state, leaf)
			, _leading0notBeforeTrailing0(state, leaf)
			# , _Z0Z_excludeThisLeaf(state, leaf)
		])

def _leading0notBeforeTrailing0(state: EliminationState, leaf: int) -> bool:
	pinnedLeavesAbove: PinnedLeaves = valfilter(lambda leafPinned: leaf < leafPinned, valfilter(notLeafOriginOrLeaf零, state.pinnedLeaves))
	pinnedLeavesBelow: PinnedLeaves = valfilter(lambda leafPinned: leafPinned < leaf, valfilter(notLeafOriginOrLeaf零, state.pinnedLeaves))
	compareThese: Iterator[tuple[tuple[int, int], tuple[int, int]]] = chain(zip(repeat((state.pile, leaf)), pinnedLeavesAbove.items()), zip(pinnedLeavesBelow.items(), repeat((state.pile, leaf))))
	return any((pileOf_k > pileOf_r for (pileOf_k, _k), (pileOf_r, _r) in filter(_moreLeading0thanTrailing0, compareThese)))

def _leafDimensionOriginNotFirst(state: EliminationState, leaf: int) -> bool:
	"""Each product of dimensions is a 'dimension origin' leaf: a `leafDimensionOrigin` always comes before multiples of itself in the `folding`."""
	listDimensionOrigins:  list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(零, state.dimensionsTotal + inclusive)]

	dimensionOrigin: int = leaf
	pileOfDimensionOrigin = state.pile
	if dimensionOrigin in listDimensionOrigins:
		for pile, leafPinned in state.pinnedLeaves.items():
			if (dimensionOrigin < leafPinned) and (leafPinned % dimensionOrigin == 0) and (pile < pileOfDimensionOrigin):
				return True

	pilesOpenBeforeLeaf: set[int] = set(range(state.pile)).difference(state.pinnedLeaves.keys())
	for dimensionOrigin in listDimensionOrigins:
		if (leaf != dimensionOrigin) and (leaf % dimensionOrigin == 0):
			if pileOfDimensionOrigin := reverseLookup(state.pinnedLeaves, dimensionOrigin):
				if state.pile < pileOfDimensionOrigin:
					return True
			elif not pilesOpenBeforeLeaf.intersection(getLeafDomain(state, dimensionOrigin)):
				return True

	return False

def _pileNotInRange(state: EliminationState, leaf: int) -> bool:
	return state.pile not in list(getLeafDomain(state, leaf))

def _leafDimensionOriginsNotInOrder(state: EliminationState, leaf: int) -> bool:
	r: int = leaf
	pileOf_r = state.pile
	if r in state.productsOfDimensions[一:None]:
		pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.pinnedLeaves.keys())
		dimensionIndexOf_r: int = state.productsOfDimensions.index(r)
		for k in state.productsOfDimensions[一:dimensionIndexOf_r]:
			if pileOf_k := reverseLookup(state.pinnedLeaves, k):
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
			if pileOf_r := reverseLookup(state.pinnedLeaves, r):
				if pileOf_r < pileOf_k:
					return True
			else:
				pilesOpenAfter_k = pilesOpenAfter_k.intersection(getLeafDomain(state, r))
				if not pilesOpenAfter_k:
					return True
				else:
					pilesOpenAfter_k.remove(max(pilesOpenAfter_k))

	return False

def _Z0Z_excludeThisLeaf(state: EliminationState, leaf: int) -> bool:
	"""{leafExcluded: {atPileExcluded: {byPileExcluder: listLeafExcluders}}}."""
	if state.dimensionsTotal == 6:
		lookup: dict[int, dict[int, dict[int, list[int]]]] = dictionary2d6LeafExcludedAtPileByPile
	if state.dimensionsTotal == 5:
		lookup = dictionary2d5LeafExcludedAtPileByPile
	else:
		return False

	if leaf in lookup and state.pile in lookup[leaf]:
		for pileExcluder, listLeafExcluders in keyfilter(mappingHasKey(state.pinnedLeaves), lookup[leaf][state.pile]).items():
			leafExcluder: int = state.pinnedLeaves[pileExcluder]
			if leafExcluder in listLeafExcluders:
				return True

	return False

# ======= Remove or disqualify `pinnedLeaves` dictionaries. =======

def removeInvalidPinnedLeavesInequalityViolation(state: EliminationState) -> EliminationState:
	listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	for pinnedLeaves in listPinnedLeaves:
		state.pinnedLeaves = pinnedLeaves
		if pinnedLeavesHasAViolation(state):
			continue
		state.listPinnedLeaves.append(pinnedLeaves)
	return state

def removeInvalidPinnedLeaves(state: EliminationState) -> EliminationState:
	listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	for pinnedLeaves in listPinnedLeaves:
		state.pinnedLeaves = pinnedLeaves
		if disqualifyDictionary(state):
			continue
		state.listPinnedLeaves.append(pinnedLeaves)
	return state

def disqualifyDictionary(state: EliminationState) -> bool:
	return any([
		_leading0notBeforeTrailing0dictionary(state.pinnedLeaves)
		, _leafTooEarlyInDomainDictionary(state)
		, _noPilesOpenFor一Dictionary(state)
		, _noPilesOpenForLeafDimensionOriginDictionary(state)
		, _leafDimensionOriginNotFirstDictionary(state)
		, _leafDimensionOriginsNotInOrderDictionary(state)
		# , Z0Z_excluder(state)
		, _notInPinPileRange(state)
	])

def _leafTooEarlyInDomainDictionary(state: EliminationState) -> bool:
	"""Some leaves can only be in the first pile of their domain if other leaves are before them."""
	for pileOf_r, r in itemfilter(_leafInFirstPileOfDomain, valfilter(lambda leafPinned: 2 < leafPinned.bit_count(), state.pinnedLeaves)).items():
		k = int(bit_flip(0, dimensionNearest首(r)).bit_flip(howMany0coordinatesAtTail(r)))
		if pileOf_k := reverseLookup(state.pinnedLeaves, k):
			if pileOf_k > pileOf_r:
				return True
		else:
			pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.pinnedLeaves.keys()).intersection(getLeafDomain(state, k))
			if not pilesOpenBefore_r:
				return True
	return False

def _leading0notBeforeTrailing0dictionary(pinnedLeaves: PinnedLeaves) -> bool:
	return any((pileOf_k > pileOf_r for (pileOf_k, _k), (pileOf_r, _r)
			in filter(_moreLeading0thanTrailing0, combinations(sorted(valfilter(notLeafOriginOrLeaf零, pinnedLeaves).items()), 2))))

def _leafDimensionOriginNotFirstDictionary(state: EliminationState) -> bool:
	for dimensionOrigin in state.productsOfDimensions[零: - (一)]:
		leavesInDimension: PinnedLeaves = valfilter(lambda leafPinned, dimensionOrigin=dimensionOrigin: (leafPinned > dimensionOrigin) and (leafPinned % dimensionOrigin == 0), state.pinnedLeaves)
		if not leavesInDimension:
			continue
		pile: int = min(leavesInDimension.keys())
		if pileOfDimensionOrigin := reverseLookup(state.pinnedLeaves, dimensionOrigin):
			if pile < pileOfDimensionOrigin:
				return True
		else:
			pilesOpenInDimension: set[int] = set(range(pile)).difference(state.pinnedLeaves.keys()).intersection(getLeafDomain(state, dimensionOrigin))
			if not pilesOpenInDimension:
				return True
	return False

def _leafDimensionOriginsNotInOrderDictionary(state: EliminationState) -> bool:
	return not is_sorted(tuple(dict(sorted(valfilter(lambda leafPinned: leafPinned in state.productsOfDimensions, state.pinnedLeaves).items())).values()))

def _noPilesOpenForLeafDimensionOriginDictionary(state: EliminationState) -> bool:
	for leaf in (filter(leafIsNotPinned(state.pinnedLeaves), state.productsOfDimensions[一: - (一)])):
		pileCeiling: int | None = min(valfilter(lambda leafPinned, leaf=leaf: leafPinned % leaf == 0
					, valfilter(notLeafOriginOrLeaf零, keyfilter(lambda pilePinned: pilePinned != state.pileLast, state.pinnedLeaves))).keys(), default=None)
		if pileCeiling:
			pileFloor: int = max(valfilter(lambda leafPinned, leaf=leaf: (leafPinned in state.productsOfDimensions) and leafPinned < leaf, state.pinnedLeaves).keys())
			if not set(range(pileFloor, pileCeiling)).difference(state.pinnedLeaves.keys()).intersection(getLeafDomain(state, leaf)):
				return True
	return False

def _noPilesOpenFor一Dictionary(state: EliminationState) -> bool:
	if (一 not in state.pinnedLeaves.values() and (一+零 not in state.pinnedLeaves.values())):
		pile: int | None = min(valfilter(lambda leafPinned: leafPinned % 一 == 0, valfilter(notLeafOriginOrLeaf零, keyfilter(lambda pilePinned: pilePinned != state.pileLast, state.pinnedLeaves))).keys(), default=None)
		if pile:
			pilesOpenInDimension: set[int] = set(range(pile)).difference(state.pinnedLeaves.keys())
			domainOf一零: list[int] = list(getLeafDomain(state, 一+零))
			if not pilesOpenInDimension.intersection(domainOf一零):
				return True
			else:
				noPilesOpenFor一 = True
				for pileOpen in pilesOpenInDimension:
					if (pileOpen + 1 in pilesOpenInDimension) and (pileOpen in domainOf一零):
						noPilesOpenFor一 = False
				return noPilesOpenFor一
	return False

def _notInPinPileRange(state: EliminationState) -> bool:
	"""The idea is sound, but the ROI is low."""
	state.pile = 一
	if not pileIsOpen(state.pinnedLeaves, state.pile) and not pileIsOpen(state.pinnedLeaves, state.pile - 1):
		pileRange = pinPile一Crease(state)
		if state.pinnedLeaves[state.pile] not in pileRange:
			return True
	state.pile = state.leavesTotal - 一
	if not pileIsOpen(state.pinnedLeaves, state.pile) and not pileIsOpen(state.pinnedLeaves, state.pile + 1):
		pileRange = pinPile首Less一Crease(state)
		if state.pinnedLeaves[state.pile] not in pileRange:
			return True

	state.pile = 一+零
	if (not pileIsOpen(state.pinnedLeaves, state.pile)
		and not pileIsOpen(state.pinnedLeaves, state.pile - 1)
		and not pileIsOpen(state.pinnedLeaves, state.leavesTotal - 一)):
		pileRange = pinPile一零Crease(state)
		if state.pinnedLeaves[state.pile] not in pileRange:
			return True

	state.pile = state.leavesTotal - (一+零)
	if not pileIsOpen(state.pinnedLeaves, state.pile) and not pileIsOpen(state.pinnedLeaves, state.pile + 1) and not pileIsOpen(state.pinnedLeaves, 一):
		pileRange = pinPile首Less一零Crease(state)
		if state.pinnedLeaves[state.pile] not in pileRange:
			return True
	state.pile = 二
	if (not pileIsOpen(state.pinnedLeaves, state.pile)
		and not pileIsOpen(state.pinnedLeaves, state.pile - 1)
		and not pileIsOpen(state.pinnedLeaves, state.leavesTotal - 一)
		and not pileIsOpen(state.pinnedLeaves, state.leavesTotal - (一+零))
		and not pileIsOpen(state.pinnedLeaves, 一)
		):
		pileRange = pinPile二Crease(state)
		if state.pinnedLeaves[state.pile] not in pileRange:
			return True
	return False

@syntacticCurry
def mappingHasKey(lookup: dict[int, Any], key: int) -> bool:
	return key in lookup

def Z0Z_excluder(state: EliminationState) -> bool:
	"""{atPileExcluded: {leafExcluded: {byPileExcluder: listLeafExcluders}}}."""
	if state.dimensionsTotal == 6:
		lookup: dict[int, dict[int, dict[int, list[int]]]] = dictionary2d6AtPileLeafExcludedByPile
	if state.dimensionsTotal == 5:
		lookup = dictionary2d5AtPileLeafExcludedByPile
	else:
		return False

	for pileExcluded, leafExcluded in keyfilter(mappingHasKey(lookup), valfilter(notLeafOriginOrLeaf零, state.pinnedLeaves)).items():
		if pileExcluded == state.pileLast:
			continue
		if leafExcluded not in lookup[pileExcluded]:
			continue

		for pileExcluder, listLeafExcluders in keyfilter(mappingHasKey(state.pinnedLeaves), lookup[pileExcluded][leafExcluded]).items():
			leafExcluder: int = state.pinnedLeaves[pileExcluder]
			if leafExcluder in listLeafExcluders:
				return True

	return False

# ======= crease-based subroutines for analyzing a specific `pile`. =======

def creasesToListLeavesAtPile(listCreases: list[int], listIndicesExcluded: list[int]) -> list[int]:
	return list(exclude(listCreases, listIndicesExcluded))

def pinPile一Crease(state: EliminationState) -> list[int]:
	listCreaseIndicesExcluded: list[int] = []
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	leafAt首Less一: int | None = state.pinnedLeaves.get(state.leavesTotal - 一)
	if leafAt首Less一 and (0 < howMany0coordinatesAtTail(leafAt首Less一)):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) - 1, state.dimensionsTotal - 2)])
	return creasesToListLeavesAtPile(getListLeavesIncrease(state, leafAtPileLess1), listCreaseIndicesExcluded)

def pinPile一零Crease(state: EliminationState) -> list[int]:
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	listLeavesIncrease: list[int] = getListLeavesIncrease(state, leafAtPileLess1)
	listCreaseIndicesExcluded: list[int] = []
	if 1 < len(listLeavesIncrease):
		listCreaseIndicesExcluded.append(0)
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	if is_even(leafAt首Less一) and (leafAtPileLess1 == 首零(state.dimensionsTotal)+零):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) + 1, state.dimensionsTotal)])
	return creasesToListLeavesAtPile(listLeavesIncrease, listCreaseIndicesExcluded)

def pinPile二Crease(state: EliminationState) -> list[int]:
	listCreaseIndicesExcluded: list[int] = []
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	if is_odd(leafAtPileLess1):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAtPileLess1), 5), ptount(leafAtPileLess1)])
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	if is_even(leafAtPileLess1) and is_even(leafAt首Less一):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafInSubHyperplane(leafAt首Less一)) - 一, (state.dimensionsTotal - 3))])
	if is_odd(leafAtPileLess1):
		listCreaseIndicesExcluded.append((int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5)
	leafAt首Less一零: int = state.pinnedLeaves[state.leavesTotal - (一+零)]
	if is_even(leafAtPileLess1) and leafAt首Less一零:
		listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafInSubHyperplane(leafAt首Less一零 - (leafAt首Less一零.bit_count() - is_even(leafAt首Less一零))).bit_count()) % (state.dimensionsTotal - 2) - is_even(leafAt首Less一零): None])
	leafAt一: int = state.pinnedLeaves[一]
	if (leafAt一 == 首零(state.dimensionsTotal)+零):
		listCreaseIndicesExcluded.extend([(int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5, howMany0coordinatesAtTail(leafAt首Less一零) - 1])
	if (leafAt一 == 首零(state.dimensionsTotal)+零) and (leafAt首Less一零 > 首零(state.dimensionsTotal)+零):
		listCreaseIndicesExcluded.extend([*range(int(leafAt首Less一零 - 2**(dimensionNearest首(leafAt首Less一零))).bit_length() - 1, state.dimensionsTotal - 2)])
	if ((leafAt一 == 首零(state.dimensionsTotal)+零) and (0 < leafAtPileLess1 - leafAt一 <= 2**(state.dimensionsTotal - 4)) and (0 < (leafAt首Less一 - leafAtPileLess1) <= 2**(state.dimensionsTotal - 3))):
		listCreaseIndicesExcluded.extend([ptount(leafAtPileLess1), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return creasesToListLeavesAtPile(getListLeavesIncrease(state, leafAtPileLess1), listCreaseIndicesExcluded)

def pinPile首Less一Crease(state: EliminationState) -> list[int]:
	listCreaseIndicesExcluded: list[int] = []
	leafAtPilePlus1: int = state.pinnedLeaves[state.pile + 1]
	leafAt一: int | None = state.pinnedLeaves.get(一)
	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listCreaseIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return creasesToListLeavesAtPile(getListLeavesDecrease(state, leafAtPilePlus1), listCreaseIndicesExcluded)

def pinPile首Less一零Crease(state: EliminationState) -> list[int]:
	listCreaseIndicesExcluded: list[int] = []
	leafAtPilePlus1: int = state.pinnedLeaves[state.pile + 1]
	if leafAtPilePlus1 < 首零一(state.dimensionsTotal):
		listCreaseIndicesExcluded.append(-1)
	leafAt一: int = state.pinnedLeaves[一]
	if (leafAtPilePlus1 == 首零(state.dimensionsTotal)+零) and (leafAt一 != 一+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 1)])
	return creasesToListLeavesAtPile(getListLeavesDecrease(state, leafAtPilePlus1), listCreaseIndicesExcluded)

# ======= Simple subroutines for a fixed `pile`. =======

def pinPileOriginFixed(state: EliminationState | None = None) -> list[int]:  # noqa: ARG001
	listLeavesAtPile: list[int] = [leafOrigin]
	return listLeavesAtPile

def pinPile零Fixed(state: EliminationState | None = None) -> list[int]:  # noqa: ARG001
	listLeavesAtPile: list[int] = [零]
	return listLeavesAtPile

def pinPile首Less零Fixed(state: EliminationState) -> list[int]:
	listLeavesAtPile: list[int] = [首零(state.dimensionsTotal)]
	return listLeavesAtPile

