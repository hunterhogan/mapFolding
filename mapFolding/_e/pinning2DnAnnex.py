from cytoolz.dicttoolz import itemfilter, keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from itertools import chain, combinations, filterfalse, repeat
from mapFolding import exclude, inclusive, mappingHasKey, reverseLookup
from mapFolding._e import (
	dimensionNearest首, dimensionSecondNearest首, getDictionaryPileRanges, getLeafDomain, getListLeavesDecrease,
	getListLeavesIncrease, howMany0coordinatesAtTail, leafInSubHyperplane, leafOrigin, pileOrigin, PinnedLeaves, ptount, 一,
	三, 二, 四, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一三, 首零一二, 首零三, 首零二, 首零二三)
from mapFolding._e._exclusions import (
	dictionary2d5AtPileLeafExcludedByPile, dictionary2d5LeafExcludedAtPileByPile, dictionary2d6AtPileLeafExcludedByPile,
	dictionary2d6LeafExcludedAtPileByPile)
from mapFolding._e.pinIt import (
	atPilePinLeaf, deconstructPinnedLeavesAtPile, leafIsNotPinned, notLeafOriginOrLeaf零, pileIsOpen)
from mapFolding.algorithms.iff import pinnedLeavesHasAViolation
from mapFolding.dataBaskets import EliminationState
from math import log, prod
from more_itertools import is_sorted
from operator import add, sub
from typing import TYPE_CHECKING

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
			, _Z0Z_excludeThisLeaf(state, leaf)
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
			if _aDimensionOriginOfLeaf(dimensionOrigin, leafPinned) and (pile < pileOfDimensionOrigin):
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
	lookup: dict[int, dict[int, dict[int, list[int]]]]
	if state.dimensionsTotal == 二+一:
		lookup = dictionary2d6LeafExcludedAtPileByPile
	elif state.dimensionsTotal == 二+零:
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
		, Z0Z_excluder(state)
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
		leafInDimension = _aDimensionOriginOfLeaf(dimensionOrigin)
		leavesInDimension: PinnedLeaves = valfilter(leafInDimension, state.pinnedLeaves)
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

def Z0Z_excluder(state: EliminationState) -> bool:
	"""{atPileExcluded: {leafExcluded: {byPileExcluder: listLeafExcluders}}}."""
	lookup: dict[int, dict[int, dict[int, list[int]]]]
	if state.dimensionsTotal == 二+一:
		lookup = dictionary2d6AtPileLeafExcludedByPile
	elif state.dimensionsTotal == 二+零:
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

# Second order
def pinPile一Crease(state: EliminationState) -> list[int]:
	direction = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.pinnedLeaves[direction(state.pile, 1)]

	listLeavesCrease: list[int] = getListLeavesIncrease(state, leafRoot)

	leafAt首Less一: int | None = state.pinnedLeaves.get(state.leavesTotal - 一)

	if leafAt首Less一 and (0 < howMany0coordinatesAtTail(leafAt首Less一)):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) - 1, state.dimensionsTotal - 2)])

	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一零Crease(state: EliminationState) -> list[int]:
	direction = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.pinnedLeaves[direction(state.pile, 1)]

	listLeavesCrease: list[int] = getListLeavesDecrease(state, leafRoot)

	leafAt一: int = state.pinnedLeaves[一]

	if leafRoot < 首零一(state.dimensionsTotal):
		listCreaseIndicesExcluded.append(-1)
	if (leafRoot == 首零(state.dimensionsTotal)+零) and (leafAt一 != 一+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 1)])

	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

# Third order
def pinPile一零Crease(state: EliminationState) -> list[int]:
	direction = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.pinnedLeaves[direction(state.pile, 1)]

	listLeavesCrease: list[int] = getListLeavesIncrease(state, leafRoot)

	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]

	if 1 < len(listLeavesCrease):
		listCreaseIndicesExcluded.append(0)
	if is_even(leafAt首Less一) and (leafRoot == 首零(state.dimensionsTotal)+零):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) + 1, state.dimensionsTotal)])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一Crease(state: EliminationState) -> list[int]:
	direction = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.pinnedLeaves[direction(state.pile, 1)]

	listLeavesCrease: list[int] = getListLeavesDecrease(state, leafRoot)

	leafAt一: int | None = state.pinnedLeaves.get(一)

	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listCreaseIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

# Fourth order
def pinPile二Crease(state: EliminationState) -> list[int]:
	direction = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.pinnedLeaves[direction(state.pile, 1)]

	listLeavesCrease: list[int] = getListLeavesIncrease(state, leafRoot)

	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	leafAt首Less一零: int = state.pinnedLeaves[state.leavesTotal - (一+零)]
	leafAt一: int = state.pinnedLeaves[一]

	if is_odd(leafRoot):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafRoot), 5), ptount(leafRoot)])
		listCreaseIndicesExcluded.append((int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5)
	if is_even(leafRoot):
		listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafInSubHyperplane(leafAt首Less一零 - (leafAt首Less一零.bit_count() - is_even(leafAt首Less一零))).bit_count()) % (state.dimensionsTotal - 2) - is_even(leafAt首Less一零): None])
		if is_even(leafAt首Less一):
			listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafInSubHyperplane(leafAt首Less一)) - 一, (state.dimensionsTotal - 3))])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listCreaseIndicesExcluded.extend([(int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5, howMany0coordinatesAtTail(leafAt首Less一零) - 1])
		if 首零(state.dimensionsTotal)+零 < leafAt首Less一零:
			listCreaseIndicesExcluded.extend([*range(int(leafAt首Less一零 - 2**(dimensionNearest首(leafAt首Less一零))).bit_length() - 1, state.dimensionsTotal - 2)])
		if ((0 < leafRoot - leafAt一 <= 2**(state.dimensionsTotal - 4)) and (0 < (leafAt首Less一 - leafRoot) <= 2**(state.dimensionsTotal - 3))):
			listCreaseIndicesExcluded.extend([ptount(leafRoot), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首less二Crease(state: EliminationState) -> list[int]:
	direction = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.pinnedLeaves[direction(state.pile, 1)]

	listLeavesCrease: list[int] = getListLeavesDecrease(state, leafRoot)

	leafAt一: int = state.pinnedLeaves[一]
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	leafAt一零: int = state.pinnedLeaves[一+零]
	leafAt首Less一零: int = state.pinnedLeaves[state.leavesTotal - (一+零)]
	leafAt二: int = state.pinnedLeaves[二]

	if is_odd(leafRoot):
		if leafRoot == 一 + leafAt首Less一:  # noqa: SIM102
			if leafAt一 == 一+零:
				if leafAt二 == 二 + leafAt一零:
					listCreaseIndicesExcluded.append(2)

				if leafAt二 == 三 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(3)

				if leafAt二 == 四 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(3)
					listCreaseIndicesExcluded.append(4)

		if leafRoot == 二 + leafAt首Less一:  # noqa: SIM102
			if leafAt一 == 一+零:
				listCreaseIndicesExcluded.append(1)

		if leafRoot == 三 + leafAt首Less一:
			if leafAt一 == 一+零:
				listCreaseIndicesExcluded.append(1)

				if leafAt二 == 二 + leafAt一零:
					listCreaseIndicesExcluded.append(2)

			if leafAt一 == 二+零:
				listCreaseIndicesExcluded.append(2)

				if leafAt一零 == 一 + leafAt一:
					listCreaseIndicesExcluded.append(1)

			if leafAt一 == 三+零:
				if leafAt一零 == 一 + leafAt一:  # noqa: SIM102
					if leafAt二 == 二 + leafAt一零:
						listCreaseIndicesExcluded.append(2)
				if leafAt一零 == 二 + leafAt一:
					listCreaseIndicesExcluded.append(1)

		if leafRoot == 四 + leafAt首Less一:
			if leafAt一 == 一+零:
				listCreaseIndicesExcluded.append(1)

				if leafAt二 == 二 + leafAt一零:
					listCreaseIndicesExcluded.append(2)

				if leafAt二 == 三 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(3)

			if leafAt一 == 二+零:
				listCreaseIndicesExcluded.append(1)
				listCreaseIndicesExcluded.append(2)
			if leafAt一 == 三+零:
				listCreaseIndicesExcluded.append(1)
				listCreaseIndicesExcluded.append(2)
				listCreaseIndicesExcluded.append(3)
			if leafAt一 == 四+零:
				if leafAt一零 == 一 + leafAt一:
					if leafAt二 == 二 + leafAt一零:
						listCreaseIndicesExcluded.append(2)
					if leafAt二 == 三 + leafAt一零:
						listCreaseIndicesExcluded.append(2)
						listCreaseIndicesExcluded.append(3)
				if leafAt一零 == 二 + leafAt一:
					listCreaseIndicesExcluded.append(1)
					if leafAt二 == 三 + leafAt一零:
						listCreaseIndicesExcluded.append(3)
				if leafAt一零 == 三 + leafAt一:
					listCreaseIndicesExcluded.append(1)
					listCreaseIndicesExcluded.append(2)

	if is_even(leafRoot):
		if leafRoot == 首一(state.dimensionsTotal):
			if leafAt一零 == 二 + leafAt一:
				listCreaseIndicesExcluded.append(1)
			if leafAt一 == 一+零:
				listCreaseIndicesExcluded.append(1)
				if leafAt二 == 二 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
				if leafAt二 == 三 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(3)
			if leafAt一 == 二+零:
				listCreaseIndicesExcluded.append(1)
				listCreaseIndicesExcluded.append(2)
			if leafAt一 == 首二(state.dimensionsTotal)+零:
				listCreaseIndicesExcluded.append(1)
				listCreaseIndicesExcluded.append(2)
				listCreaseIndicesExcluded.append(3)
			if leafAt一 == 首零(state.dimensionsTotal)+零:
				if leafAt一零 == 一 + leafAt一:
					listCreaseIndicesExcluded.append(1)
					if leafAt二 == 三 + leafAt一零:
						listCreaseIndicesExcluded.append(2)
				if leafAt一零 == 首零三(state.dimensionsTotal)+零:
					listCreaseIndicesExcluded.append(1)
					listCreaseIndicesExcluded.append(2)
					if leafAt二 == 首零二三(state.dimensionsTotal)+零:
						listCreaseIndicesExcluded.append(3)
				if leafAt一零 == 首零二(state.dimensionsTotal)+零:
					listCreaseIndicesExcluded.append(1)
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(3)
				if leafAt一零 == 首零一(state.dimensionsTotal)+零:
					if leafAt二 == 一 + leafAt一零:
						listCreaseIndicesExcluded.append(1)
					if leafAt二 == 首零一三(state.dimensionsTotal)+零:
						listCreaseIndicesExcluded.append(1)
						listCreaseIndicesExcluded.append(2)
					if leafAt二 == 首零一二(state.dimensionsTotal)+零:
						listCreaseIndicesExcluded.append(1)
						listCreaseIndicesExcluded.append(2)
						listCreaseIndicesExcluded.append(3)
		if leafRoot == 首零二三(state.dimensionsTotal):
			listCreaseIndicesExcluded.append(0)
			listCreaseIndicesExcluded.append(1)
			if leafAt一零 == 二 + leafAt一:
				listCreaseIndicesExcluded.append(1)
		if leafRoot == 首零一(state.dimensionsTotal)+一:
			listCreaseIndicesExcluded.append(0)
		if leafRoot == 首零一三(state.dimensionsTotal):
			listCreaseIndicesExcluded.append(0)
			listCreaseIndicesExcluded.append(1)
			if leafAt一零 == 二 + leafAt一:
				listCreaseIndicesExcluded.append(1)
			if leafAt二 == 首零二三(state.dimensionsTotal)+零 and leafAt首Less一零 == 首零一三(state.dimensionsTotal):
				listCreaseIndicesExcluded.append(3)
		if leafRoot == 首零一二(state.dimensionsTotal):
			listCreaseIndicesExcluded.append(0)
			listCreaseIndicesExcluded.append(1)
			listCreaseIndicesExcluded.append(2)
			if leafAt一零 == 二 + leafAt一:
				listCreaseIndicesExcluded.append(1)

		if leafAt首Less一 == 首零(state.dimensionsTotal)+一:
			listCreaseIndicesExcluded.append(0)
			if leafAt二 == 43 and leafAt首Less一零 == 首零一(state.dimensionsTotal)+一:
				listCreaseIndicesExcluded.append(2)
		if leafAt首Less一 == 首零三(state.dimensionsTotal):
			listCreaseIndicesExcluded.append(0)
			listCreaseIndicesExcluded.append(1)

		if leafAt一零 == 一 + leafAt一:
			if leafAt二 == 二 + leafAt一零:
				listCreaseIndicesExcluded.append(2)
			if leafAt二 == 三 + leafAt一零:
				listCreaseIndicesExcluded.append(3)

	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首零Less零AfterFourthOrder(state: EliminationState) -> list[int]:
	leaf: int = -1
	sumsProductsOfDimensions: list[int] = [sum(state.productsOfDimensions[0:dimension]) for dimension in range(state.dimensionsTotal + inclusive)]

	dictionaryPileToLeaves: dict[int, list[int]] = getDictionaryPileRanges(state)
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
	if (leafAt首Less一零 != getListLeavesDecrease(state, 首零(state.dimensionsTotal)+零)[0]) and (leafAt一 == 一+零):
		listRemoveLeaves.append(首一(state.dimensionsTotal))
	if (leafAt一 == 首二(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less一 + 零])
	if leafAt一.bit_length() < state.dimensionsTotal - 2:
		listRemoveLeaves.extend([一, leafAt首Less一 + 一])

	return sorted(set(dictionaryPileToLeaves[state.pile]).difference(set(listRemoveLeaves)))


