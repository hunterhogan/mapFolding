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
	atPilePinLeaf, deconstructLeavesPinnedAtPile, leafIsNotPinned, leafIsPinned, notLeafOriginOrLeaf零, notPileLast,
	pileIsOpen)
from mapFolding.algorithms.iff import leavesPinnedHasAViolation
from mapFolding.dataBaskets import EliminationState
from math import log, log2, prod
from more_itertools import is_sorted
from operator import add, sub
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator

#  ====== Boolean filters ======================

@syntacticCurry
def _aDimensionOriginOfLeaf(leafDimensionOrigin: int, leaf: int) -> bool:
	return (leafDimensionOrigin == leafOrigin) or ((leafDimensionOrigin < leaf) and (leaf % leafDimensionOrigin == 0))

def _leafInFirstPileOfDomain(pileLeaf: tuple[int, int]) -> bool:
	return pileLeaf[0] == pileLeaf[1].bit_count() + (2**(howMany0coordinatesAtTail(pileLeaf[1]) + 1) - 2)

def _moreLeading0thanTrailing0(pileLeaf_pileLeaf: tuple[tuple[int, int], tuple[int, int]]) -> bool:
	return dimensionNearest首(pileLeaf_pileLeaf[0][1]) <= howMany0coordinatesAtTail(pileLeaf_pileLeaf[1][1])

# ======= "Beans and cornbread" functions =======

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, leavesPinned: PinnedLeaves) -> bool:
	return any((beans in leavesPinned.values()) ^ (cornbread in leavesPinned.values()) for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))))

def pinLeafCornbread(state: EliminationState) -> EliminationState:
	leafBeans = state.leavesPinned[state.pile]
	if leafBeans in [一+零, 首一(state.dimensionsTotal)]:
		leafCornbread = getListLeavesIncrease(state, leafBeans).pop()
		state.pile += 1
	else:
		leafCornbread = getListLeavesDecrease(state, leafBeans).pop()
		state.pile -= 1

	if disqualifyAppendingLeafAtPile(state, leafCornbread):
		state.leavesPinned = {}
	else:
		state.leavesPinned = atPilePinLeaf(state.leavesPinned, state.pile, leafCornbread)

	return state

# ======= append `leavesPinned` at `pile` if qualified =======

def appendLeavesPinnedAtPile(state: EliminationState, listLeavesAtPile: list[int]) -> EliminationState:
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	leavesToPin: list[int] = list(filterfalse(disqualify, listLeavesAtPile))

	dictionaryPinnedLeaves: dict[int, PinnedLeaves] = deconstructLeavesPinnedAtPile(state.leavesPinned, state.pile, leavesToPin)

	sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())

	beansOrCornbread: Callable[[PinnedLeaves], bool] = beansWithoutCornbread(sherpa)

	sherpa.listLeavesPinned.extend(tuple(valfilter(complement(beansOrCornbread), dictionaryPinnedLeaves).values()))

	for leavesPinned in valfilter(beansOrCornbread, dictionaryPinnedLeaves).values():
		stateCornbread: EliminationState = pinLeafCornbread(EliminationState(state.mapShape, pile=state.pile, leavesPinned=leavesPinned))
		if stateCornbread.leavesPinned:
			sherpa.listLeavesPinned.append(stateCornbread.leavesPinned)

	sherpa = removeInvalidLeavesPinned(sherpa)
	state.listLeavesPinned.extend(sherpa.listLeavesPinned)

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: int) -> bool:
		return any([
			_pileNotInRangeByLeaf(state, leaf)
			, leafIsPinned(state.leavesPinned, leaf)
			, mappingHasKey(state.leavesPinned, state.pile)
		])

def _pileNotInRangeByLeaf(state: EliminationState, leaf: int) -> bool:
	return state.pile not in list(getLeafDomain(state, leaf))

# ======= "by leaf" functions for disqualifying `leavesPinned` that are looking for a job =======

def _leading0notBeforeTrailing0ByLeaf(state: EliminationState, leaf: int) -> bool:
	leavesPinnedAbove: PinnedLeaves = valfilter(lambda leafPinned: leaf < leafPinned, valfilter(notLeafOriginOrLeaf零, state.leavesPinned))
	leavesPinnedBelow: PinnedLeaves = valfilter(lambda leafPinned: leafPinned < leaf, valfilter(notLeafOriginOrLeaf零, state.leavesPinned))
	compareThese: Iterator[tuple[tuple[int, int], tuple[int, int]]] = chain(zip(repeat((state.pile, leaf)), leavesPinnedAbove.items()), zip(leavesPinnedBelow.items(), repeat((state.pile, leaf))))
	return any((pileOf_r < pileOf_k for (pileOf_k, _k), (pileOf_r, _r) in filter(_moreLeading0thanTrailing0, compareThese)))

def _leafDimensionOriginNotFirstByLeaf(state: EliminationState, leaf: int) -> bool:
	"""Each product of dimensions is a 'dimension origin' leaf: a `leafDimensionOrigin` always comes before multiples of itself in the `folding`."""
	listDimensionOrigins:  list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(零, state.dimensionsTotal + inclusive)]

	dimensionOrigin: int = leaf
	pileOfDimensionOrigin = state.pile
	if dimensionOrigin in listDimensionOrigins:
		for pile, leafPinned in state.leavesPinned.items():
			if _aDimensionOriginOfLeaf(dimensionOrigin, leafPinned) and (pile < pileOfDimensionOrigin):
				return True

	pilesOpenBeforeLeaf: set[int] = set(range(state.pile)).difference(state.leavesPinned.keys())
	for dimensionOrigin in listDimensionOrigins:
		if (leaf != dimensionOrigin) and (leaf % dimensionOrigin == 0):
			if pileOfDimensionOrigin := reverseLookup(state.leavesPinned, dimensionOrigin):
				if state.pile < pileOfDimensionOrigin:
					return True
			elif not pilesOpenBeforeLeaf.intersection(getLeafDomain(state, dimensionOrigin)):
				return True

	return False

def _leafDimensionOriginsNotInOrderByLeaf(state: EliminationState, leaf: int) -> bool:
	r: int = leaf
	pileOf_r = state.pile
	if r in state.productsOfDimensions[一:None]:
		pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.leavesPinned.keys())
		dimensionIndexOf_r: int = state.productsOfDimensions.index(r)
		for k in state.productsOfDimensions[一:dimensionIndexOf_r]:
			if pileOf_k := reverseLookup(state.leavesPinned, k):
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
		pilesOpenAfter_k: set[int] = set(range(pileOf_k, state.leavesTotal)).difference(state.leavesPinned.keys())
		dimensionIndexOf_k: int = state.productsOfDimensions.index(k)
		for r in state.productsOfDimensions[dimensionIndexOf_k + 零: -(一)]:
			if pileOf_r := reverseLookup(state.leavesPinned, r):
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

	if (leaf in lookup) and (state.pile in lookup[leaf]):
		for pileExcluder, listLeafExcluders in keyfilter(mappingHasKey(state.leavesPinned), lookup[leaf][state.pile]).items():
			leafExcluder: int = state.leavesPinned[pileExcluder]
			if leafExcluder in listLeafExcluders:
				return True

	return False

# ======= Remove or disqualify `leavesPinned` dictionaries. =======

def removeInvalidLeavesPinnedInequalityViolation(state: EliminationState) -> EliminationState:
	listLeavesPinned: list[PinnedLeaves] = state.listLeavesPinned.copy()
	state.listLeavesPinned = []
	for leavesPinned in listLeavesPinned:
		state.leavesPinned = leavesPinned
		if leavesPinnedHasAViolation(state):
			continue
		state.listLeavesPinned.append(leavesPinned)
	return state

def removeInvalidLeavesPinned(state: EliminationState) -> EliminationState:
	listLeavesPinned: list[PinnedLeaves] = state.listLeavesPinned.copy()
	state.listLeavesPinned = []
	for leavesPinned in listLeavesPinned:
		state.leavesPinned = leavesPinned
		if disqualifyDictionary(state):
			continue
		state.listLeavesPinned.append(leavesPinned)
	return state
# ruff: noqa: ERA001
def disqualifyDictionary(state: EliminationState) -> bool:
	return any([
		# _leading0notBeforeTrailing0(state.leavesPinned)
		# , _leafTooEarlyInDomain(state)
		# , _noPilesOpenFor一(state)
		# , _noPilesOpenForLeafDimensionOrigin(state)
		# , _leafDimensionOriginNotFirst(state)
		# , _leafDimensionOriginsNotInOrder(state)
		Z0Z_notEnoughOpenPiles(state)
		, Z0Z_excluder(state)
		# , _notInPinPileRange(state)
	])

def _leafTooEarlyInDomain(state: EliminationState) -> bool:
	"""Some leaves can only be in the first pile of their domain if other leaves are before them."""
	for pileOf_r, r in itemfilter(_leafInFirstPileOfDomain, valfilter(lambda leafPinned: 2 < leafPinned.bit_count(), state.leavesPinned)).items():
		k = int(bit_flip(0, dimensionNearest首(r)).bit_flip(howMany0coordinatesAtTail(r)))
		if pileOf_k := reverseLookup(state.leavesPinned, k):
			if pileOf_k > pileOf_r:
				return True
		else:
			pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.leavesPinned.keys()).intersection(getLeafDomain(state, k))
			if not pilesOpenBefore_r:
				return True
	return False

def _leading0notBeforeTrailing0(leavesPinned: PinnedLeaves) -> bool:
	return any((pileOf_r < pileOf_k for (pileOf_k, _k), (pileOf_r, _r)
			in filter(_moreLeading0thanTrailing0, combinations(sorted(valfilter(notLeafOriginOrLeaf零, leavesPinned).items()), 2))))

def _leafDimensionOriginNotFirst(state: EliminationState) -> bool:
	for dimensionOrigin in state.productsOfDimensions[零: - (一)]:
		leafInDimension = _aDimensionOriginOfLeaf(dimensionOrigin)
		leavesInDimension: PinnedLeaves = valfilter(leafInDimension, state.leavesPinned)
		if not leavesInDimension:
			continue
		pile: int = min(leavesInDimension.keys())
		if pileOfDimensionOrigin := reverseLookup(state.leavesPinned, dimensionOrigin):
			if pile < pileOfDimensionOrigin:
				return True
		else:
			pilesOpenInDimension: set[int] = set(range(pile)).difference(state.leavesPinned.keys()).intersection(getLeafDomain(state, dimensionOrigin))
			if not pilesOpenInDimension:
				return True
	return False

def _leafDimensionOriginsNotInOrder(state: EliminationState) -> bool:
	return not is_sorted(tuple(dict(sorted(valfilter(lambda leafPinned: leafPinned in state.productsOfDimensions, state.leavesPinned).items())).values()))

def _noPilesOpenForLeafDimensionOrigin(state: EliminationState) -> bool:
	for leaf in (filter(leafIsNotPinned(state.leavesPinned), state.productsOfDimensions[一: - (一)])):
		pileCeiling: int | None = min(valfilter(lambda leafPinned, leaf=leaf: leafPinned % leaf == 0
					, valfilter(notLeafOriginOrLeaf零, keyfilter(lambda pilePinned: pilePinned != state.pileLast, state.leavesPinned))).keys(), default=None)
		if pileCeiling:
			pileFloor: int = max(valfilter(lambda leafPinned, leaf=leaf: (leafPinned in state.productsOfDimensions) and leafPinned < leaf, state.leavesPinned).keys())
			if not set(range(pileFloor, pileCeiling)).difference(state.leavesPinned.keys()).intersection(getLeafDomain(state, leaf)):
				return True
	return False

def _noPilesOpenFor一(state: EliminationState) -> bool:
	if (一 not in state.leavesPinned.values() and (一+零 not in state.leavesPinned.values())):
		pile: int | None = min(valfilter(lambda leafPinned: leafPinned % 一 == 0, valfilter(notLeafOriginOrLeaf零, keyfilter(lambda pilePinned: pilePinned != state.pileLast, state.leavesPinned))).keys(), default=None)
		if pile:
			pilesOpenInDimension: set[int] = set(range(pile)).difference(state.leavesPinned.keys())
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
	if not pileIsOpen(state.leavesPinned, state.pile) and not pileIsOpen(state.leavesPinned, state.pile - 1):
		pileRange = pinPile一Crease(state)
		if state.leavesPinned[state.pile] not in pileRange:
			return True
	state.pile = state.leavesTotal - 一
	if not pileIsOpen(state.leavesPinned, state.pile) and not pileIsOpen(state.leavesPinned, state.pile + 1):
		pileRange = pinPile首Less一Crease(state)
		if state.leavesPinned[state.pile] not in pileRange:
			return True

	state.pile = 一+零
	if (not pileIsOpen(state.leavesPinned, state.pile)
		and not pileIsOpen(state.leavesPinned, state.pile - 1)
		and not pileIsOpen(state.leavesPinned, state.leavesTotal - 一)):
		pileRange = pinPile一零Crease(state)
		if state.leavesPinned[state.pile] not in pileRange:
			return True

	state.pile = state.leavesTotal - (一+零)
	if not pileIsOpen(state.leavesPinned, state.pile) and not pileIsOpen(state.leavesPinned, state.pile + 1) and not pileIsOpen(state.leavesPinned, 一):
		pileRange = pinPile首Less一零Crease(state)
		if state.leavesPinned[state.pile] not in pileRange:
			return True
	state.pile = 二
	if (not pileIsOpen(state.leavesPinned, state.pile)
		and not pileIsOpen(state.leavesPinned, state.pile - 1)
		and not pileIsOpen(state.leavesPinned, state.leavesTotal - 一)
		and not pileIsOpen(state.leavesPinned, state.leavesTotal - (一+零))
		and not pileIsOpen(state.leavesPinned, 一)
		):
		pileRange = pinPile二Crease(state)
		if state.leavesPinned[state.pile] not in pileRange:
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

	for pileExcluded, leafExcluded in keyfilter(mappingHasKey(lookup), valfilter(notLeafOriginOrLeaf零, state.leavesPinned)).items():
		if pileExcluded == state.pileLast:
			continue
		if leafExcluded not in lookup[pileExcluded]:
			continue

		for pileExcluder, listLeafExcluders in keyfilter(mappingHasKey(state.leavesPinned), lookup[pileExcluded][leafExcluded]).items():
			leafExcluder: int = state.leavesPinned[pileExcluder]
			if leafExcluder in listLeafExcluders:
				return True

	return False

def Z0Z_notEnoughOpenPiles(state: EliminationState) -> bool:
	"""
	Prototype.

	Some leaves must be before or after other leaves, such as the dimension origin leaves.

	One function to replace
		_leading0notBeforeTrailing0(state.leavesPinned)
		, _leafTooEarlyInDomain(state)
		, _noPilesOpenFor一(state)
		, _noPilesOpenForLeafDimensionOrigin(state)
		, _leafDimensionOriginNotFirst(state)
		, _leafDimensionOriginsNotInOrder(state)

	For each pinned leaf, get all of the required leaves for before and after, and check if there are enough open piles for all of
	them.

	If the set of open piles does not intersect with the domain of a required leaf, return True.

	If a required leaf can only be pinned in one pile of the open piles, pin it at that pile in Z0Z_tester. Use the real pinning
	functions with the disposable Z0Z_tester.

	With the required leaves that are not pinned, somehow check if there are enough open piles for them.

	"""
	Z0Z_tester = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())

	while True:
		pilesOpen: list[int] = sorted(set(range(Z0Z_tester.leavesTotal)) - set(Z0Z_tester.leavesPinned.keys()))
		Z0Z_restart = False

		for pile, leaf in sorted(valfilter(notLeafOriginOrLeaf零, Z0Z_tester.leavesPinned).items()):
			# 1. Identify leaves that MUST be BEFORE `leaf`
			tailCoordinates = howMany0coordinatesAtTail(leaf)

			@syntacticCurry
			def mustBeBeforeLeaf(k: int, leaf: int, pile: int, tailCoordinates: int) -> bool:
				if k == leaf:
					return False
				if dimensionNearest首(k) <= tailCoordinates:
					return True
				if _leafInFirstPileOfDomain((pile, leaf)):
					k_special = int(bit_flip(0, dimensionNearest首(leaf)).bit_flip(howMany0coordinatesAtTail(leaf)))
					return k == k_special
				return False

			leavesRequiredBefore: set[int] = set(filter(mustBeBeforeLeaf(leaf=leaf, pile=pile, tailCoordinates=tailCoordinates), filter(notLeafOriginOrLeaf零, range(Z0Z_tester.leavesTotal))))

			unpinnedRequiredBefore: list[int] = list(filterfalse(lambda k: k in Z0Z_tester.leavesPinned.values(), leavesRequiredBefore))

			if any(reverseLookup(Z0Z_tester.leavesPinned, k) > pile for k in leavesRequiredBefore if k in Z0Z_tester.leavesPinned.values()):
				return True

			pilesOpenBefore: list[int] = list(filter(lambda aPile: aPile < pile, pilesOpen))
			if len(unpinnedRequiredBefore) > len(pilesOpenBefore):
				return True

			for k in unpinnedRequiredBefore:
				Z0Z_pilesValid: list[int] = list(filter(lambda aPile: aPile in list(getLeafDomain(Z0Z_tester, k)), pilesOpenBefore))
				if not Z0Z_pilesValid:
					return True
				if len(Z0Z_pilesValid) == 1:
					Z0Z_tester.leavesPinned = atPilePinLeaf(Z0Z_tester.leavesPinned, Z0Z_pilesValid[0], k)
					Z0Z_restart = True
					break
			if Z0Z_restart:
				break

			# 2. Identify leaves that MUST be AFTER `leaf`
			dimensionHead = dimensionNearest首(leaf)

			@syntacticCurry
			def mustBeAfterLeaf(r: int, leaf: int, dimensionHead: int) -> bool:
				if r == leaf:
					return False
				return howMany0coordinatesAtTail(r) >= dimensionHead

			requiredAfter: set[int] = set(filter(mustBeAfterLeaf(leaf=leaf, dimensionHead=dimensionHead), filter(notLeafOriginOrLeaf零, range(Z0Z_tester.leavesTotal))))

			unpinnedRequiredAfter: list[int] = list(filterfalse(lambda r: r in Z0Z_tester.leavesPinned.values(), requiredAfter))

			if any(reverseLookup(Z0Z_tester.leavesPinned, r) < pile for r in requiredAfter if r in Z0Z_tester.leavesPinned.values()):
				return True

			pilesOpenAfter: list[int] = list(filter(lambda aPile: aPile > pile, pilesOpen))
			if len(unpinnedRequiredAfter) > len(pilesOpenAfter):
				return True

			for r in unpinnedRequiredAfter:
				domain = getLeafDomain(Z0Z_tester, r)
				Z0Z_pilesValid: list[int] = list(filter(lambda aPile: aPile in domain, pilesOpenAfter))
				if not Z0Z_pilesValid:
					return True
				if len(Z0Z_pilesValid) == 1:
					Z0Z_tester.leavesPinned[Z0Z_pilesValid[0]] = r
					Z0Z_restart = True
					break
			if Z0Z_restart:
				break

		if not Z0Z_restart:
			break

	return False

# ======= crease-based subroutines for analyzing a specific `pile`. =======

def _getListLeavesCrease(state: EliminationState, leaf: int) -> list[int]:
	if 0 < leaf:
		listLeavesCrease: list[int] = getListLeavesDecrease(state, abs(leaf))
	else:
		listLeavesCrease: list[int] = getListLeavesIncrease(state, abs(leaf))
	return listLeavesCrease

# Second order
def pinPile一Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt首Less一: int | None = state.leavesPinned.get(state.leavesTotal - 一)

	if leafAt首Less一 and (0 < howMany0coordinatesAtTail(leafAt首Less一)):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) - 零, state.dimensionsTotal - 一)])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int | None = state.leavesPinned.get(一)

	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listCreaseIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

# Third order
def pinPile一零Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]

	if 1 < len(listLeavesCrease):
		listCreaseIndicesExcluded.append(0)
	if is_even(leafAt首Less一) and (leafAt一 == 首零(state.dimensionsTotal)+零):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) + 零, state.dimensionsTotal)])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一零Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]

	if leafAt首Less一 < 首零一(state.dimensionsTotal):
		listCreaseIndicesExcluded.append(-1)
	if (leafAt首Less一 == 首零(state.dimensionsTotal)+零) and (leafAt一 != 一+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 零)])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

# Fourth order
def pinPile二Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]
	leafAt一零: int = state.leavesPinned[一+零]
	leafAt首Less一零: int = state.leavesPinned[state.leavesTotal - (一+零)]

	if is_odd(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一零), 5), ptount(leafAt一零)])
		listCreaseIndicesExcluded.append((int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5)
	if is_even(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafInSubHyperplane(leafAt首Less一零 - (leafAt首Less一零.bit_count() - is_even(leafAt首Less一零))).bit_count()) % (state.dimensionsTotal - 2) - is_even(leafAt首Less一零): None])
		if is_even(leafAt首Less一):
			listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafInSubHyperplane(leafAt首Less一)) - 一, (state.dimensionsTotal - 3))])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listCreaseIndicesExcluded.extend([(int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5, howMany0coordinatesAtTail(leafAt首Less一零) - 1])
		if 首零(state.dimensionsTotal)+零 < leafAt首Less一零:
			listCreaseIndicesExcluded.extend([*range(int(leafAt首Less一零 - int(bit_flip(0, dimensionNearest首(leafAt首Less一零)))).bit_length() - 1, state.dimensionsTotal - 2)])
		if ((0 < leafAt一零 - leafAt一 <= bit_flip(0, state.dimensionsTotal - 4)) and (0 < (leafAt首Less一 - leafAt一零) <= bit_flip(0, state.dimensionsTotal - 3))):
			listCreaseIndicesExcluded.extend([ptount(leafAt一零), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首less二Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]
	leafAt一零: int = state.leavesPinned[一+零]
	leafAt首Less一零: int = state.leavesPinned[state.leavesTotal - (一+零)]
	leafAt二: int = state.leavesPinned[二]

	if is_odd(leafAt首Less一零):
		if leafAt首Less一零 == 一 + leafAt首Less一:  # noqa: SIM102
			if leafAt一 == 一+零:
				if leafAt二 == 二 + leafAt一零:
					listCreaseIndicesExcluded.append(int(log2(二)))

				if leafAt二 == 三 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(int(log2(三)))

				if leafAt二 == 四 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(3)
					listCreaseIndicesExcluded.append(int(log2(四)))

		if leafAt首Less一零 == 二 + leafAt首Less一:  # noqa: SIM102
			if leafAt一 == 一+零:
				listCreaseIndicesExcluded.append(int(log2(一)))

		if leafAt首Less一零 == 三 + leafAt首Less一:
			if leafAt一 == 一+零:
				listCreaseIndicesExcluded.append(int(log2(一)))
				if leafAt二 == 二 + leafAt一零:
					listCreaseIndicesExcluded.append(int(log2(二)))

			if leafAt一 == 二+零:
				listCreaseIndicesExcluded.append(int(log2(二)))

				if leafAt一零 == 一 + leafAt一:
					listCreaseIndicesExcluded.append(int(log2(一)))

			if leafAt一 == 三+零:
				if leafAt一零 == 一 + leafAt一:  # noqa: SIM102
					if leafAt二 == 二 + leafAt一零:
						listCreaseIndicesExcluded.append(int(log2(二)))
				if leafAt一零 == 二 + leafAt一:
					listCreaseIndicesExcluded.append(1)

		if leafAt首Less一零 == 四 + leafAt首Less一:
			if leafAt一 == 一+零:
				listCreaseIndicesExcluded.append(int(log2(一)))

				if leafAt二 == 二 + leafAt一零:
					listCreaseIndicesExcluded.append(int(log2(二)))

				if leafAt二 == 三 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(int(log2(三)))

			if leafAt一 == 二+零:
				listCreaseIndicesExcluded.append(1)
				listCreaseIndicesExcluded.append(int(log2(二)))
			if leafAt一 == 三+零:
				listCreaseIndicesExcluded.append(1)
				listCreaseIndicesExcluded.append(2)
				listCreaseIndicesExcluded.append(int(log2(三)))
			if leafAt一 == 四+零:
				if leafAt一零 == 一 + leafAt一:
					if leafAt二 == 二 + leafAt一零:
						listCreaseIndicesExcluded.append(int(log2(二)))
					if leafAt二 == 三 + leafAt一零:
						listCreaseIndicesExcluded.append(2)
						listCreaseIndicesExcluded.append(int(log2(三)))
				if leafAt一零 == 二 + leafAt一:
					listCreaseIndicesExcluded.append(1)
					if leafAt二 == 三 + leafAt一零:
						listCreaseIndicesExcluded.append(int(log2(三)))
				if leafAt一零 == 三 + leafAt一:
					listCreaseIndicesExcluded.append(1)
					listCreaseIndicesExcluded.append(2)

	if is_even(leafAt首Less一零):
		if leafAt首Less一零 == 首一(state.dimensionsTotal):
			if leafAt一零 == 二 + leafAt一:
				listCreaseIndicesExcluded.append(1)
			if leafAt一 == 一+零:
				listCreaseIndicesExcluded.append(1)
				if leafAt二 == 二 + leafAt一零:
					listCreaseIndicesExcluded.append(int(log2(二)))
				if leafAt二 == 三 + leafAt一零:
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(int(log2(三)))
			if leafAt一 == 二+零:
				listCreaseIndicesExcluded.append(1)
				listCreaseIndicesExcluded.append(int(log2(二)))
			if leafAt一 == 首二(state.dimensionsTotal)+零:
				listCreaseIndicesExcluded.append(1)
				listCreaseIndicesExcluded.append(2)
				listCreaseIndicesExcluded.append(3)
			if leafAt一 == 首零(state.dimensionsTotal)+零:
				if leafAt一零 == 一 + leafAt一:
					listCreaseIndicesExcluded.append(int(log2(一)))
					if leafAt二 == 三 + leafAt一零:
						listCreaseIndicesExcluded.append(2)
				if leafAt一零 == 二 + leafAt一:
					listCreaseIndicesExcluded.append(1)
					listCreaseIndicesExcluded.append(int(log2(二)))
					if leafAt二 == 三 + leafAt一零:
						listCreaseIndicesExcluded.append(int(log2(三)))
				if leafAt一零 == 三 + leafAt一:
					listCreaseIndicesExcluded.append(1)
					listCreaseIndicesExcluded.append(2)
					listCreaseIndicesExcluded.append(int(log2(三)))
				if leafAt一零 == 四 + leafAt一:
					if leafAt二 == 一 + leafAt一零:
						listCreaseIndicesExcluded.append(int(log2(一)))
					if leafAt二 == 二 + leafAt一零:
						listCreaseIndicesExcluded.append(1)
						listCreaseIndicesExcluded.append(int(log2(二)))
					if leafAt二 == 三 + leafAt一零:
						listCreaseIndicesExcluded.append(1)
						listCreaseIndicesExcluded.append(2)
						listCreaseIndicesExcluded.append(int(log2(三)))
		if leafAt首Less一零 == 首零二三(state.dimensionsTotal):
			listCreaseIndicesExcluded.append(0)
			listCreaseIndicesExcluded.append(1)
			if leafAt一零 == 二 + leafAt一:
				listCreaseIndicesExcluded.append(1)
		if leafAt首Less一零 == 首零一(state.dimensionsTotal)+一:
			listCreaseIndicesExcluded.append(0)
			if leafAt二 == 三 + leafAt一零:
				listCreaseIndicesExcluded.append(2)
		if leafAt首Less一零 == 首零一三(state.dimensionsTotal):
			listCreaseIndicesExcluded.append(0)
			listCreaseIndicesExcluded.append(1)
			if leafAt一零 == 二 + leafAt一:
				listCreaseIndicesExcluded.append(1)
			if leafAt二 == 三 + leafAt一零:
				listCreaseIndicesExcluded.append(int(log2(三)))
		if leafAt首Less一零 == 首零一二(state.dimensionsTotal):
			listCreaseIndicesExcluded.append(0)
			listCreaseIndicesExcluded.append(1)
			listCreaseIndicesExcluded.append(2)
			if leafAt一零 == 二 + leafAt一:
				listCreaseIndicesExcluded.append(1)

		if leafAt首Less一 == 首零(state.dimensionsTotal)+一:
			listCreaseIndicesExcluded.append(0)
		if leafAt首Less一 == 首零三(state.dimensionsTotal):
			listCreaseIndicesExcluded.append(0)
			listCreaseIndicesExcluded.append(1)

		if leafAt一零 == 一 + leafAt一:
			if leafAt二 == 二 + leafAt一零:
				listCreaseIndicesExcluded.append(int(log2(二)))
			if leafAt二 == 三 + leafAt一零:
				listCreaseIndicesExcluded.append(int(log2(三)))

	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

# ======= Domain-based subroutines for analyzing a specific `pile`. =======

def pinPile首零Less零AfterFourthOrder(state: EliminationState) -> list[int]:
	leaf: int = -1
	sumsProductsOfDimensions: list[int] = [sum(state.productsOfDimensions[0:dimension]) for dimension in range(state.dimensionsTotal + inclusive)]

	dictionaryPileToLeaves: dict[int, list[int]] = getDictionaryPileRanges(state)
	listRemoveLeaves: list[int] = []

	pileExcluder: int = 一
	leafAtPileExcluder: int = state.leavesPinned[pileExcluder]
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
	leafAtPileExcluder = state.leavesPinned[pileExcluder]
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
	leafAtPileExcluder = state.leavesPinned[pileExcluder]
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
	leafAtPileExcluder = state.leavesPinned[pileExcluder]
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
	leafAtPileExcluder = state.leavesPinned[pileExcluder]

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

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]
	leafAt一零: int = state.leavesPinned[一+零]
	leafAt首Less一零: int = state.leavesPinned[state.leavesTotal - (一+零)]

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


