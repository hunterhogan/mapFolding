from collections.abc import Iterable
from cytoolz.dicttoolz import keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from cytoolz.itertoolz import frequencies
from functools import cache
from gmpy2 import mpz, xmpz
from hunterMakesPy import raiseIfNone
from itertools import filterfalse
from mapFolding import inclusive
from mapFolding._e import (
	between, dimensionNearestTail, dimensionNearest首, DOTgetPileIfLeaf, DOTvalues, getAntiPileRangeOfLeaves, getLeafDomain,
	getLeavesCreaseBack, getLeavesCreaseNext, getZ0Z_precedence, leafIsPinned, mappingHasKey, mapShapeIs2上nDimensions,
	notLeafOriginOrLeaf零, notPileLast, oopsAllLeaves, oopsAllPileRangesOfLeaves, PermutationSpace, pileIsOpen,
	PileRangeOfLeaves, pileRangeOfLeavesAND, thisIsALeaf, Z0Z_JeanValjean, 一, 零, 首一, 首零一)
from mapFolding._e.algorithms.iff import removePermutationSpaceViolations
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinIt import atPilePinLeaf, deconstructPermutationSpaceAtPile
from more_itertools import filter_map, ilen as lenIterator, one, partition as more_itertools_partition, split_at
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

# ======= append `leavesPinned` at `pile` if qualified =======

def appendLeavesPinnedAtPile(state: EliminationState, leavesToPin: Iterable[int]) -> EliminationState:
	sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned)
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	beansOrCornbread: Callable[[PermutationSpace], bool] = beansWithoutCornbread(sherpa)

	dictionaryPermutationSpace: dict[int, PermutationSpace] = deconstructPermutationSpaceAtPile(state.leavesPinned, state.pile, filterfalse(disqualify, leavesToPin))

	sherpa.listPermutationSpace.extend(DOTvalues(valfilter(complement(beansOrCornbread), dictionaryPermutationSpace)))

	for leavesPinned in DOTvalues(valfilter(beansOrCornbread, dictionaryPermutationSpace)):
		stateCornbread: EliminationState = pinLeafCornbread(EliminationState(state.mapShape, pile=state.pile, leavesPinned=leavesPinned))
		if stateCornbread.leavesPinned:
			sherpa.listPermutationSpace.append(stateCornbread.leavesPinned)

	sherpa = updateListPermutationSpacePileRangesOfLeaves(sherpa)

	sherpa = removeInvalidPermutationSpace(sherpa)
	state.listPermutationSpace.extend(sherpa.listPermutationSpace)

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: int) -> bool:
	return any([_pileNotInRangeByLeaf(state, leaf), leafIsPinned(state.leavesPinned, leaf), not pileIsOpen(state.leavesPinned, state.pile)])

def _pileNotInRangeByLeaf(state: EliminationState, leaf: int) -> bool:
	return state.pile not in getLeafDomain(state, leaf)

# ======= Updating pile-ranges of leaves =======

def updateListPermutationSpacePileRangesOfLeaves(state: EliminationState) -> EliminationState:
	"""Flow control to apply per-`PermutationSpace` functions to all of `state.listPermutationSpace`."""
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_leafIsPinned(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_headsBeforeTails(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_conditionalPredecessors(state), listPermutationSpace))

	return state

@syntacticCurry
def _conditionalPredecessors(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
		return leavesPinned
	doItAgain: bool = True
	Z0Z_precedence: dict[int, dict[int, list[int]]] = getZ0Z_precedence(state)
	while doItAgain:
		doItAgain = False
		for pile, leaf in sorted(valfilter(mappingHasKey(Z0Z_precedence), keyfilter(notPileLast, valfilter(notLeafOriginOrLeaf零, oopsAllLeaves(leavesPinned)))).items()):
			if mappingHasKey(Z0Z_precedence[leaf], pile):
				antiPileRangeOfLeaves: mpz = getAntiPileRangeOfLeaves(state.leavesTotal, Z0Z_precedence[leaf][pile])
				for pileOf_k, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(keyfilter(between(pile + inclusive, state.pileLast), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_k] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_k]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_k, leavesPinned=leavesPinned)).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, leavesPinned)):
						return None
					break
	return leavesPinned

@syntacticCurry
def _headsBeforeTails(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:  # noqa: PLR0911
	doItAgain: bool = True
	while doItAgain:
		doItAgain = False
		for pile, leaf in sorted(keyfilter(notPileLast, valfilter(notLeafOriginOrLeaf零, oopsAllLeaves(leavesPinned))).items()):
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead:
				leaves_r = range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead])
				antiPileRangeOfLeaves: mpz = getAntiPileRangeOfLeaves(state.leavesTotal, leaves_r)
				for pileOf_r, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(keyfilter(between(2, pile - inclusive), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_r] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_r]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_r, leavesPinned=leavesPinned)).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, leavesPinned)):
						return None
					break
			if 0 < dimensionTail:
				leaves_k = range(0, state.sumsOfProductsOfDimensions[dimensionTail], 1)
				antiPileRangeOfLeaves: mpz = getAntiPileRangeOfLeaves(state.leavesTotal, leaves_k)
				for pileOf_k, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(keyfilter(between(pile + inclusive, state.pileLast), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_k] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_k]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_k, leavesPinned=leavesPinned)).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, leavesPinned)):
						return None
					break
	return leavesPinned

@syntacticCurry
def _leafIsPinned(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	"""Update or invalidate `leavesPinned`: for every `leaf` pinned at a `pile`, remove `leaf` from `PileRangeOfLeaves` from every other `pile`; or return `None` if the updated `leavesPinned` is invalid.

	If the `PileRangeOfLeaves` for a `pile` is reduced to one `leaf`, then convert from `pile: pileRangeOfLeaves` to `pile: leaf`.
	If that results in "beans without cornbread", then pin the complementary "cornbread" `leaf` at the appropriate adjacent
	`pile`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	leavesPinned : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: pileRangeOfLeaves`.

	Returns
	-------
	updatedLeavesPinned : PermutationSpace | None
		An updated `leavesPinned` if valid; otherwise `None`.

	"""
	doItAgain: bool = True

	while doItAgain:
		doItAgain = False
		antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, DOTvalues(oopsAllLeaves(leavesPinned)))
		for pile, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(leavesPinned).items():
			if (ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
				return None
			leavesPinned[pile] = ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript
			if thisIsALeaf(leavesPinned[pile]):
				if beansWithoutCornbread(state, leavesPinned):
					leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pile, leavesPinned=leavesPinned)).leavesPinned
					if not leavesPinned:
						return None
				doItAgain = True
				break
	return leavesPinned
# TODO There must be flow control more sophisticated than `for`-`break` for this situation.

@syntacticCurry
def suDONTku(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	"""My implementation breaks `eliminationCrease` and possibly other things.

	Sudoku trick:
	in a restricted space (square, row, or column), if two numbers have the same domain of two cells, then all other numbers are excluded from those two cells.
	^^^ generalizes to if n numbers have the same domain of n cells, all other numbers are excluded from that domain of n cells.
	"""
	doItAgain: bool = True
	while doItAgain:
		doItAgain = False
		ff: dict[mpz, int] = valfilter(between(2, 9001), frequencies(map(mpz, DOTvalues(oopsAllPileRangesOfLeaves(leavesPinned)))))
		for mpzPileRangesOfLeaves, howManyPiles in ff.items():
			pileWillAcceptThisManyDifferentLeaves: int = mpzPileRangesOfLeaves.bit_count() - 1
			if pileWillAcceptThisManyDifferentLeaves < howManyPiles:
				return None
			if pileWillAcceptThisManyDifferentLeaves == howManyPiles:
				for pile, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(leavesPinned).items():
					if mpzPileRangesOfLeaves == leafOrPileRangeOfLeaves:
						continue
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(mpzPileRangesOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pile] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pile]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pile, leavesPinned=leavesPinned)).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, leavesPinned)):
						return None
					break
	return leavesPinned

# ======= "Beans and cornbread" functions =======

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, leavesPinned: PermutationSpace) -> bool:
	return any((beans in DOTvalues(leavesPinned)) ^ (cornbread in DOTvalues(leavesPinned)) for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))))

def pinLeafCornbread(state: EliminationState) -> EliminationState:
	leafBeans: int = raiseIfNone(DOTgetPileIfLeaf(state.leavesPinned, state.pile))
	if leafBeans in [一+零, 首一(state.dimensionsTotal)]:
		leafCornbread: int = one(getLeavesCreaseNext(state, leafBeans))
		state.pile += 1
	else:
		leafCornbread = one(getLeavesCreaseBack(state, leafBeans))
		state.pile -= 1

	if disqualifyAppendingLeafAtPile(state, leafCornbread):
		state.leavesPinned = {}
	else:
		state.leavesPinned = atPilePinLeaf(state.leavesPinned, state.pile, leafCornbread)

	return state

# ======= Remove or disqualify `PermutationSpace` dictionaries. =======

def removeInvalidPermutationSpace(state: EliminationState) -> EliminationState:
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	for leavesPinned in listPermutationSpace:
		state.leavesPinned = leavesPinned
		if disqualifyDictionary(state):
			continue
		state.listPermutationSpace.append(leavesPinned)
	return removePermutationSpaceViolations(state)

def disqualifyDictionary(state: EliminationState) -> bool:
	return any([notEnoughOpenPiles(state)])

def notEnoughOpenPiles(state: EliminationState) -> bool:  # noqa: PLR0911
	"""Prototype.

	Some leaves must be before or after other leaves, such as the dimension origin leaves. For each pinned leaf, get all of the
	required leaves for before and after, and check if there are enough open piles for all of them. If the set of open piles does
	not intersect with the domain of a required leaf, return True. If a required leaf can only be pinned in one pile of the open
	piles, pin it at that pile in Z0Z_tester. Use the real pinning functions with the disposable Z0Z_tester. With the required
	leaves that are not pinned, check if there are enough open piles for them.
	"""
	Z0Z_tester = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned)
	Z0Z_precedence: dict[int, dict[int, list[int]]] = getZ0Z_precedence(state)
	if state.dimensionsTotal < 6:
		Z0Z_precedence = {}

	doItAgain: bool = True
	while doItAgain:
		doItAgain = False

		dictionaryPileLeaf: dict[int, int] = oopsAllLeaves(Z0Z_tester.leavesPinned)
		leavesFixed: tuple[int, ...] = tuple(DOTvalues(dictionaryPileLeaf))
		leavesNotPinned: frozenset[int] = frozenset(range(Z0Z_tester.leavesTotal)).difference(leavesFixed)
		pilesOpen: frozenset[int] = frozenset(range(Z0Z_tester.pileLast + inclusive)).difference(dictionaryPileLeaf.keys())

		for pile, leaf in sorted(keyfilter(notPileLast, valfilter(notLeafOriginOrLeaf零, dictionaryPileLeaf)).items()):
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)
			def notLeaf(comparand: int, leaf: int = leaf) -> bool:
				return comparand != leaf
			def IAmTheMFingLeaf(comparand: int, leaf: int = leaf) -> bool:
				return comparand == leaf
			@cache
			def mustBeAfterLeaf(r: int, dimensionHead: int = dimensionHead) -> bool:
				return dimensionNearestTail(r) >= dimensionHead
			@cache
			def mustBeBeforeLeaf(k: int, leaf: int = leaf, pile: int = pile, dimensionTail: int = dimensionTail) -> bool:
				if dimensionNearest首(k) <= dimensionTail:
					return True
				if mappingHasKey(Z0Z_precedence, leaf) and mappingHasKey(Z0Z_precedence[leaf], pile):
					return k in Z0Z_precedence[leaf][pile]
				return False

			leavesFixedBeforePile, leavesFixedAfterPile = split_at(leavesFixed, IAmTheMFingLeaf, maxsplit=1)
			if leavesRequiredBeforePile := set(filter(mustBeBeforeLeaf, filter(notLeaf, filter(notLeafOriginOrLeaf零, range(Z0Z_tester.leavesTotal))))
					).intersection(leavesFixedAfterPile):
				return True

			pilesOpenAfterLeaf, pilesOpenBeforeLeaf = more_itertools_partition(lambda aPileOpen, pile = pile: aPileOpen < pile, pilesOpen)
			if lenIterator(pilesOpenBeforeLeaf) < len(leavesRequiredBeforePileNotPinned := leavesNotPinned.intersection(leavesRequiredBeforePile)):
				return True

			for k in leavesRequiredBeforePileNotPinned:
				pilesOpenFor_k: set[int] = set(pilesOpenBeforeLeaf).intersection(xmpz(Z0Z_tester.leavesPinned[k]).iter_set())
				match len(pilesOpenFor_k):
					case 0:
						return True
					case 1:
						Z0Z_tester.leavesPinned = atPilePinLeaf(Z0Z_tester.leavesPinned, pilesOpenFor_k.pop(), k)
						doItAgain = True
						break
					case _:
						pass

			if doItAgain:
				break

			if leavesRequiredAfterPile := set(filter(mustBeAfterLeaf, filter(notLeaf, filter(notLeafOriginOrLeaf零, range(Z0Z_tester.leavesTotal))))
					).intersection(leavesFixedBeforePile):
				return True
			if lenIterator(pilesOpenAfterLeaf) < len(leavesRequiredAfterPileNotPinned := leavesNotPinned.intersection(leavesRequiredAfterPile)):
				return True

			for r in leavesRequiredAfterPileNotPinned:
				pilesOpenFor_r: set[int] = set(pilesOpenAfterLeaf).intersection(xmpz(Z0Z_tester.leavesPinned[r]).iter_set())
				match len(pilesOpenFor_r):
					case 0:
						return True
					case 1:
						Z0Z_tester.leavesPinned = atPilePinLeaf(Z0Z_tester.leavesPinned, pilesOpenFor_r.pop(), r)
						doItAgain = True
						break
					case _:
						pass
			if doItAgain:
				break

	return False

