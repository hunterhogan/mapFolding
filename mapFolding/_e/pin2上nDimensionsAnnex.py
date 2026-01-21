# ruff: noqa: ERA001
from collections import deque
from collections.abc import Iterable
from cytoolz.dicttoolz import keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_test as isBit1吗, xmpz
from hunterMakesPy import raiseIfNone
from itertools import combinations, filterfalse, product as CartesianProduct
from mapFolding import inclusive
from mapFolding._e import (
	between, dimensionNearestTail, dimensionNearest首, DOTgetPileIfLeaf, DOTitems, DOTvalues, getAntiPileRangeOfLeaves,
	getDictionaryConditionalLeafPredecessors, getLeafDomain, getLeavesCreaseBack, getLeavesCreaseNext, Leaf, leafIsPinned,
	mappingHasKey, mapShapeIs2上nDimensions, notLeafOriginOrLeaf零, notPileLast, oopsAllLeaves, oopsAllPileRangesOfLeaves,
	PermutationSpace, Pile, pileIsOpen, PileRangeOfLeaves, pileRangeOfLeavesAND, PinnedLeaves, thisHasThat, thisIsALeaf,
	Z0Z_JeanValjean, 一, 零, 首一, 首零一)
from mapFolding._e.algorithms.iff import removePermutationSpaceViolations
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinIt import atPilePinLeaf, deconstructPermutationSpaceAtPile
from more_itertools import filter_map, ilen as lenIterator, one, partition as more_itertools_partition, split_at
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

#======== append `permutationSpace` at `pile` if qualified =======

def appendPermutationSpaceAtPile(state: EliminationState, leavesToPin: Iterable[Leaf]) -> EliminationState:
	sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, permutationSpace=state.permutationSpace)
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	beansOrCornbread: Callable[[PermutationSpace], bool] = beansWithoutCornbread(sherpa)

	dictionaryPermutationSpace: dict[Leaf, PermutationSpace] = deconstructPermutationSpaceAtPile(state.permutationSpace, state.pile, filterfalse(disqualify, leavesToPin))

	sherpa.listPermutationSpace.extend(DOTvalues(valfilter(complement(beansOrCornbread), dictionaryPermutationSpace)))

	for permutationSpace in DOTvalues(valfilter(beansOrCornbread, dictionaryPermutationSpace)):
		stateCornbread: EliminationState = pinLeafCornbread(EliminationState(state.mapShape, pile=state.pile, permutationSpace=permutationSpace))
		if stateCornbread.permutationSpace:
			sherpa.listPermutationSpace.append(stateCornbread.permutationSpace)

	sherpa = updateListPermutationSpacePileRangesOfLeaves(sherpa)

	sherpa = removeInvalidPermutationSpace(sherpa)
	state.listPermutationSpace.extend(sherpa.listPermutationSpace)

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: Leaf) -> bool:
	return any([_pileNotInRangeByLeaf(state, leaf), leafIsPinned(state.permutationSpace, leaf), not pileIsOpen(state.permutationSpace, state.pile)])

def _pileNotInRangeByLeaf(state: EliminationState, leaf: Leaf) -> bool:
	return state.pile not in getLeafDomain(state, leaf)

#======== Updating `PileRangeOfLeaves` =======

def updateListPermutationSpacePileRangesOfLeaves(state: EliminationState) -> EliminationState:
	"""Flow control to apply per-`PermutationSpace` functions to all of `state.listPermutationSpace`."""
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_updatePileRangesOfLeavesLeafIsPinned(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_updatePileRangesOfLeavesHeadsBeforeTails(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_updatePileRangesOfLeavesConditionalPredecessors(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_updatePileRangesOfLeavesCrossedCreases(state), listPermutationSpace))

	return state

def Z0Z_subroutine(state: EliminationState, permutationSpace: PermutationSpace, pilesWithPileRangeOfLeavesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]], antiPileRangeOfLeaves: PileRangeOfLeaves) -> PermutationSpace:
	permutationSpaceHasNewLeaf: bool = False
	while pilesWithPileRangeOfLeavesToUpdate and not permutationSpaceHasNewLeaf:
		pileToUpdate, pileRangeOfLeaves = pilesWithPileRangeOfLeavesToUpdate.pop()
		if (ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, pileRangeOfLeaves))) is None:
			return {}
# TODO These statements are syntactically necessary because I'm using subscripts AND walrus operators. Does that suggest there is
# a "better" flow paradigm, or is this merely a limitation of Python syntax?
		permutationSpace[pileToUpdate] = ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript
		if thisIsALeaf(permutationSpace[pileToUpdate]):
			if beansWithoutCornbread(state, permutationSpace) and not (permutationSpace := pinLeafCornbread(EliminationState(state.mapShape, pile=pileToUpdate, permutationSpace=permutationSpace)).permutationSpace):
				return {}
			permutationSpaceHasNewLeaf = True
	if permutationSpaceHasNewLeaf:
		qq: PermutationSpace | None = _updatePileRangesOfLeavesLeafIsPinned(state, permutationSpace)
		if not qq:
			return {}
		else:
			permutationSpace = qq
	return permutationSpace

@syntacticCurry
def leafParityInDimension(leaf: Leaf, dimension: int) -> bool:
	return isBit1吗(leaf, dimension)

@syntacticCurry
def _updatePileRangesOfLeavesCrossedCreases(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""Development notes.

	Even parity cases.
	case leaves: k, r
		I don't think I have enough information.
	case leaves: k,	   k+1
		I guess I can compute the combined domain of r, r+1. I'm not sure if I can do much with it. But if I am willing to bifurcate the PermutationSpace, I could probably exclude impossible r,r+1 domains.

	case leaves: k, r, k+1
		I should be able to compute the domain of r+1.
	case leaves: k, r,		r+1
		I should be able to compute the domain of k+1.

	case leaves: k,	   k+1, r+1
		Equivalent to odd parity: k, r, k-1
	case leaves: 	r, k+1, r+1
		Equivalent to odd parity: k, r, 	 r-1

	case leaves: k, r, k+1, r+1
		Case will be checked by `permutationSpaceHasAViolation`. Should I punt the case directly to `permutationSpaceHasAViolation`, `pass`, or something else?

	If they are leaves, k+1 and r+1 will be in `oddInDimension`.
	NOTE I'm intentionally doing everything in small incremental steps.
	"""
	# NOTE Reminder this is for 2^n-dimensional maps

	for dimension in range(state.dimensionsTotal):
		permutationSpaceHasNewLeaf = True
		while permutationSpaceHasNewLeaf:
			permutationSpaceHasNewLeaf = False

			dictionaryLeafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in oopsAllLeaves(permutationSpace).items()}
			dictionaryPileLeaf: PinnedLeaves = oopsAllLeaves(permutationSpace)

			dictionaryPileLeafEvenInDimension: PinnedLeaves = valfilter(complement(leafParityInDimension(dimension=dimension)), dictionaryPileLeaf)
			dictionaryPileLeafOddInDimension: PinnedLeaves = valfilter(leafParityInDimension(dimension=dimension), dictionaryPileLeaf) # For efficiency, I wish I could create these dictionaries with one operation.

			dequeCombinationsToCheck: deque[tuple[PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]] = deque(CartesianProduct((dictionaryPileLeafOddInDimension,), combinations(dictionaryPileLeafEvenInDimension.items(), 2)))
			dequeCombinationsToCheck.extend(deque(CartesianProduct((dictionaryPileLeafEvenInDimension,), combinations(dictionaryPileLeafOddInDimension.items(), 2))))

			while dequeCombinationsToCheck and not permutationSpaceHasNewLeaf:
				dictionaryOtherParity, ((pileOf_k, k), (pileOf_r, r)) = dequeCombinationsToCheck.pop()
				kCrease = int(bit_flip(k, dimension))
				rCrease = int(bit_flip(r, dimension))

				pileOf_kCrease: bool | Pile = leafIsPinned(dictionaryOtherParity, kCrease)
				if pileOf_kCrease:
					pileOf_kCrease = dictionaryLeafToPile[kCrease]

				pileOf_rCrease: bool | Pile = leafIsPinned(dictionaryOtherParity, rCrease)
				if pileOf_rCrease:
					pileOf_rCrease = dictionaryLeafToPile[rCrease]

				match pileOf_kCrease, pileOf_rCrease:
					case int(), False: # case leaves: k, r, kCrease
						antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, (rCrease,))

						pilesLeafIsExcludedFrom: Iterable[Pile] = []
						if pileOf_k < pileOf_r < pileOf_kCrease: # pileOf_k < pileOf_r < pileOf_kCrease < pileOf_rCrease or pileOf_rCrease < pileOf_k < pileOf_r < pileOf_kCrease
							pilesLeafIsExcludedFrom: Iterable[int] = [*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)]
						elif pileOf_kCrease < pileOf_r < pileOf_k: # pileOf_rCrease < pileOf_kCrease < pileOf_r < pileOf_k or pileOf_kCrease < pileOf_r < pileOf_k < pileOf_rCrease
							pilesLeafIsExcludedFrom = [*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)]
						elif (pileOf_r < pileOf_kCrease < pileOf_k): # pileOf_r < pileOf_kCrease < pileOf_rCrease < pileOf_k
							pilesLeafIsExcludedFrom = range(pileOf_kCrease + 1, pileOf_k)
						elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r): # pileOf_k < pileOf_rCrease < pileOf_kCrease < pileOf_r or  pileOf_r < pileOf_k < pileOf_rCrease < pileOf_kCrease
							pilesLeafIsExcludedFrom = range(pileOf_k + 1, pileOf_kCrease)
						# elif (pileOf_kCrease < pileOf_k < pileOf_r): # pileOf_kCrease < pileOf_rCrease < pileOf_k < pileOf_r
						# 	pilesLeafIsExcludedFrom = range(pileOf_kCrease + 1, pileOf_k)

						pilesWithPileRangeOfLeavesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(DOTitems(keyfilter(thisHasThat(pilesLeafIsExcludedFrom), oopsAllPileRangesOfLeaves(permutationSpace))))

						Z0Z_sumPermutationSpace: int = sum(map(dimensionNearest首, permutationSpace.values()))
						if not (permutationSpace := Z0Z_subroutine(state, permutationSpace, pilesWithPileRangeOfLeavesToUpdate, antiPileRangeOfLeaves)):
							return None
						if sum(map(dimensionNearest首, permutationSpace.values())) < Z0Z_sumPermutationSpace:
							permutationSpaceHasNewLeaf = True

					case False, int(): # case leaves: k, r,		rCrease
						antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, (kCrease,))

						pilesLeafIsExcludedFrom: Iterable[Pile] = []
						if pileOf_rCrease < pileOf_k < pileOf_r: # pileOf_kCrease < pileOf_rCrease < pileOf_k < pileOf_r or pileOf_rCrease < pileOf_k < pileOf_r < pileOf_kCrease
							pilesLeafIsExcludedFrom = [*range(pileOf_rCrease), *range(pileOf_r + 1, state.pileLast + inclusive)]
						elif pileOf_r < pileOf_k < pileOf_rCrease: # pileOf_kCrease < pileOf_r < pileOf_k < pileOf_rCrease or pileOf_r < pileOf_k < pileOf_rCrease < pileOf_kCrease
							pilesLeafIsExcludedFrom = [*range(pileOf_r), *range(pileOf_rCrease + 1, state.pileLast + inclusive)]
						elif (pileOf_k < pileOf_r < pileOf_rCrease) or (pileOf_r < pileOf_rCrease < pileOf_k): # pileOf_k < pileOf_r < pileOf_kCrease < pileOf_rCrease or  pileOf_r < pileOf_kCrease < pileOf_rCrease < pileOf_k
							pilesLeafIsExcludedFrom = range(pileOf_r + 1, pileOf_rCrease)
						elif (pileOf_k < pileOf_rCrease < pileOf_r) or (pileOf_rCrease < pileOf_r < pileOf_k): # pileOf_k < pileOf_rCrease < pileOf_kCrease < pileOf_r or pileOf_rCrease < pileOf_kCrease < pileOf_r < pileOf_k
							pilesLeafIsExcludedFrom = range(pileOf_rCrease + 1, pileOf_r)

						pilesWithPileRangeOfLeavesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(DOTitems(keyfilter(thisHasThat(pilesLeafIsExcludedFrom), oopsAllPileRangesOfLeaves(permutationSpace))))

						Z0Z_sumPermutationSpace = sum(map(dimensionNearest首, permutationSpace.values()))
						if not (permutationSpace := Z0Z_subroutine(state, permutationSpace, pilesWithPileRangeOfLeavesToUpdate, antiPileRangeOfLeaves)):
							return None
						if sum(map(dimensionNearest首, permutationSpace.values())) < Z0Z_sumPermutationSpace:
							permutationSpaceHasNewLeaf = True

					case int(), int():
					# case leaves: k, r, k+1, r+1
					# 	Case will be checked by `permutationSpaceHasAViolation`. Should I punt the case directly to `permutationSpaceHasAViolation`, `pass`, or something else?
						pass
					case False, False: # pyright: ignore[reportUnnecessaryComparison]
					# case leaves: k, r
					# 	I don't think I have enough information.
						# TODO ??? Pattern will never be matched for subject type "Never"
						pass
					case _:
						pass

	return permutationSpace

@syntacticCurry
def _updatePileRangesOfLeavesConditionalPredecessors(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
		return permutationSpace

	dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		dequePileLeafToCheck: deque[tuple[Pile, Leaf]] = deque(sorted(DOTitems(valfilter(mappingHasKey(dictionaryConditionalLeafPredecessors), keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, oopsAllLeaves(permutationSpace)))))))

		while dequePileLeafToCheck and not permutationSpaceHasNewLeaf:
			pile, leaf = dequePileLeafToCheck.pop()

			if mappingHasKey(dictionaryConditionalLeafPredecessors[leaf], pile):
				# For this `pile:leaf` in `permutationSpace`, `dictionaryConditionalLeafPredecessors` has a `list` of at least one
				# `leaf` that must precede this `pile:leaf`, so the `list` cannot follow this `pile:leaf`, so remove the `list`
				# from the `PileRangeOfLeaves` at piles after `pile`.
				antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, dictionaryConditionalLeafPredecessors[leaf][pile])

				pilesWithPileRangeOfLeavesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(DOTitems(oopsAllPileRangesOfLeaves(keyfilter(between(pile + inclusive, state.pileLast), permutationSpace))))

				Z0Z_sumPermutationSpace: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := Z0Z_subroutine(state, permutationSpace, pilesWithPileRangeOfLeavesToUpdate, antiPileRangeOfLeaves)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < Z0Z_sumPermutationSpace:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _updatePileRangesOfLeavesHeadsBeforeTails(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		dequePileLeafToCheck: deque[tuple[Pile, Leaf]] = deque(sorted(keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, oopsAllLeaves(permutationSpace))).items()))
		while dequePileLeafToCheck and not permutationSpaceHasNewLeaf:
			pile, leaf = dequePileLeafToCheck.pop()
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead:
				leavesForbidden = range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead])
				Z0Z_fromHereToThere: tuple[Pile, Pile] = (2, pile - inclusive)

				antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, leavesForbidden)
				pilesWithPileRangeOfLeavesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(oopsAllPileRangesOfLeaves(keyfilter(between(*Z0Z_fromHereToThere), permutationSpace)).items())

				Z0Z_sumPermutationSpace: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := Z0Z_subroutine(state, permutationSpace, pilesWithPileRangeOfLeavesToUpdate, antiPileRangeOfLeaves)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < Z0Z_sumPermutationSpace:
					permutationSpaceHasNewLeaf = True

			if 0 < dimensionTail:
				leavesForbidden = range(0, state.sumsOfProductsOfDimensions[dimensionTail], 1)
				Z0Z_fromHereToThere = (pile + inclusive, state.pileLast)

				antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, leavesForbidden)
				pilesWithPileRangeOfLeavesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(oopsAllPileRangesOfLeaves(keyfilter(between(*Z0Z_fromHereToThere), permutationSpace)).items())

				Z0Z_sumPermutationSpace = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := Z0Z_subroutine(state, permutationSpace, pilesWithPileRangeOfLeavesToUpdate, antiPileRangeOfLeaves)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < Z0Z_sumPermutationSpace:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _updatePileRangesOfLeavesLeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""Update or invalidate `permutationSpace`: for every `leaf` pinned at a `pile`, remove `leaf` from `PileRangeOfLeaves` from every other `pile`; or return `None` if the updated `permutationSpace` is invalid.

	If the `PileRangeOfLeaves` for a `pile` is reduced to one `leaf`, then convert from `pile: pileRangeOfLeaves` to `pile: leaf`.
	If that results in "beans without cornbread", then pin the complementary "cornbread" `leaf` at the appropriate adjacent
	`pile`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: pileRangeOfLeaves`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		An updated `permutationSpace` if valid; otherwise `None`.

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, DOTvalues(oopsAllLeaves(permutationSpace)))

		pileRangesOfLeavesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(oopsAllPileRangesOfLeaves(permutationSpace).items())
		while pileRangesOfLeavesToUpdate and not permutationSpaceHasNewLeaf:
			pile, leafOrPileRangeOfLeaves = pileRangesOfLeavesToUpdate.pop()

			if (ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
				return None

			permutationSpace[pile] = ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript
			if thisIsALeaf(permutationSpace[pile]):
				if beansWithoutCornbread(state, permutationSpace) and not (permutationSpace := pinLeafCornbread(EliminationState(state.mapShape, pile=pile, permutationSpace=permutationSpace)).permutationSpace):
					return None
				permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def sudoku(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:  # noqa: ARG001
	"""My implementation broke `eliminationCrease` and possibly other things.

	Sudoku trick:
	in a restricted space (square, row, or column), if two numbers have the same domain of two cells, then all other numbers are excluded from those two cells.
	^^^ generalizes to if n numbers have the same domain of n cells, all other numbers are excluded from that domain of n cells.
	"""
	return permutationSpace

#======== "Beans and cornbread" functions =======

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, permutationSpace: PermutationSpace) -> bool:
	return any((beans in DOTvalues(permutationSpace)) ^ (cornbread in DOTvalues(permutationSpace)) for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))))

def pinLeafCornbread(state: EliminationState) -> EliminationState:
	leafBeans: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, state.pile))
	if leafBeans in [一+零, 首一(state.dimensionsTotal)]:
		leafCornbread: Leaf = one(getLeavesCreaseNext(state, leafBeans))
		state.pile += 1
	else:
		leafCornbread = one(getLeavesCreaseBack(state, leafBeans))
		state.pile -= 1

	if disqualifyAppendingLeafAtPile(state, leafCornbread):
		state.permutationSpace = {}
	else:
		state.permutationSpace = atPilePinLeaf(state.permutationSpace, state.pile, leafCornbread)

	return state

#======== Remove or disqualify `PermutationSpace` dictionaries. =======

def removeInvalidPermutationSpace(state: EliminationState) -> EliminationState:
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	for permutationSpace in listPermutationSpace:
		state.permutationSpace = permutationSpace
		if disqualifyDictionary(state):
			continue
		state.listPermutationSpace.append(permutationSpace)
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
	Z0Z_tester = EliminationState(state.mapShape, pile=state.pile, permutationSpace=state.permutationSpace)
	Z0Z_precedence: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)
	if state.dimensionsTotal < 6:
		Z0Z_precedence = {}

	doItAgain: bool = True
	while doItAgain:
		doItAgain = False

		dictionaryPileLeaf: PinnedLeaves = oopsAllLeaves(Z0Z_tester.permutationSpace)
		leavesFixed: tuple[Leaf, ...] = tuple(DOTvalues(dictionaryPileLeaf))
		leavesNotPinned: frozenset[Leaf] = frozenset(range(Z0Z_tester.leavesTotal)).difference(leavesFixed)
		pilesOpen: frozenset[Pile] = frozenset(range(Z0Z_tester.pileLast + inclusive)).difference(dictionaryPileLeaf.keys())

		for pile, leaf in sorted(keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, dictionaryPileLeaf)).items()):
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)
			def notLeaf(comparand: Leaf, leaf: Leaf = leaf) -> bool:
				return comparand != leaf
			def IAmTheMFingLeaf(comparand: Leaf, leaf: Leaf = leaf) -> bool:
				return comparand == leaf
			@cache
			def mustBeAfterLeaf(r: Leaf, dimensionHead: int = dimensionHead) -> bool:
				return dimensionNearestTail(r) >= dimensionHead
			@cache
			def mustBeBeforeLeaf(k: Leaf, leaf: Leaf = leaf, pile: Pile = pile, dimensionTail: int = dimensionTail) -> bool:
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
				pilesOpenFor_k: set[int] = set(pilesOpenBeforeLeaf).intersection(xmpz(Z0Z_tester.permutationSpace[k]).iter_set())
				match len(pilesOpenFor_k):
					case 0:
						return True
					case 1:
						Z0Z_tester.permutationSpace = atPilePinLeaf(Z0Z_tester.permutationSpace, pilesOpenFor_k.pop(), k)
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
				pilesOpenFor_r: set[int] = set(pilesOpenAfterLeaf).intersection(xmpz(Z0Z_tester.permutationSpace[r]).iter_set())
				match len(pilesOpenFor_r):
					case 0:
						return True
					case 1:
						Z0Z_tester.permutationSpace = atPilePinLeaf(Z0Z_tester.permutationSpace, pilesOpenFor_r.pop(), r)
						doItAgain = True
						break
					case _:
						pass
			if doItAgain:
				break

	return False

