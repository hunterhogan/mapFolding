"""Reduce permutation spaces through iterative constraint propagation.

You can use this module to shrink the search space for map-folding computations by applying
multiple constraint-propagation strategies in a unified reduction loop. The module implements
a single large constraint-satisfaction algorithm expressed as a collection of specialized
reduction functions that reinforce each other. Each function encodes one constraint type
(crease adjacency, conditional predecessors, crossed creases, naked subsets, etc.), and the
module orchestrates iterative application of these constraints until no further reduction
occurs.

Architecture
------------
The module is organized as one conceptual algorithm split across multiple functions for
readability and maintainability:

1. `reduceAllPermutationSpaceInEliminationState` is the orchestrator that applies each
   reduction function in sequence until the permutation space stabilizes.

2. The `_reducePermutationSpace_*` functions are specialized constraint encoders that each
   implement one type of constraint. These functions are curried to accept `state` first,
   then `permutationSpace`, enabling use with `filter_map` [1].

3. `_reduceLeafSpace` is the shared subroutine that handles the mechanical work of updating
   `LeafOptions` at specified piles and propagating newly pinned leaves. All constraint
   encoders call `_reduceLeafSpace` to perform the actual updates.

The functions are not independent algorithms; the functions are interdependent components of
a constraint-propagation system. Each function assumes other functions will run afterward to
propagate the consequences of newly pinned leaves or reduced domains.

Functions
---------
Public
	reduceAllPermutationSpaceInEliminationState
		Reduce permutation space by iteratively applying constraint propagation.

Private (Constraint Encoders)
	_reducePermutationSpace_byCrease
		I use this to enforce crease adjacency constraints.
	_reducePermutationSpace_ConditionalPredecessors
		I use this to enforce conditional predecessor constraints.
	_reducePermutationSpace_CrossedCreases
		I use this to detect and eliminate crossed creases.
	_reducePermutationSpace_HeadsBeforeTails
		I use this to enforce head-before-tail ordering constraints.
	_reducePermutationSpace_LeafIsPinned
		I use this to propagate leaf pinning constraints.
	_reducePermutationSpace_nakedSubset
		I use this to detect and exploit naked subset constraints.
	_reducePermutationSpace_noConsecutiveDimensions
		I use this to enforce non-consecutive dimension constraints.
	_reducePermutationSpace_leafDomainOf1
		I use this to detect and pin leaves with domain size one.

Private (Shared Subroutine)
	_reduceLeafSpace
		I use this to update permutation space by removing forbidden leaves from piles.

Private (Utilities)
	ImaOddLeaf2上nDimensional
		I use this to check parity for 2^n-dimensional maps using bit operations.

References
----------
[1] more_itertools.filter_map
	https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.filter_map

"""
# ruff: noqa: ERA001
from collections import Counter, deque
from collections.abc import Iterable
from cytoolz.curried import map as toolz_map
from cytoolz.dicttoolz import itemfilter, keyfilter, valfilter
from cytoolz.functoolz import complement, compose, curry as syntacticCurry
from cytoolz.itertoolz import unique
from gmpy2 import bit_flip, bit_test as isBit1吗
from itertools import chain, combinations, product as CartesianProduct
from mapFolding import errorL33T, inclusive
from mapFolding._e import (
	bifurcatePermutationSpace, dimensionIndex, dimensionNearestTail, dimensionNearest首, DOTitems, DOTkeys, DOTvalues,
	getDictionaryConditionalLeafPredecessors, getIteratorOfLeaves, getLeavesCreaseAnte, getLeavesCreasePost,
	howManyLeavesInLeafOptions, JeanValjean, Leaf, LeafOptions, leafOptionsAND, leafOrigin, LeafSpace, makeLeafAntiOptions,
	mapShapeIs2上nDimensions, PermutationSpace, Pile, PinnedLeaves, UndeterminedPiles, 一, 零, 首一, 首零一)
from mapFolding._e.algorithms.iff import thisIsAViolation
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import (
	between吗, extractPinnedLeaves, extractUndeterminedPiles, leafIsInPileRange, leafIsPinned, mappingHasKey,
	notLeafOriginOrLeaf零, notPileLast, thisHasThat, thisIsALeaf, thisIsLeafOptions, thisNotHaveThat)
from mapFolding._e.pinIt import atPilePinLeaf, disqualifyPinningLeafAtPile
from math import prod
from more_itertools import filter_map, one, pairwise, triplewise
from operator import neg, pos
from typing import TYPE_CHECKING
from typing_extensions import TypeIs

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator

#======== Boolean filters ======================================

@syntacticCurry
def ImaOddLeaf2上nDimensional(leaf: Leaf, dimension: int) -> bool:
	"""Check parity of `leaf` in `dimension` for 2ⁿ-dimensional maps using bit operations.

	I use this specialized parity checker for 2ⁿ-dimensional maps where parity checking reduces
	to a bit test. This function is a performance-optimized special case of `ImaOddLeaf` [1] in
	`mapFolding._e.algorithms.iff` [2] that avoids mixed-radix coordinate extraction when
	`mapShape` is known to be `(2,) * n`.

	Mathematical Basis
	------------------
	For 2ⁿ-dimensional maps where `mapShape = (2, 2, ..., 2)`, leaf indices are binary-encoded
	coordinates. The coordinate in `dimension` is simply the bit at position `dimension` in the
	binary representation of `leaf`. The function uses `gmpy2.bit_test` [3] to extract the bit
	directly without mixed-radix arithmetic.

	Let `leaf` be a leaf index in a 2ⁿ-dimensional map. The coordinate of `leaf` in `dimension`
	is the bit at position `dimension` in the binary representation of `leaf`. The function
	returns `True` when the bit is 1 (odd) and `False` when the bit is 0 (even).

	Parameters
	----------
	leaf : Leaf
		A leaf index.
	dimension : int
		A dimension index (bit position).

	Returns
	-------
	isOdd : bool
		`True` when `leaf` has odd coordinate in `dimension`; `False` when `leaf` has even
		coordinate in `dimension`.

	References
	----------
	[1] mapFolding._e.algorithms.iff.ImaOddLeaf
		Internal package reference
	[2] mapFolding._e.algorithms.iff
		Internal package reference
	[3] gmpy2.bit_test
		https://gmpy2.readthedocs.io/en/latest/mpz.html#bit-test

	"""
	return isBit1吗(leaf, dimension)

#======== Reducing `LeafOptions` ===============================

def reduceAllPermutationSpaceInEliminationState(state: EliminationState) -> EliminationState:
	"""Reduce permutation space by iteratively applying constraint propagation.

	You can use this function to shrink the search space for map-folding computations by applying
	multiple constraint-propagation strategies in a loop until the permutation space stabilizes.
	The function orchestrates the unified constraint-satisfaction algorithm implemented across
	the specialized `_reducePermutationSpace_*` functions in this module. Each iteration applies
	each constraint type in sequence. The function continues iterating until the total permutation
	space size stops decreasing.

	The function is the orchestrator for the constraint-propagation system. The function treats
	the specialized reduction functions as interdependent components of a single large algorithm,
	not as independent transformations. Each function assumes other functions will run afterward
	to propagate newly discovered constraints.

	Algorithm Details
	-----------------
	The function applies these constraint types in sequence:

	1. Crease adjacency (via `_reducePermutationSpace_byCrease`)
	2. Pinned leaf propagation (via `_reducePermutationSpace_LeafIsPinned`)
	3. Head-before-tail ordering (via `_reducePermutationSpace_HeadsBeforeTails`)
	4. Conditional predecessors (via `_reducePermutationSpace_ConditionalPredecessors`)
	5. Crossed crease detection (via `_reducePermutationSpace_CrossedCreases`)
	6. Non-consecutive dimensions (via `_reducePermutationSpace_noConsecutiveDimensions`)
	7. Domain size one detection (via `_reducePermutationSpace_leafDomainOf1`)
	8. Naked subset elimination (via `_reducePermutationSpace_nakedSubset`)

	The function measures the total permutation space size before and after each full iteration.
	When the size stops decreasing, the function terminates and returns `state` with the reduced
	`state.listPermutationSpace`.

	The function uses `filter_map` [1] to apply each reduction function, automatically filtering
	out invalidated permutation spaces (those that return `None`).

	Parameters
	----------
	state : EliminationState
		A data basket containing `listPermutationSpace` to reduce and supporting computed
		properties.

	Returns
	-------
	updatedState : EliminationState
		The `state` with `state.listPermutationSpace` reduced by constraint propagation.

	Examples
	--------
	>>> from mapFolding._e.algorithms.eliminationCrease import doTheNeedful
	>>> sherpa = moveFoldingToListFolding(
	...     removeIFFViolationsFromEliminationState(
	...         reduceAllPermutationSpaceInEliminationState(sherpa)))

	References
	----------
	[1] more_itertools.filter_map
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.filter_map

	"""
	def prodOfDOTvalues(listLeafOptions: Iterable[LeafOptions]) -> int:
		return prod(map(howManyLeavesInLeafOptions, listLeafOptions))

	permutationsPermutationSpaceTotal: Callable[[list[PermutationSpace]], int] = compose(sum, toolz_map(compose(prodOfDOTvalues, DOTvalues, extractUndeterminedPiles)))
	permutationSpaceTotal: int = permutationsPermutationSpaceTotal(state.listPermutationSpace)
	continueReduction: bool = True

	while continueReduction:
		continueReduction = False

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_byCrease(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_LeafIsPinned(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_HeadsBeforeTails(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_ConditionalPredecessors(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_CrossedCreases(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_noConsecutiveDimensions(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_leafDomainOf1(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_nakedSubset(state), listPermutationSpace))

		# listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		# state.listPermutationSpace = []
		# state.listPermutationSpace.extend(filter_map(Z0Z_reducePermutationSpace_dimensionRestrictions(state), listPermutationSpace))

		permutationSpaceTotalReduced: int = permutationsPermutationSpaceTotal(state.listPermutationSpace)

		if permutationSpaceTotalReduced < permutationSpaceTotal:
			continueReduction = True
			permutationSpaceTotal = permutationSpaceTotalReduced

	return state

#-------- Shared logic -----------------------------------------

def _reduceLeafSpace(state: EliminationState, permutationSpace: PermutationSpace, pilesToUpdate: deque[tuple[Pile, LeafOptions]], leafAntiOptions: LeafOptions) -> PermutationSpace:
	"""I use this to update permutation space by removing forbidden leaves from piles.

	I use this shared subroutine to handle the mechanical work of updating `LeafOptions` at
	specified piles by removing forbidden leaves. All constraint encoders (`_reducePermutationSpace_*`)
	call this function to perform the actual updates. I process each pile in `pilesToUpdate`,
	remove leaves specified by `leafAntiOptions`, and propagate newly pinned leaves. I detect
	beans-without-cornbread configurations and pin the complementary cornbread leaf when
	appropriate.

	I do not return a `bool` for `permutationSpaceHasNewLeaf`. Calling functions compare
	`permutationSpace` properties before and after calling this function to detect whether new
	leaves were pinned.

	Algorithm Details
	-----------------
	For each pile in `pilesToUpdate`:

	1. Remove forbidden leaves by computing `leafOptionsAND(leafAntiOptions, leafOptions)`.
	2. Use `JeanValjean` [1] to convert the result to `LeafSpace` (either `Leaf` or
		`LeafOptions`).
	3. If the result is `None` (empty domain), invalidate `permutationSpace` by setting to `{}`.
	4. If the result is a `Leaf`, check for beans-without-cornbread configurations:
		- Beans-without-cornbread occurs when one member of a crease pair (beans/cornbread) is pinned but the adjacent crease neighbor (cornbread/beans) is not.
		- Pin the complementary cornbread leaf at the appropriate adjacent pile.
		- Set `permutationSpaceHasNewLeaf = True` to signal the calling function.

	When `permutationSpaceHasNewLeaf` becomes `True`, I call `_reducePermutationSpace_LeafIsPinned`
	to propagate the newly pinned leaf before returning.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.
	pilesToUpdate : deque[tuple[Pile, LeafOptions]]
		Piles to update with `pile` and existing `leafOptions`.
	leafAntiOptions : LeafOptions
		A bitset of leaves to remove from `LeafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace
		The updated `permutationSpace` if valid; otherwise an empty dictionary (invalid).

	Examples
	--------
	Calling functions detect `permutationSpaceHasNewLeaf` by comparing properties before and
	after:

	>>> sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
	>>> permutationSpace = _reduceLeafSpace(state, permutationSpace, pilesToUpdate, leafAntiOptions)
	>>> if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
	...     permutationSpaceHasNewLeaf = True

	References
	----------
	[1] mapFolding._e.JeanValjean
		Internal package reference

	"""
	permutationSpaceHasNewLeaf: bool = False
	while permutationSpace and pilesToUpdate and not permutationSpaceHasNewLeaf:
		pile, leafOptions = pilesToUpdate.pop()

		leafSpace: LeafSpace | None = JeanValjean(leafOptionsAND(leafAntiOptions, leafOptions))
		if leafSpace is not None:

			permutationSpace[pile] = leafSpace
			if thisIsALeaf(permutationSpace[pile]):
				leafBeans: Leaf | None = None
				for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))):
					beansPinned: bool = leafIsPinned(permutationSpace, beans)
					cornbreadPinned: bool = leafIsPinned(permutationSpace, cornbread)
					if beansPinned ^ cornbreadPinned:
						leafBeans = beans if beansPinned else cornbread
						break

				if leafBeans is not None:

					pileCornbread: Pile = pile
					if leafBeans in [一+零, 首一(state.dimensionsTotal)]:
						pileCornbread += 1
						leafCornbread: Leaf = one(getLeavesCreasePost(state, leafBeans))
					else:
						pileCornbread -= 1
						leafCornbread = one(getLeavesCreaseAnte(state, leafBeans))

					if disqualifyPinningLeafAtPile(EliminationState(state.mapShape, pile=pileCornbread, permutationSpace=permutationSpace), leafCornbread):
						permutationSpace = {}
					else:
						permutationSpace = atPilePinLeaf(permutationSpace, pileCornbread, leafCornbread)
				permutationSpaceHasNewLeaf = True
		else:
			permutationSpace = {}

	if permutationSpace and permutationSpaceHasNewLeaf:
		sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, permutationSpace)
		if not sherpa:
			permutationSpace = {}
		else:
			permutationSpace = sherpa
	return permutationSpace

#-------- Functions that use the shared logic -----------------------------------------

@syntacticCurry
def _reducePermutationSpace_byCrease(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to enforce crease adjacency constraints.

	I use this constraint encoder to enforce that when a leaf is pinned at a pile and the
	adjacent pile has undetermined `LeafOptions`, the adjacent pile can only contain leaves
	that are crease neighbors of the pinned leaf. I identify pinned-leaf-adjacent-to-undetermined
	configurations and restrict the undetermined pile to crease neighbors using `_reduceLeafSpace`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	leavesCrease: Iterator[Leaf] = iter(())
	pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque()

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		dequePileLeafSpacePileLeafSpace: deque[tuple[tuple[Pile, LeafSpace], tuple[Pile, LeafSpace]]] = deque(pairwise(sorted(permutationSpace.items())))
		while dequePileLeafSpacePileLeafSpace and not permutationSpaceHasNewLeaf:
			(pile_k, leafSpace_k), (pile_r, leafSpace_r) = dequePileLeafSpacePileLeafSpace.pop()

			if thisIsALeaf(leafSpace_k) and thisIsLeafOptions(leafSpace_r):
				pilesToUpdate = deque([(pile_r, leafSpace_r)])
				leavesCrease = getLeavesCreasePost(state, leafSpace_k)
			elif thisIsLeafOptions(leafSpace_k) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile_k, leafSpace_k)])
				leavesCrease = getLeavesCreaseAnte(state, leafSpace_r)
			else:
				continue

			sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reduceLeafSpace(state, permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease)))):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
				permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_ConditionalPredecessors(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to enforce conditional predecessor constraints.

	I use this constraint encoder to enforce that when a leaf is pinned at a pile and the leaf
	has conditional predecessors at that pile position, those predecessor leaves cannot appear
	after the pile. I consult `dictionaryConditionalLeafPredecessors` [1] to identify pinned
	leaves with predecessors and remove those predecessors from `LeafOptions` at subsequent
	piles using `_reduceLeafSpace`.

	The function only operates on 2ⁿ-dimensional maps with `n ≥ 6` because conditional predecessor
	data is only precomputed for those cases.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	References
	----------
	[1] mapFolding._e.getDictionaryConditionalLeafPredecessors
		Internal package reference

	"""
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
		return permutationSpace

	dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		dequePileLeaf: deque[tuple[Pile, Leaf]] = deque(sorted(DOTitems(valfilter(mappingHasKey(dictionaryConditionalLeafPredecessors),
			keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, extractPinnedLeaves(permutationSpace)))))))

		while dequePileLeaf and not permutationSpaceHasNewLeaf:
			pile, leaf = dequePileLeaf.pop()

			if mappingHasKey(dictionaryConditionalLeafPredecessors[leaf], pile):
				sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
						, pilesToUpdate = deque(DOTitems(extractUndeterminedPiles(keyfilter(between吗(pile + inclusive, state.pileLast), permutationSpace))))
						, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, dictionaryConditionalLeafPredecessors[leaf][pile])
					)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_CrossedCreases(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to detect and eliminate crossed creases.

	I use this constraint encoder to detect configurations where two creases would cross
	physically and either invalidate `permutationSpace` or restrict forbidden pile positions
	for unpinned crease leaves. For each dimension, I partition pinned leaves by parity (even/odd
	coordinate in that dimension), identify crease pairs where one leaf is pinned and the other
	is not, and compute forbidden pile positions where the unpinned leaf cannot appear without
	causing a crease crossing. I use `thisIsAViolation` [1] to detect invalid configurations
	and `_reduceLeafSpace` to remove forbidden leaves from forbidden piles.

	Mathematical Basis
	------------------
	Only creases whose constituent leaves have matching parity in a dimension can physically
	cross in that dimension. For two creases (k, k+1) and (r, r+1), if k and r have matching
	parity in `dimension`, the creases can cross. The function checks all pairs of pinned leaves
	with matching parity and determines forbidden pile positions for their unpinned crease
	partners based on the relative positions of the pinned leaves.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	References
	----------
	[1] mapFolding._e.algorithms.iff.thisIsAViolation
		Internal package reference

	"""
	pileOf_kCrease: Pile = errorL33T
	pileOf_rCrease: Pile = errorL33T
	pilesForbidden: Iterable[Pile] = []
	leafAntiOptions: LeafOptions = makeLeafAntiOptions(state.leavesTotal, frozenset())

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		for dimension in range(state.dimensionsTotal):

			dictionaryLeafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in extractPinnedLeaves(permutationSpace).items()}

			# For efficiency, I wish I could create the two dictionaries with one operation and without the intermediate `leavesPinned`.
			leavesPinned: PinnedLeaves = extractPinnedLeaves(permutationSpace)
			leavesPinnedEvenInDimension: PinnedLeaves = valfilter(complement(ImaOddLeaf2上nDimensional(dimension=dimension)), leavesPinned)
			leavesPinnedOddInDimension: PinnedLeaves = valfilter(ImaOddLeaf2上nDimensional(dimension=dimension), leavesPinned)

			dequePileLeafPileLeaf: deque[tuple[PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]] = deque(
												CartesianProduct((leavesPinnedOddInDimension,), combinations(leavesPinnedEvenInDimension.items(), 2)))
			dequePileLeafPileLeaf.extend(deque(CartesianProduct((leavesPinnedEvenInDimension,), combinations(leavesPinnedOddInDimension.items(), 2))))

			while dequePileLeafPileLeaf and not permutationSpaceHasNewLeaf:
				leavesPinnedParityOpposite, ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) = dequePileLeafPileLeaf.pop()
				leaf_kCrease: Leaf = int(bit_flip(leaf_k, dimension))
				leaf_rCrease: Leaf = int(bit_flip(leaf_r, dimension))

				if (leaf_kCreaseIsPinned := leafIsPinned(leavesPinnedParityOpposite, leaf_kCrease)):
					pileOf_kCrease = dictionaryLeafToPile[leaf_kCrease]
				if (leaf_rCreaseIsPinned := leafIsPinned(leavesPinnedParityOpposite, leaf_rCrease)):
					pileOf_rCrease = dictionaryLeafToPile[leaf_rCrease]

				if leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
					leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, (leaf_rCrease,))

					if pileOf_k < pileOf_r < pileOf_kCrease:
						pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
					elif pileOf_kCrease < pileOf_r < pileOf_k:
						pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
					elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
						pilesForbidden = frozenset(range(pileOf_kCrease + 1, pileOf_k))
					elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
						pilesForbidden = frozenset(range(pileOf_k + 1, pileOf_kCrease))

				elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
					leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, (leaf_kCrease,))

					if pileOf_rCrease < pileOf_k < pileOf_r:
						pilesForbidden = frozenset([*range(pileOf_rCrease), *range(pileOf_r + 1, state.pileLast + inclusive)])
					elif pileOf_r < pileOf_k < pileOf_rCrease:
						pilesForbidden = frozenset([*range(pileOf_r), *range(pileOf_rCrease + 1, state.pileLast + inclusive)])
					elif (pileOf_k < pileOf_r < pileOf_rCrease) or (pileOf_r < pileOf_rCrease < pileOf_k):
						pilesForbidden = frozenset(range(pileOf_r + 1, pileOf_rCrease))
					elif (pileOf_k < pileOf_rCrease < pileOf_r) or (pileOf_rCrease < pileOf_r < pileOf_k):
						pilesForbidden = frozenset(range(pileOf_rCrease + 1, pileOf_r))

				elif leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
					if thisIsAViolation(pileOf_k, pileOf_r, pileOf_kCrease, pileOf_rCrease):
						return None
					continue

				else: # elif not leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
					continue

				sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
					, pilesToUpdate = deque(DOTitems(keyfilter(thisHasThat(pilesForbidden), extractUndeterminedPiles(permutationSpace))))
					, leafAntiOptions=leafAntiOptions)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_HeadsBeforeTails(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to enforce head-before-tail ordering constraints.

	I use this constraint encoder to enforce that leaves with large coordinates in a dimension
	(tail) can only appear after all leaves with small coordinates in that dimension (head) have
	appeared. When a leaf with nonzero nearest head dimension is pinned, I remove all leaves with
	larger coordinates in that dimension from preceding piles. When a leaf with nonzero nearest
	tail dimension is pinned, I remove all leaves with smaller coordinates in that dimension from
	subsequent piles.

	Algorithm Details
	-----------------
	For each pinned leaf:

	1. Compute `dimensionNearest首(leaf)` [1] to identify the dimension with the smallest coordinate magnitude from the head.
	2. If nonzero, remove all leaves with larger coordinates in that dimension from piles before `pile`.
	3. Compute `dimensionNearestTail(leaf)` [2] to identify the dimension with the smallest coordinate magnitude from the tail.
	4. If nonzero, remove all leaves with smaller coordinates in that dimension from piles after `pile`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	References
	----------
	[1] mapFolding._e.dimensionNearest首
		Internal package reference
	[2] mapFolding._e.dimensionNearestTail
		Internal package reference

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		dequePileLeaf: deque[tuple[Pile, Leaf]] = deque(sorted(DOTitems(keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, extractPinnedLeaves(permutationSpace))))))

		while dequePileLeaf and not permutationSpaceHasNewLeaf:
			pile, leaf = dequePileLeaf.pop()

			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead:
				sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
						, pilesToUpdate = deque(extractUndeterminedPiles(keyfilter(between吗(2, pile - inclusive), permutationSpace)).items())
						, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead]))
					)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
					permutationSpaceHasNewLeaf = True

			dimensionTail: int = dimensionNearestTail(leaf)
			if 0 < dimensionTail:
				sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
						, pilesToUpdate = deque(extractUndeterminedPiles(keyfilter(between吗(pile + inclusive, state.pileLast), permutationSpace)).items())
						, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, range(leafOrigin, state.sumsOfProductsOfDimensions[dimensionTail]))
					)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_LeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to propagate leaf pinning constraints.

	I use this constraint encoder to enforce that every pinned leaf can appear at only one pile.
	For every leaf pinned at a pile, I remove that leaf from `LeafOptions` at all other piles.
	When `LeafOptions` at a pile reduces to a single leaf, I convert `pile: leafOptions` to
	`pile: leaf` (pinning the leaf). When that creates a beans-without-cornbread configuration,
	I pin the complementary cornbread leaf at the appropriate adjacent pile.

	This function is the primary propagator for newly pinned leaves. All other constraint encoders
	call `_reduceLeafSpace`, which calls this function when new leaves are pinned. This function
	iteratively applies pinning until no new leaves are discovered.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)
		sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
		if not (permutationSpace := _reduceLeafSpace(state, permutationSpace, deque(pilesUndetermined.items()), makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned)))):
			return None
		if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_nakedSubset(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to detect and exploit naked subset constraints.

	I use this constraint encoder to detect naked subsets in the permutation space and remove
	subset leaves from all other piles. A naked subset occurs when `n` piles share the same
	`LeafOptions` containing exactly `n` leaves. Those `n` leaves can only appear in those `n`
	piles, so I remove those leaves from `LeafOptions` at all other piles using `_reduceLeafSpace`.

	Algorithm Details
	-----------------
	The function implements a specialized naked subset detector optimized for high throughput:

	1. Extract `UndeterminedPiles` (piles with `LeafOptions`).
	2. Group piles by their `LeafOptions` values.
	3. Filter groups where the number of leaves in `LeafOptions` equals the number of piles sharing that `LeafOptions` (the naked subset criterion).
	4. For each naked subset, remove subset leaves from all other piles.

	The function iterates until no new leaves are pinned. The function is not a comprehensive
	naked subset solver; the function prioritizes high throughput for a strong return on
	investment.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	permutationSpaceHasNewLeaf: bool = True
	leafOptionsKey: int = 0
	piles: int = 1
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		pilesUndetermined: UndeterminedPiles = extractUndeterminedPiles(permutationSpace)

		groupByLeafOptions: dict[LeafOptions, set[Pile]] = {}
		for pile, leafOptions in valfilter(thisNotHaveThat(unique(pilesUndetermined.values())), pilesUndetermined).items():
			groupByLeafOptions.setdefault(leafOptions, set()).add(pile)

		dequeLeafOptionsAndPiles: deque[tuple[LeafOptions, set[Pile]]] = deque(DOTitems(
			itemfilter(lambda groupBy: (howManyLeavesInLeafOptions(groupBy[leafOptionsKey])) == len(groupBy[piles]), groupByLeafOptions)))

		while dequeLeafOptionsAndPiles and not permutationSpaceHasNewLeaf:
			leafOptions, setPiles = dequeLeafOptionsAndPiles.pop()

			sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
					, pilesToUpdate = deque(DOTitems(keyfilter(thisNotHaveThat(setPiles), pilesUndetermined)))
					, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))
				)):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
				permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_noConsecutiveDimensions(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to enforce non-consecutive dimension constraints.

	I use this constraint encoder to detect arithmetic progressions in pinned leaves and forbid
	the next term in the progression from appearing at the adjacent pile. When three consecutive
	piles contain leaves forming an arithmetic progression (leaf_k, leaf, leaf_r where
	`leaf - leaf_k == leaf_r - leaf`), the next term in the progression cannot appear at the
	next pile because map foldings cannot have four consecutive leaves in arithmetic progression.

	The function examines all triples of consecutive piles and identifies configurations where:
	1. Two adjacent piles have pinned leaves and the third has `LeafOptions`, or
	2. The middle pile has `LeafOptions` and the outer two have pinned leaves.

	For each pattern, I compute the forbidden leaf (the next term in the arithmetic progression)
	and remove that leaf from the undetermined pile using `_reduceLeafSpace`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	leafForbidden: Leaf = -errorL33T
	pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque()

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		dequeTriplePileLeafSpace: deque[tuple[tuple[Pile, LeafSpace], tuple[Pile, LeafSpace], tuple[Pile, LeafSpace]]] = deque(triplewise(sorted(DOTitems(permutationSpace))))

		while dequeTriplePileLeafSpace and not permutationSpaceHasNewLeaf:
			(pile_k, leafSpace_k), (pile, leafSpace), (pile_r, leafSpace_r) = dequeTriplePileLeafSpace.pop()

			if thisIsALeaf(leafSpace_k) and thisIsALeaf(leafSpace) and thisIsLeafOptions(leafSpace_r):
				pilesToUpdate = deque([(pile_r, leafSpace_r)])
				leafForbidden = leafSpace + (leafSpace - leafSpace_k)
			elif thisIsALeaf(leafSpace_k) and thisIsLeafOptions(leafSpace) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile, leafSpace)])
				leafForbidden = (leafSpace_k + leafSpace_r) // 2
			elif thisIsLeafOptions(leafSpace_k) and thisIsALeaf(leafSpace) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile_k, leafSpace_k)])
				leafForbidden = leafSpace - (leafSpace_r - leafSpace)
			else:
				continue

			if 0 <= leafForbidden < state.leavesTotal:
				sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, [leafForbidden]))):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

"""# TODO implement
1. The signs of the magnitudes alternate: if the difference between two leaves is +2, for example, then before there can be another difference of +2, there must be a difference of -2.
2. The total number of differences equal to `pos(state.leavesTotal // 2)` is always exactly one more than the total number of differences equal to `neg(state.leavesTotal // 2)`.
	1. Therefore, the first and last differences with magnitude `state.leavesTotal // 2` are positive.
3. For all other magnitudes in `state.productsOfDimensions[0:-2]`, the total number of positive and negative differences is always equal.
	1. Therefore, the first and last differences with those magnitudes must have opposite signs.
"""
@syntacticCurry
def Z0Z_reducePermutationSpace_dimensionRestrictions(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	forbiddenLeaves: set[Leaf] = set()
	thePile: int = 0
	theLeaf: int = 1
	Z0Z_limitSignFromOrigin: list[list[int]] = [[1]] + [[]] * len(state.productsOfDimensions[1:-2]) + [[-1]]
	Z0Z_limitSignFromEnd: list[list[int]] = [[1]] + [[]] * len(state.productsOfDimensions[1:-2]) + [[-1]]
	permutationSpaceHasNewLeaf: bool = True

	def pileLeaf吗(pileLeafSpace: tuple[Pile, LeafSpace]) -> TypeIs[tuple[Pile, Leaf]]:
		return thisIsALeaf(pileLeafSpace[1])

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned: PinnedLeaves = extractPinnedLeaves(permutationSpace)
		Z0Z_deque: deque[tuple[Pile, LeafSpace]] = deque(sorted(DOTitems(keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, permutationSpace)))))

		while (pileLeaf吗(Z0Z_deque[0])) or (pileLeaf吗(Z0Z_deque[-1])):
			thereIsNoLeftSpoon: tuple[Pile, LeafSpace] = Z0Z_deque.popleft()
			if pileLeaf吗(thereIsNoLeftSpoon):
				difference: int = thereIsNoLeftSpoon[theLeaf] - leavesPinned[thereIsNoLeftSpoon[thePile] - 1]
				if 0 < difference:
					sign: int = pos(1)
				else:
					sign = neg(1)
				indexDimension: int = dimensionIndex(abs(difference))
				Z0Z_limitSignFromOrigin[indexDimension].append(sign)
				if not Z0Z_limitSignFromEnd[indexDimension]:
					Z0Z_limitSignFromEnd[indexDimension].append(sign)
			else:
				Z0Z_deque.appendleft(thereIsNoLeftSpoon)
				thereIsNoRightSpoon: tuple[Pile, LeafSpace] = Z0Z_deque.pop()
				if pileLeaf吗(thereIsNoRightSpoon):
					difference: int = leavesPinned[thereIsNoRightSpoon[thePile] + 1] - thereIsNoRightSpoon[theLeaf]
					if 0 < difference:
						sign = 1
					else:
						sign = -1
					indexDimension: int = dimensionIndex(abs(difference))
					Z0Z_limitSignFromEnd[indexDimension].append(sign)
					if not Z0Z_limitSignFromOrigin[indexDimension]:
						Z0Z_limitSignFromOrigin[indexDimension].append(sign)
				else:
					message: str = 'I think this is a logic error.'
					raise ValueError(message)

		while Z0Z_deque and not permutationSpaceHasNewLeaf:
			something: tuple[Pile, LeafSpace] = Z0Z_deque.popleft()  # noqa: F841
			# forbiddenLeaves
			leafAntiOptions: LeafOptions = makeLeafAntiOptions(state.leavesTotal, forbiddenLeaves)  # noqa: F841

	return permutationSpace

#-------- Functions that do NOT use the shared logic -----------------------------------------

@syntacticCurry
def _reducePermutationSpace_leafDomainOf1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to detect and pin leaves with domain size one.

	I use this constraint encoder to detect leaves that can appear at only one pile (domain size
	one) and pin those leaves. I compute the domain size for each leaf by counting how many piles
	contain that leaf (either pinned or in `LeafOptions`). When a leaf appears at exactly one
	pile, I pin that leaf at that pile using `atPilePinLeaf` [1] and propagate the pinning using
	`_reducePermutationSpace_LeafIsPinned`.

	The function also validates that every leaf has nonzero domain size. When any leaf has zero
	domain (cannot appear anywhere), I invalidate `permutationSpace` by returning `None`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	References
	----------
	[1] mapFolding._e.pinIt.atPilePinLeaf
		Internal package reference

	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)

		counterLeafDomainSize: Counter[Leaf] = Counter(chain(chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

		if set(range(state.leavesTotal)).difference(counterLeafDomainSize.keys()):
			return None

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(valfilter((1).__eq__, counterLeafDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])
		if leavesWithDomainOf1:
			leaf: Leaf = leavesWithDomainOf1.pop()
			sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, atPilePinLeaf(permutationSpace, one(DOTkeys(valfilter(leafIsInPileRange(leaf), pilesUndetermined))), leaf))
			if (sherpa is None) or (not sherpa):
				return None
			else:
				permutationSpace = sherpa
			permutationSpaceHasNewLeaf = True
	return permutationSpace
