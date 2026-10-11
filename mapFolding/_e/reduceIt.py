"""Reduce leaf sets in elimination permutation spaces.

(AI generated docstring)

You can propagate pinned leaves, restrict repeated leaf sets, and detect leaves with no remaining pile
or only one remaining pile. Use the default reduction sequence to also restrict leaf choices using
crease crossings.

Contents
--------
Functions
	reduceLeafSpace
		Restrict specified pile domains with an allowed-leaf mask.
	reducePermutationSpace_LeafIsPinned
		Remove pinned leaves from the remaining pile leaf sets.
	reducePermutationSpace_leafDomainOf0or1
		Pin leaves with one remaining pile and invalidate leaves with no pile.
	reducePermutationSpace_nakedSubset
		Reserve identical leaf sets for matching groups of piles.
Reduction sequences
	boxOfFunctionsReductionDEFAULT
		Access the default sequence of permutation-space reductions.
"""

from __future__ import annotations

from collections import Counter
from functools import partial
from humpy_cytoolz import dissoc, get, groupby as toolz_groupby, valfilter as filterLeaf, valfilter as filterValue
from hunterMakesPy import errorL33T, inclusive, raiseIfNone
from itertools import chain, combinations
from mapFolding import _e
from mapFolding._e.algorithms.iff import creaseViolation吗, getCreasePost, oddLeaf吗
from mapFolding._e.filters import leafInChoicesLeaf吗, leafPinned吗, pileChoicesLeaf吗
from mapFolding._e.theTypes import ChoicesLeaf
from mapFolding.theTypes import Leaf
from more_itertools import extract, first, one
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems, DOTvalues, reverseLookup

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Sequence
	from mapFolding._e.dataBaskets import PermutationSpace, StateElimination
	from mapFolding._e.theTypes import LeafSpace, PinnedLeaves, UndeterminedPiles
	from mapFolding.theTypes import Pile

def reduceLeafSpace(permutationSpace: PermutationSpace, pilesToUpdate: Iterable[tuple[Pile, ChoicesLeaf]], antiChoicesLeaf: ChoicesLeaf) -> PermutationSpace:
	"""Restrict specified pile domains with an allowed-leaf mask.

	(AI generated docstring)

	You can intersect each supplied leaf set with `antiChoicesLeaf` and write the normalized result into
	`permutationSpace`. You pin a leaf when only one remains. With an empty result, you set the validity
	flag to `False` and return immediately.

	Domain preservation
	-------------------
	You must supply a bitset that appropriately restricts each pile. Check membership in the existing
	dictionary value yourself; your supplied bitset is not intersected with that value.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Permutation space to mutate.
	pilesToUpdate : Iterable[tuple[Pile, ChoicesLeaf]]
		Pile indices and bitsets to intersect with the mask. The supplied bitsets, rather than the
		current dictionary values, determine the new choices.
	antiChoicesLeaf : ChoicesLeaf
		Allowed-leaf mask including the sentinel. Cleared bits identify excluded leaves.

	Returns
	-------
	permutationSpace : PermutationSpace
		The same object with updated pile domains or a false validity flag. Earlier updates remain applied on
		failure.
	"""
	for pile, choicesLeaf in pilesToUpdate:
		leafSpace: LeafSpace | None = _e.choicesLeafLeafNone(_e.choicesLeafAND(antiChoicesLeaf, choicesLeaf))
		if leafSpace is None:
			#=SIN= Early return.
			permutationSpace.valid = False
			return permutationSpace
		else:
			permutationSpace[pile] = leafSpace
	return permutationSpace

def _odd吗(mapShape: tuple[int, ...], dimension: int) -> Callable[[tuple[Pile, Leaf]], bool]:
	def workhorse(pileLeaf: tuple[Pile, Leaf]) -> bool:
		return bool(oddLeaf吗(mapShape, dimension=dimension, leaf=pileLeaf[1]))
	return workhorse

def _crossedCreases(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Detect and eliminate crossed creases.

	You can detect configurations where two creases would cross and either invalidate `permutationSpace`
	or restrict forbidden pile positions for unpinned crease leaves. For each dimension, you partition
	pinned leaves by coordinate parity, identify crease pairs with one pinned and one unpinned leaf, and
	compute forbidden pile positions where the unpinned leaf would cause a crossing.

	Parameters
	----------
	state : StateElimination
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: choicesLeaf`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.
	"""
	pileOf_kCrease: Pile = errorL33T
	pileOf_rCrease: Pile = errorL33T
	pilesForbidden: Iterable[Pile] = []
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		for dimension in range(state.totalDimensions):
			groupedByParity: dict[bool, list[tuple[Pile, Leaf]]] = toolz_groupby(_odd吗(state.mapShape, dimension), DOTitems(permutationSpace.pinnedLeaves()))

			for upDown, leftRight in ((False, True), (True, False)):
				leavesPinnedParityOpposite: PinnedLeaves = dict(get(upDown, groupedByParity, ()))

				for ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) in combinations(sorted(get(leftRight, groupedByParity, ())), 2):
					leaf_kCrease: Leaf | None = getCreasePost(state.mapShape, leaf_k, dimension)
					if leaf_kCrease is None:
						continue
					leaf_rCrease: Leaf | None = getCreasePost(state.mapShape, leaf_r, dimension)
					if leaf_rCrease is None:
						continue

					if leaf_kCreaseIsPinned := leafPinned吗(leavesPinnedParityOpposite, leaf_kCrease):
						pileOf_kCrease = raiseIfNone(reverseLookup(leavesPinnedParityOpposite, leaf_kCrease))
					if leaf_rCreaseIsPinned := leafPinned吗(leavesPinnedParityOpposite, leaf_rCrease):
						pileOf_rCrease = raiseIfNone(reverseLookup(leavesPinnedParityOpposite, leaf_rCrease))

					if leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
						antiChoicesLeaf: ChoicesLeaf = _e.makeAntiChoicesLeaf(state.totalLeaves, (leaf_rCrease,))

						if pileOf_k < pileOf_r < pileOf_kCrease:
							pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
						elif pileOf_kCrease < pileOf_r < pileOf_k:
							pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
						elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
							pilesForbidden = range(pileOf_kCrease + 1, pileOf_k)
						elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
							pilesForbidden = range(pileOf_k + 1, pileOf_kCrease)

					elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
						antiChoicesLeaf = _e.makeAntiChoicesLeaf(state.totalLeaves, (leaf_kCrease,))

						if pileOf_rCrease < pileOf_k < pileOf_r:
							pilesForbidden = frozenset([*range(pileOf_rCrease), *range(pileOf_r + 1, state.pileLast + inclusive)])
						elif pileOf_r < pileOf_k < pileOf_rCrease:
							pilesForbidden = frozenset([*range(pileOf_r), *range(pileOf_rCrease + 1, state.pileLast + inclusive)])
						elif (pileOf_k < pileOf_r < pileOf_rCrease) or (pileOf_r < pileOf_rCrease < pileOf_k):
							pilesForbidden = range(pileOf_r + 1, pileOf_rCrease)
						elif (pileOf_k < pileOf_rCrease < pileOf_r) or (pileOf_rCrease < pileOf_r < pileOf_k):
							pilesForbidden = range(pileOf_rCrease + 1, pileOf_r)

					elif leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
						if creaseViolation吗(pileOf_k, pileOf_r, pileOf_kCrease, pileOf_rCrease):
							permutationSpace.valid = False
							#=SIN= Early return.
							return permutationSpace
						continue

					else:
						continue

					permutationSpace = reduceLeafSpace(permutationSpace
							, filter(pileChoicesLeaf吗, extract(permutationSpace.items(), pilesForbidden))
							, antiChoicesLeaf
					)
					if not permutationSpace.valid:
						#=SIN= Early return.
						return permutationSpace

		if leafCount < permutationSpace.leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_LeafIsPinned(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Remove pinned leaves from the remaining pile leaf sets.

	You can enforce one pile per pinned leaf by removing pinned leaf indices from undetermined piles. You
	repeat when an exclusion pins another leaf and stop when no additional leaf is pinned or the
	permutation space becomes invalid.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the total leaf count.
	permutationSpace : PermutationSpace
		Permutation space to mutate.

	Returns
	-------
	permutationSpace : PermutationSpace
		The same object with propagated assignments or a false validity flag.
	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False
		leavesPinned, pilesUndetermined = permutationSpace.bifurcate()
		permutationSpace = reduceLeafSpace(permutationSpace, DOTitems(pilesUndetermined), _e.makeAntiChoicesLeaf(state.totalLeaves, DOTvalues(leavesPinned)))
		if len(leavesPinned) < permutationSpace.leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_nakedSubset(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Reserve identical leaf sets for matching groups of piles.

	You can find groups of at least two piles with the same leaf bitset. When the leaf count equals the
	pile count, you exclude those leaves from every other undetermined pile.

	Subset coverage
	---------------
	You compare identical bitsets rather than arbitrary unions of leaf sets. The current repeat condition
	requires the pinned-leaf count to decrease, so use the state's repeated reduction method [1] for
	further propagation.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the total leaf count.
	permutationSpace : PermutationSpace
		Permutation space to mutate.

	Returns
	-------
	permutationSpace : PermutationSpace
		The same object with reduced leaf sets or a false validity flag.

	References
	----------
	[1] `mapFolding._e.dataBaskets.StateElimination.reduceAllPermutationSpace`
	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		pilesUndetermined: UndeterminedPiles = permutationSpace.undeterminedPiles()
		groupByChoicesLeaf: dict[ChoicesLeaf, set[Pile]] = {}
		for pile, choicesLeaf in pilesUndetermined.items():
			groupByChoicesLeaf.setdefault(choicesLeaf, set()).add(pile)

		groupByChoicesLeaf = filterValue(lambda boxOfPiles: 1 < len(boxOfPiles), groupByChoicesLeaf)
		for choicesLeaf, boxOfPiles in groupByChoicesLeaf.items():
			if _e.lengthChoicesLeaf(choicesLeaf) == len(boxOfPiles):

				permutationSpace = reduceLeafSpace(permutationSpace, DOTitems(dissoc(pilesUndetermined, *boxOfPiles))
					, _e.makeAntiChoicesLeaf(state.totalLeaves, _e.getIteratorOfLeaves(choicesLeaf))
				)

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_leafDomainOf0or1(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Pin leaves with one remaining pile and invalidate leaves with no pile.

	You can count each leaf's occurrences across pinned assignments and leaf sets. You invalidate the
	permutation space for a missing leaf, or pin an unpinned leaf with one possible pile and propagate
	pinned-leaf exclusions before counting again.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the total leaf count.
	permutationSpace : PermutationSpace
		Permutation space to reduce. Pinning can create a replacement object.

	Returns
	-------
	permutationSpace : PermutationSpace
		Reduced permutation space, possibly a replacement object, with a false validity flag if any leaf has
		no pile.
	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = permutationSpace.bifurcate()

		countDomainSizes: Counter[Leaf] = Counter(chain(chain.from_iterable(map(_e.getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

		if set(range(state.totalLeaves)).difference(countDomainSizes.keys()):
			permutationSpace.valid = False
		else:
			leaf: Leaf | None = first(set(filterValue((1).__eq__, countDomainSizes, factory=dict[Leaf, int])).difference(leavesPinned.values()).difference([state.totalLeaves]), None)
			if leaf is not None:
				permutationSpace = reducePermutationSpace_LeafIsPinned(state, permutationSpace.atPilePinLeaf(one(filterLeaf(partial(leafInChoicesLeaf吗, leaf), pilesUndetermined, factory=dict[Leaf, ChoicesLeaf])), leaf))
				permutationSpaceHasNewLeaf = True
	return permutationSpace

boxOfFunctionsReductionDEFAULT: Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]] = (
	reducePermutationSpace_nakedSubset
	, reducePermutationSpace_leafDomainOf0or1
	, _crossedCreases
	, reducePermutationSpace_LeafIsPinned
)
"""Access the default sequence of permutation-space reductions.

(AI generated docstring)

You can restrict matching leaf sets, pin leaves with a single pile domain, remove crease crossings,
and propagate pinned leaves. With the state reduction method [1], you apply the sequence in reverse
order and restart when dictionary values change.

References
----------
[1] `mapFolding._e.dataBaskets.StateElimination.reduceAllPermutationSpace`
"""
