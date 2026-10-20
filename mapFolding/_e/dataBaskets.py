"""Store and manipulate elimination search spaces and computation state.

You can represent pile assignments as permutation spaces and store active branches, completed
foldings, and symmetry multipliers in elimination state. Check each operation for its mutation and
replacement behavior.

Contents
--------
Classes
	PermutationSpace
		Store pinned leaves and leaf sets keyed by pile index.
	StateElimination
		Store search branches, folding representatives, and map-derived elimination values.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from gmpy2 import bit_mask
from humpy_cytoolz import (
	assoc as associateKeyValue, compose, dissoc as dissociatePile, first, groupby as toolz_groupby, merge, valfilter as filterLeaf)
from hunterMakesPy import raiseIfNone
from itertools import combinations as Y7Y_combinations, filterfalse
from mapFolding import _e
from mapFolding._e.algorithms.iff import creaseViolation吗, getCreasePost, oddLeaf吗
from mapFolding._e.filters import choicesLeaf吗, leafInChoicesLeaf吗, leaf吗, 是valid
from mapFolding._e.reduceIt import boxOfFunctionsReductionDEFAULT
from mapFolding._e.theTypes import LeafSpace
from mapFolding.beDRY import getTotalLeaves, validateMapShape
from mapFolding.theTypes import Folding, Pile
from math import prod
from operator import attrgetter, methodcaller
from typing import cast, overload, TYPE_CHECKING
from Z0Z_tools import DOTitems, DOTkeys, DOTvalues
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding._e.theTypes import ChoicesLeaf, PinnedLeaves, UndeterminedPiles
	from mapFolding.theTypes import Leaf
	from typing import Self

#=EndNotes##pinning=
class PermutationSpace(dict[Pile, LeafSpace]):
	"""Store pinned leaves and leaf sets keyed by pile index.

	You can represent a partial leaf arrangement by mapping each pile to either a pinned integer leaf or
	a leaf bitset. You can normalize singleton leaf sets and mark impossible assignments with a false
	validity flag. Include every pile in a complete search-space mapping.

	Keep permutation spaces disjoint in collections you use for counting. When you pin a leaf, replace
	the source space with the returned branches to avoid counting overlapping spaces.

	Attributes
	----------
	valid : bool = True
		Validity flag. Reduction and exclusion operations set the flag to `False` when an assignment is
		impossible.
	leafCount : int
		Number of values already represented as pinned integer leaves. Reading the property does not
		normalize bitsets.
	"""

	#============== Modify inherited methods and attributes =======================================

	"""
	disable:
		del d[key]
		clear()
		pop
		popitem

	fromkeys(iterable, value=None): remove the default of `value`.
	setdefault(key, default=None, /): remove the default of `default`.
	"""

	def copy(self) -> PermutationSpace:
		"""Copy pile assignments into a new permutation space.

		(AI generated docstring)

		You can create a shallow copy of the dictionary entries. You receive a copy with the default
		validity flag, without copying an instance-level `valid` value.

		Returns
		-------
		permutationSpace : PermutationSpace
			New permutation space containing the same pile assignments.
		"""
		return PermutationSpace(self)

	#============== New methods and attributes ====================================================

	def atPileExcludeLeaf(self, pile: Pile, leaf: Leaf) -> PermutationSpace:
		"""Exclude a leaf from a selected pile.

		(AI generated docstring)

		You can exclude `leaf` from the choices at `pile` and normalize the resulting leaf set. You mark the
		permutation space invalid if you exclude an already pinned leaf or empty the set. You receive
		`KeyError` for a missing pile key.

		Parameters
		----------
		pile : Pile
			Existing pile index whose assignment will be modified.
		leaf : Leaf
			Leaf index to exclude.

		Returns
		-------
		permutationSpace : PermutationSpace
			The same object after exclusion and choice normalization.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		leafSpaceAtPile: LeafSpace = self[pile]
		if choicesLeaf吗(leafSpaceAtPile):
			self[pile] = leafSpaceAtPile.bit_clear(leaf)
			self._solidifyLeafSpaceAtPile(pile)
		elif leafSpaceAtPile == leaf:
			self.valid = False
		return self

	def atPilePinLeaf(self, pile: Pile, leaf: Leaf) -> PermutationSpace:
		"""Create a new permutation space with a leaf pinned at a selected pile.

		You can copy the assignments and replace the value at `pile` with `leaf`, leaving the source
		dictionary unchanged. Validate the assignment and propagate exclusions to other piles separately.

		Assignment integrity
		--------------------
		Ensure that `pile` is open or already holds `leaf`, that `leaf` is allowed, and that `leaf` is not
		pinned at a different pile. You can overwrite an assignment or duplicate a leaf if you violate those
		requirements. Replace the source permutation space with the new branch before counting to avoid
		overlap.

		Parameters
		----------
		pile : Pile
			Pile at which to pin `leaf`.
		leaf : Leaf
			Leaf index to pin.

		Returns
		-------
		permutationSpace : PermutationSpace
			New permutation space with the requested assignment and the default validity flag.

		Examples
		--------
		The domain expansion replaces a working branch after pinning.

			```python
			permutationSpace = permutationSpace.atPilePinLeaf(boxOfPiles[次], leaves[次])
			```
		"""
		return PermutationSpace(associateKeyValue(self, pile, leaf, PermutationSpace))

	def bifurcate(self) -> tuple[PinnedLeaves, UndeterminedPiles]:
		"""Separate pinned leaves from possible pile assignments.

		You can partition the current dictionary values into fixed assignments and leaf bitsets without
		normalizing singleton leaf sets.

		Returns
		-------
		leavesPinned : PinnedLeaves
			New dictionary of piles mapped to pinned leaf indices.
		pilesUndetermined : UndeterminedPiles
			New dictionary of piles mapped to leaf bitsets.
		"""
		leavesPinned: PinnedLeaves = self.pinnedLeaves()
		#=SIN= `cast`: type checkers cannot infer that partitioning `PermutationSpace` preserves `UndeterminedPiles`.
		return (leavesPinned, cast('UndeterminedPiles', dissociatePile(self, *DOTkeys(leavesPinned))))

	def deconstructDomainOfLeaf(self, leaf: Leaf, leafDomain: Iterable[Pile]) -> list[PermutationSpace]:
		"""Create branches that pin a leaf within its supplied pile domain.

		You can create one branch for each open pile in `leafDomain` whose leaf set contains `leaf`. If
		`leaf` is already pinned, you receive the original object in a list without checking the supplied
		domain.

		Source normalization
		--------------------
		You can normalize the source dictionary and change its validity flag when testing possible piles.
		Replace the source branch with the returned branches when maintaining a disjoint search-space
		collection.

		Parameters
		----------
		leaf : Leaf
			Leaf index to pin.
		leafDomain : Iterable[Pile]
			Possible pile positions for `leaf`. Each position must exist in the mapping.

		Returns
		-------
		deconstructedPermutationSpace : list[PermutationSpace]
			New pinning branches, or a list containing the original object when the leaf is already pinned.
		"""
		deconstructedPermutationSpace: list[PermutationSpace] = []
		if self.leafNotPinned吗(leaf):
			leafInPileRange: Callable[[int], bool] = compose(
				partial(leafInChoicesLeaf吗, leaf), partial(self.getChoicesLeaf, default=bit_mask(len(self)))
			)
			pinLeafAt: Callable[[int], PermutationSpace] = partial(self.atPilePinLeaf, leaf=leaf)
			deconstructedPermutationSpace.extend(map(pinLeafAt, filter(leafInPileRange, filter(self.pileUndetermined吗, leafDomain))))
		else:
			deconstructedPermutationSpace.append(self)
		return deconstructedPermutationSpace

	def deconstructDomainsCombined(self, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]]) -> list[PermutationSpace]:
		"""Create branches for compatible Z0Z_combinations of leaf pile positions.

		You can match each leaf in `leaves` with the corresponding pile position in each item of
		`leavesDomain` and create a new pinned branch for each compatible combination. If every requested
		leaf is already pinned, you receive the original object without checking the Z0Z_combinations.

		Source normalization
		--------------------
		You normalize the source permutation space before testing Z0Z_combinations. Replace the source with the
		returned branches when maintaining a disjoint search-space collection.

		Parameters
		----------
		leaves : Sequence[Leaf]
			Distinct leaf indices to pin, ordered to match each pile-position sequence.
		leavesDomain : Iterable[Sequence[Pile]]
			Possible pile-position sequences, each with one distinct existing pile per leaf.

		Returns
		-------
		deconstructedPermutationSpace : list[PermutationSpace]
			New branches for compatible Z0Z_combinations, or the original object when all requested leaves are
			pinned.
		"""
		deconstructedPermutationSpace: list[PermutationSpace] = []

		def pileOpenByIndex(次: int) -> CallableFunction[[Sequence[Pile]], bool]:
			def workhorse(domain: Sequence[Pile]) -> bool:
				return self.pileUndetermined吗(domain[次])
			return workhorse

		def leafInPileRangeByIndex(次: int) -> CallableFunction[[Sequence[Pile]], bool]:
			def workhorse(domain: Sequence[Pile]) -> bool:
				choicesLeaf: ChoicesLeaf = raiseIfNone(self.getChoicesLeaf(domain[次], default=bit_mask(len(self))))
				return leafInChoicesLeaf吗(leaves[次], choicesLeaf)
			return workhorse

		def isPinnedAtPileByIndex(leaf: Leaf, 次: int) -> CallableFunction[[Sequence[Pile]], bool]:
			def workhorse(domain: Sequence[Pile]) -> bool:
				return self.leafPinnedAtPile吗(leaf, domain[次])
			return workhorse

		self._solidifyLeafSpace()
		if any(map(self.leafNotPinned吗, leaves)):
			for 次 in range(len(leaves)):
				"""Redefine leavesDomain by filtering out domains that are not possible with the current `PermutationSpace`."""
				if self.leafNotPinned吗(leaves[次]):
					"""`leaves[次]` is not pinned, so it needs a pile.
					In each iteration of `leavesDomain`, `boxOfPiles`, the pile it needs is `boxOfPiles[次]`.
					Therefore, if `boxOfPiles[次]` is open, filter in the iteration. If `boxOfPiles[次]` is occupied, filter out the iteration."""
					leavesDomain = filter(pileOpenByIndex(次), leavesDomain)
					"""`leaves[次]` is not pinned, it wants `boxOfPiles[次]`, and `boxOfPiles[次]` is open.
					Is `leaves[次]` in the pile-range of `boxOfPiles[次]`?"""
					leavesDomain = filter(leafInPileRangeByIndex(次), leavesDomain)
				else:
					"""`leaves[次]` is pinned.
					In each iteration of `leavesDomain`, `boxOfPiles`, the pile in which `leaves[次]` is pinned must match `boxOfPiles[次]`.
					Therefore, if the pile in which `leaves[次]` is pinned matches `boxOfPiles[次]`, filter in the iteration. Otherwise, filter out the iteration."""
					leavesDomain = filter(isPinnedAtPileByIndex(leaves[次], 次), leavesDomain)

			for boxOfPiles in leavesDomain:
				"""Properly and safely deconstruct `permutationSpace` by the combined domain of leaves.
				The parameter `leavesDomain` is the full domain of the leaves, so deconstructing with `leavesDomain` preserves the permutation space.
				For each leaf in `leaves`, you filter out occupied piles to avoid overwriting pinned leaves and
				invalidating the permutation space.
				You apply filters that prevent pinning the same leaf twice.
				For each domain in `leavesDomain`, you can therefore pin `leaves[次]` at `boxOfPiles[次]` without
				corrupting the permutation space."""
				permutationSpace: PermutationSpace = self.copy()
				for 次 in range(len(leaves)):
					permutationSpace = permutationSpace.atPilePinLeaf(boxOfPiles[次], leaves[次])
				deconstructedPermutationSpace.append(permutationSpace)
		else:
			deconstructedPermutationSpace.append(self)

		return deconstructedPermutationSpace

	def deconstructPile(self, pile: Pile | None = None, leavesToPin: Iterable[Leaf] = ()) -> Iterable[PermutationSpace]:
		"""Create branches that pin possible leaves at a selected pile.

		(AI generated docstring)

		You can expand the leaf choices at `pile` into alternative pinning branches. Pass `None` for `pile`
		to select the first dictionary entry holding a leaf bitset. You can normalize the source assignment
		when testing the selected pile.

		Branch consumption
		------------------
		Consume the returned iterable before changing the source assignments, because leaf filtering
		reads the source when iteration occurs. Replace the source branch with its alternatives when
		maintaining a disjoint search-space collection.

		Parameters
		----------
		pile : Pile | None = None
			Existing pile to expand. `None` requires at least one bitset entry.
		leavesToPin : Iterable[Leaf] = ()
			Leaf indices to pin. A false-valued argument uses every choice in the selected bitset. Supply
			leaves that belong to that bitset. You filter out only leaves already pinned elsewhere.

		Returns
		-------
		deconstructed : Iterable[PermutationSpace]
			Lazy iterable of new branches, or a list containing the original object if the pile has no choice
			bitset after normalization.
		"""
		if pile is None:
			pile = first(filterLeaf(choicesLeaf吗, self))
		if (choicesLeaf := self.getChoicesLeaf(pile)) is None:
			deconstructed: Iterable[PermutationSpace] = [self]
		else:
			leavesToPin = leavesToPin or _e.getIteratorOfLeaves(choicesLeaf)
			deconstructed = map(partial(self.atPilePinLeaf, pile), filter(self.leafNotPinned吗, leavesToPin))
		return deconstructed

	@overload
	def getLeaf(self, pile: Pile, default: None = None) -> Leaf | None: ...
	@overload
	def getLeaf(self, pile: Pile, default: Leaf) -> Leaf: ...
	@overload
	def getLeaf[个](self, pile: Pile, default: 个) -> Leaf | 个: ...
	def getLeaf[个](self, pile: Pile, default: Leaf | 个 | None = None) -> Leaf | 个 | None:
		"""Read a pinned leaf at a pile or return a fallback value.

		You can normalize the assignment at `pile` and read the pinned leaf when the normalized value is an
		integer leaf index. You receive `default` for a remaining leaf bitset, or `KeyError` for a missing
		pile key.

		Parameters
		----------
		pile : Pile
			Existing pile index to inspect.
		default : Leaf | 个 | None = None
			Fallback for an assignment that remains a bitset.

		Returns
		-------
		leafOrDefault : Leaf | 个 | None
			Pinned leaf index or the supplied fallback.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		ImaLeaf: LeafSpace = self[pile]
		if leaf吗(ImaLeaf):
			return ImaLeaf
		return default

	@overload
	def getChoicesLeaf(self, pile: Pile, default: None = None) -> ChoicesLeaf | None: ...
	@overload
	def getChoicesLeaf(self, pile: Pile, default: ChoicesLeaf) -> ChoicesLeaf: ...
	@overload
	def getChoicesLeaf[个](self, pile: Pile, default: 个) -> ChoicesLeaf | 个: ...
	def getChoicesLeaf[个](self, pile: Pile, default: ChoicesLeaf | 个 | None = None) -> ChoicesLeaf | 个 | None:
		"""Read a pile's bitset or return a fallback value.

		You can normalize the assignment at `pile` and read its bitset when leaf choices remain. You receive
		`default` for a pinned leaf. With an empty leaf set, you mark the permutation space invalid but
		still receive a bitset. You receive `KeyError` for a missing pile key.

		Parameters
		----------
		pile : Pile
			Existing pile index to inspect.
		default : ChoicesLeaf | 个 | None = None
			Fallback for a pinned leaf assignment.

		Returns
		-------
		choicesLeafOrDefault : ChoicesLeaf | 个 | None
			Remaining bitset or the supplied fallback.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		ImaChoicesLeaf: LeafSpace = self[pile]
		if choicesLeaf吗(ImaChoicesLeaf):
			return ImaChoicesLeaf
		return default

	@property
	def leafCount(self) -> int:
		"""Count assignments already represented as pinned leaves.

		You can count integer leaf values in the current dictionary without normalizing leaf sets. You count
		entries rather than distinct leaf indices.

		Returns
		-------
		leafCount : int
			Number of dictionary values represented as pinned leaves.
		"""
		return sum(map(leaf吗, self.values()))

	def leafNotPinned吗(self, leaf: Leaf) -> bool:
		"""Test whether a leaf is absent from the pinned assignments.

		You can check whether `leaf` occurs among the dictionary values without normalizing leaf bitsets.
		Treat a singleton leaf set as unpinned until you normalize it.

		Parameters
		----------
		leaf : Leaf
			Leaf index to find.

		Returns
		-------
		leafIsNotPinned : bool
			`True` when `leaf` is absent from the dictionary values.
		"""
		return leaf not in self.values()

	def leafPinned吗(self, leaf: Leaf) -> bool:
		"""Test whether a leaf is present in the pinned assignments.

		You can check whether `leaf` occurs among the dictionary values without normalizing leaf bitsets.

		Parameters
		----------
		leaf : Leaf
			Leaf index to find.

		Returns
		-------
		leafIsPinned : bool
			`True` when `leaf` occurs among the dictionary values.
		"""
		return leaf in self.values()

	def leafPinnedAtPile吗(self, leaf: Leaf, pile: Pile) -> bool:
		"""Test whether a selected pile holds a selected pinned leaf.

		You can normalize the assignment at `pile` and compare the resulting value with `leaf`. You can pin
		the only remaining leaf or mark the space invalid during normalization. You receive `KeyError` for a
		missing pile key.

		Parameters
		----------
		leaf : Leaf
			Leaf index whose assignment is being checked.
		pile : Pile
			Existing pile index to inspect.

		Returns
		-------
		leafIsPinnedAtPile : bool
			`True` when the normalized assignment at `pile` equals `leaf`.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		return leaf == self[pile]

	def makeFolding(self, leavesToInsert: Sequence[Leaf] = ()) -> Folding:
		"""Complete pile assignments and return the leaf order as a folding.

		(AI generated docstring)

		You can fill undetermined piles with `leavesToInsert` and receive all leaf assignments sorted by
		pile index, leaving the source dictionary unchanged. Validate uniqueness, leaf membership, and
		crease crossings separately. You receive `ValueError` if the supplied leaf count differs from the
		undetermined pile count.

		Parameters
		----------
		leavesToInsert : Sequence[Leaf] = ()
			One leaf per undetermined pile, matching the insertion order of the undetermined-pile dictionary. The
			empty default requires every pile to be pinned.

		Returns
		-------
		folding : Folding
			Pinned and inserted leaf indices ordered by ascending pile index.
		"""
		pilesToInsert: Iterator[Pile] = DOTkeys(self.undeterminedPiles())
		#=SIN= `cast` because the type checkers cannot possible know that the prior logic leads to all int.
		return tuple(DOTvalues(dict(sorted(DOTitems(cast('PinnedLeaves', merge(self, dict(zip(pilesToInsert, leavesToInsert, strict=True)), factory=PermutationSpace)))))))

	def pilePinned吗(self, pile: Pile) -> bool:
		"""Test whether a pile holds a pinned leaf after normalization.

		You can normalize the assignment at `pile` and test whether the result is an integer leaf. You can
		modify the dictionary or validity flag during normalization. You receive `KeyError` for a missing
		pile key.

		Parameters
		----------
		pile : Pile
			Existing pile index to inspect.

		Returns
		-------
		pileIsPinned : bool
			`True` when the normalized assignment is a pinned leaf.

		See Also
		--------
		`pileUndetermined吗`
			Check whether the pile still holds a bitset.
		`mapFolding._e.filters.leaf吗`
			Inspect an existing value without normalizing a pile assignment.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		return leaf吗(self[pile])

	def pileUndetermined吗(self, pile: Pile) -> bool:
		"""Test whether a pile still holds a bitset after normalization.

		You can normalize the assignment at `pile` and test whether the result is a leaf bitset. With an
		empty leaf set, you mark the permutation space invalid but retain the bitset. You receive `KeyError`
		for a missing pile key.

		Parameters
		----------
		pile : Pile
			Existing pile index to inspect.

		Returns
		-------
		pileIsUndetermined : bool
			`True` when the normalized assignment remains a bitset.

		See Also
		--------
		`pilePinned吗`
			Check whether the pile holds a pinned leaf.
		`mapFolding._e.filters.choicesLeaf吗`
			Inspect an existing value without normalizing a pile assignment.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		return choicesLeaf吗(self[pile])

	def _solidifyLeafSpace(self) -> None:
		count: int = self.leafCount
		if count < len(self):
			tuple(map(self._solidifyLeafSpaceAtPile, self))
			leavesPinned: PinnedLeaves = self.pinnedLeaves()
			antiChoicesLeaf: ChoicesLeaf = _e.makeAntiChoicesLeaf(len(self), leavesPinned.values())
			for pile in filterfalse(leavesPinned.__contains__, self):
				self[pile] &= antiChoicesLeaf  # ty: ignore[invalid-argument-type]
			tuple(map(self._solidifyLeafSpaceAtPile, leavesPinned))
			if count < self.leafCount:
				self._solidifyLeafSpace()

	def _solidifyLeafSpaceAtPile(self, pile: Pile) -> None:
		rangeOfPile: LeafSpace | None = self[pile]
		if choicesLeaf吗(rangeOfPile):
			# If the range size of `pile` is 0 or 1, convert to None or `Leaf`.
			rangeOfPile = _e.choicesLeafLeafNone(rangeOfPile)
			if rangeOfPile is None:
				self.valid = False
			elif choicesLeaf吗(rangeOfPile):
				self[pile] = rangeOfPile
			elif rangeOfPile in self.values():
				self.valid = False
			else:
				self[pile] = rangeOfPile

	# Does it matter whether this is a property or a method?
	def pinnedLeaves(self) -> PinnedLeaves:
		"""Extract the pile assignments already represented as pinned leaves.

		You can create a dictionary containing integer leaf assignments without normalizing leaf bitsets or
		validating the assignments.

		Returns
		-------
		leavesPinned : PinnedLeaves
			New pile-to-leaf dictionary in the source dictionary's insertion order.
		"""
		return filterLeaf(leaf吗, self)

	def undeterminedPiles(self) -> UndeterminedPiles:
		"""Extract the pile assignments represented as bitsets.

		You can create a dictionary containing leaf bitsets without normalizing empty or singleton leaf
		sets.

		Returns
		-------
		pilesUndetermined : UndeterminedPiles
			New pile-to-bitset dictionary in the source dictionary's insertion order.
		"""
		return filterLeaf(choicesLeaf吗, self)

	def updatePilesMissing(self, missing: PermutationSpace | UndeterminedPiles | PinnedLeaves) -> PermutationSpace:
		"""Create a complete pile mapping while preserving existing assignments.

		You can merge `missing` with the existing assignments, giving existing assignments precedence. You
		receive a new permutation space sorted by pile index, leaving the source dictionary unchanged.

		Parameters
		----------
		missing : PermutationSpace | UndeterminedPiles | PinnedLeaves
			Pile assignments used only for keys absent from the source dictionary.

		Returns
		-------
		permutationSpace : PermutationSpace
			New permutation space with merged assignments and the default validity flag.
		"""
		#=EndNotes##sorted=
		self = PermutationSpace(sorted(DOTitems(merge(missing, self))))
		#=Wrong= It is necessary to assign to self before returning.
		return self  # ruff: ignore[unnecessary-assign]

	valid: bool = True

@dataclasses.dataclass(slots=True)
class StateElimination:
	"""Store search branches, folding representatives, and map-derived elimination values.

	You can keep the map description, current search spaces, and counting results together. You receive a
	validated map shape, leaf counts, and dimension strides during initialization. Use the methods to
	update search branches and return the same state for successive operations.

	Constructor fields are the map shape, active spaces, workbench space, reduction sequence,
	representative count, folding collection, workbench pile, and four symmetry multipliers. Derived
	fields are computed during initialization.

	Attributes
	----------
	mapShape : tuple[int, ...]
		Validated nonnegative dimension lengths in the supplied order.
	boxOfPermutationSpace : list[PermutationSpace]
		Active disjoint search branches. The default is an empty list.
	permutationSpace : PermutationSpace
		Current workbench branch. The default is an empty permutation space.
	boxOfFunctionsReduction : Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]]
		Preferred reduction sequence. The default is an empty list, which selects the default reductions.
	groupsOfFolds : int = 0
		Count of folding representatives before applying symmetry multipliers.
	boxOfFolding : list[Folding]
		Collected folding representatives. The default is an empty list.
	pile : Pile = -1
		Current workbench pile index, initially unset.
	Theorem2aMultiplier : int = 1
		Origin-pinning multiplier, initialized to the total leaf count when positive.
	Theorem2Multiplier : int = 1
		Additional factor used by the Theorem 2(b) ordering restriction.
	Theorem3Multiplier : int = 1
		Additional symmetry multiplier included in the total fold count.
	Theorem4Multiplier : int = 1
		Dimension-ordering symmetry multiplier.
	totalDimensions : int
		Number of dimension lengths in the validated shape.
	leafLast : Leaf
		Largest zero-based leaf index, or `-1` for a map with zero leaves.
	totalLeaves : int
		Total number of leaves derived from the validated shape.
	pileLast : Pile
		Largest zero-based pile index, or `-1` for a map with zero leaves.
	pilesTotal : int
		Total number of piles, equal to the total leaf count.
	mapShapeProducts : tuple[int, ...]
		Cumulative dimension products from the empty product through the total leaf count.
	mapShapeProductsSums : tuple[int, ...]
		Cumulative sums of dimension products, including the initial empty sum.
	mapShape首ProductsSums : tuple[int, ...]
		Cumulative sums of reversed dimension strides, excluding the total leaf-count product.
	首 : int
		Out-of-bounds leaf index equal to the total leaf count.
	totalFolds : int
		Representative count multiplied by all stored symmetry factors.
	"""

	mapShape: tuple[int, ...] = dataclasses.field(init=True)
	"""Inspect the validated dimension lengths of the map."""

	boxOfPermutationSpace: list[PermutationSpace] = dataclasses.field(default_factory=list[PermutationSpace], init=True)
	"""Access the active disjoint permutation-space branches."""

	permutationSpace: PermutationSpace = dataclasses.field(default_factory=PermutationSpace, init=True)
	"""Access the current permutation space on the workbench."""

	boxOfFunctionsReduction: Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]] = dataclasses.field(default_factory=list[Callable[['StateElimination', PermutationSpace], PermutationSpace]], init=True)

	groupsOfFolds: int = 0
	"""Inspect the representative count before symmetry multipliers."""

	boxOfFolding: list[Folding] = dataclasses.field(default_factory=list[Folding], init=True)
	"""Access the collected folding representatives."""
	pile: Pile = -1
	"""Inspect the current workbench pile index."""

	Theorem2aMultiplier: int = 1
	Theorem2Multiplier: int = 1
	Theorem3Multiplier: int = 1
	Theorem4Multiplier: int = 1

	totalDimensions: int = dataclasses.field(init=False)
	"""Inspect the dimension count derived during initialization."""
	leafLast: Leaf = dataclasses.field(init=False)
	"""Inspect the largest zero-based leaf index."""
	totalLeaves: int = dataclasses.field(init=False)
	"""Inspect the leaf count derived during initialization."""
	pileLast: Pile = dataclasses.field(init=False)
	"""Inspect the largest zero-based pile index."""
	pilesTotal: int = dataclasses.field(init=False)
	"""Inspect the pile count derived during initialization."""
	mapShapeProducts: tuple[int, ...] = dataclasses.field(init=False)
	"""Access cumulative dimension products, including the empty product."""
	mapShapeProductsSums: tuple[int, ...] = dataclasses.field(init=False)
	"""Access cumulative sums of dimension products, beginning with zero."""
	mapShape首ProductsSums: tuple[int, ...] = dataclasses.field(init=False)
	"""Access cumulative sums of reversed dimension strides."""
	首: int = dataclasses.field(init=False)
	"""Inspect the first out-of-bounds leaf index, equal to the total leaf count."""

	@property
	def totalFolds(self) -> int:
		"""Compute the total fold count from representatives and symmetry factors.

		You can multiply `self.groupsOfFolds` by the four stored symmetry multipliers without additional
		enumeration or validation.

		Returns
		-------
		totalFolds : int
			Representative count after applying every stored symmetry multiplier.
		"""
		return prod(
			(self.groupsOfFolds, self.Theorem2aMultiplier, self.Theorem2Multiplier, self.Theorem3Multiplier, self.Theorem4Multiplier)
		)

	def moveToBoxOfFolding(self) -> Self:
		"""Move fully assigned permutation spaces into the folding collection.

		(AI generated docstring)

		You can separate active spaces by pinned-leaf count and convert spaces with as many pinned entries
		as map leaves to foldings. You retain other spaces as active branches. Normalize leaf sets, check
		crossings, and update the representative count separately.

		Returns
		-------
		state : Self
			The same state with completed assignments removed from active spaces and appended to the folding
			collection.
		"""
		foldingGroup吗: dict[bool, list[PermutationSpace]] = toolz_groupby(
			compose(self.totalLeaves.__eq__, attrgetter('leafCount')), self.boxOfPermutationSpace
		)
		self.boxOfPermutationSpace = list(foldingGroup吗.get(False, ()))
		self.boxOfFolding.extend(map(methodcaller('makeFolding'), foldingGroup吗.get(True, ())))
		return self

	def permutationSpaceCreaseViolation吗(self, permutationSpace: PermutationSpace) -> bool:
		"""Detect crease crossings among a permutation space's pinned leaves.

		You can check `permutationSpace` for forbidden crossings in each map dimension. You compare a crease
		only when both endpoints are pinned and the forward endpoint has a nonzero pile index, skipping
		unresolved endpoints.

		Crease comparison
		-----------------
		You group crease endpoint pairs by coordinate parity and check matching-parity pairs with
		`mapFolding._e.algorithms.iff.creaseViolation吗` [1]. Normalize leaf sets and test other assignment
		constraints separately.

		Parameters
		----------
		permutationSpace : PermutationSpace
			Permutation space whose currently pinned assignments will be inspected.

		Returns
		-------
		hasViolation : bool
			`True` when a tested pair of pinned creases crosses.

		References
		----------
		[1] `mapFolding._e.algorithms.iff.creaseViolation吗`
		"""
		leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(permutationSpace.pinnedLeaves())}

		for dimension in range(self.totalDimensions):
			boxOfPileCreaseByParity: list[list[tuple[Pile, Pile]]] = [[], []]
			for pile, leaf in permutationSpace.pinnedLeaves().items():
				crease: int | None = getCreasePost(self.mapShape, leaf, dimension)
				if crease:
					pileCrease: int | None = leafToPile.get(crease)
					if pileCrease:
						boxOfPileCreaseByParity[oddLeaf吗(self.mapShape, leaf, dimension)].append((pile, pileCrease))
			for groupedParity in boxOfPileCreaseByParity:
				if any(creaseViolation吗(pile, pileComparand, pileCrease, pileComparandCrease)
					for (pile, pileCrease), (pileComparand, pileComparandCrease) in Y7Y_combinations(sorted(groupedParity), 2)):
					return True
		return False

	def pinAt_pile吗(self, leaf: Leaf) -> bool:
		"""Test whether a leaf can be considered for the current workbench pile.

		(AI generated docstring)

		You can check that `leaf` is not already pinned, that `self.pile` remains undetermined, and that
		`self.pile` belongs to the shape-based leaf domain [1]. Check membership in the pile's current leaf
		bitset separately.

		Workbench requirements
		----------------------
		The current pile must exist in `self.permutationSpace`. Checking the pile normalizes its assignment
		and can change the dictionary or validity flag.

		Parameters
		----------
		leaf : Leaf
			Leaf index to consider for the workbench pile.

		Returns
		-------
		canConsiderPinning : bool
			`True` when all three eligibility checks pass.

		References
		----------
		[1] `mapFolding._e.leafDomains.getDomainLeaf`
		"""
		return all((
			self.permutationSpace.leafNotPinned吗(leaf)
			, self.permutationSpace.pileUndetermined吗(self.pile)
			, self.pile in _e.getDomainLeaf(self, leaf)
		))

	def reduceAllPermutationSpace(self, boxOfFunctionsReduction: Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]] | None = None) -> Self:
		"""Apply reductions repeatedly to every valid active permutation space.

		(AI generated docstring)

		You can reduce the active search spaces and retain only spaces that remain valid after the reduction
		sequence stops changing their value sums. You may receive mutated spaces or replacement spaces from
		reducers.

		Reduction order
		---------------
		You apply functions in reverse sequence order and restart from the end whenever the sum of
		dictionary values changes. You stop reducing a space when a reducer sets its validity flag to
		`False`.

		Parameters
		----------
		boxOfFunctionsReduction : Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]] | None = None
			Reduction sequence to apply. A false-valued argument uses the state's preferred sequence, then the
			default sequence [1] if the preferred sequence is also false-valued.

		Returns
		-------
		state : Self
			The same state with active spaces replaced by the valid reduced results.

		References
		----------
		[1] `mapFolding._e.reduceIt.boxOfFunctionsReductionDEFAULT`
		"""
		boxOfFunctionsReduction = boxOfFunctionsReduction or self.boxOfFunctionsReduction or boxOfFunctionsReductionDEFAULT
		boxOfPermutationSpace: list[PermutationSpace] = 是valid(self.boxOfPermutationSpace)
		self.boxOfPermutationSpace = []
		boxOfPermutationSpaceIrreducible: list[PermutationSpace] = []

		functionsReduction: list[Callable[[StateElimination, PermutationSpace], PermutationSpace]] = list(boxOfFunctionsReduction)
		for permutationSpace in boxOfPermutationSpace:
			#------------ Initialize `permutationSpace` ------------------------------
			sumPermutationSpace: Leaf | ChoicesLeaf = sum(permutationSpace.values())
			次: int = len(functionsReduction)

			while 次:
				次 -= 1
				reducer: Callable[[StateElimination, PermutationSpace], PermutationSpace] = functionsReduction[次]
				permutationSpace: PermutationSpace = reducer(self, permutationSpace)

				if not permutationSpace.valid:
					次 = 0
				elif sumPermutationSpace != sum(permutationSpace.values()):
					次 = len(boxOfFunctionsReduction)
					sumPermutationSpace = sum(permutationSpace.values())
				elif 次 == 0:
					boxOfPermutationSpaceIrreducible.append(permutationSpace)

		else:
			self.boxOfPermutationSpace.extend(boxOfPermutationSpaceIrreducible)

		return self

	def removeCreaseViolations(self) -> Self:
		"""Remove active spaces with detected crossings between pinned creases.

		(AI generated docstring)

		You can filter the active spaces with `StateElimination.permutationSpaceCreaseViolation吗` [1]. Apply
		other validity checks or leaf-set reductions separately.

		Returns
		-------
		state : Self
			The same state with crease-crossing spaces removed from the active list.

		References
		----------
		[1] `StateElimination.permutationSpaceCreaseViolation吗`
		"""
		boxOfPermutationSpace: list[PermutationSpace] = self.boxOfPermutationSpace.copy()
		self.boxOfPermutationSpace = []
		self.boxOfPermutationSpace.extend(filterfalse(self.permutationSpaceCreaseViolation吗, boxOfPermutationSpace))

		return self

	def __post_init__(self) -> None:
		"""Initialize map-derived elimination values."""
		self.mapShape = validateMapShape(self.mapShape)
		self.totalDimensions = len(self.mapShape)
		self.totalLeaves = getTotalLeaves(self.mapShape)
		if 0 < self.totalLeaves:
			self.Theorem2aMultiplier = self.totalLeaves
		self.leafLast = self.totalLeaves - 1
		self.pilesTotal = self.totalLeaves
		self.pileLast = self.pilesTotal - 1
		self.首 = self.totalLeaves
		self.mapShapeProducts = _e.getMapShapeProducts(self.mapShape)
		self.mapShapeProductsSums = _e.getMapShapeProductsSums(self.mapShape)
		self.mapShape首ProductsSums = _e.getMapShape首ProductsSums(
			self.mapShapeProducts, self.totalDimensions, self.totalDimensions
		)
