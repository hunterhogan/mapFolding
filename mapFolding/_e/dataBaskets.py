# TODO idk enough to choose between `UserDict` and subclassing `dict`.
# ruff: file-ignore[subclass-builtin]
"""Use data baskets to easily move data, including values that affect computations: don't limit yourself to one data basket per algorithm."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from gmpy2 import bit_mask
from humpy_cytoolz import (
	assoc as associateKeyValue, compose, dissoc as dissociatePile, first, groupby as toolz_groupby, merge, valfilter as filterLeaf)
from hunterMakesPy import raiseIfNone
from itertools import combinations, filterfalse
from mapFolding import _e
from mapFolding._e.algorithms.iff import creaseViolation吗, getCreasePost, oddLeaf吗
from mapFolding._e.filters import choicesLeaf吗, leafInChoicesLeaf吗, leaf吗, 是valid
from mapFolding._e.reduceIt import boxOfFunctionsReductionDEFAULT
from mapFolding._e.theTypes import Folding, LeafSpace, Pile
from mapFolding.beDRY import getTotalLeaves, validateMapShape
from math import prod
from operator import attrgetter, methodcaller
from typing import cast, overload, TYPE_CHECKING
from Z0Z_tools import DOTitems, DOTkeys, DOTvalues
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding._e.theTypes import ChoicesLeaf, Leaf, PinnedLeaves, UndeterminedPiles
	from typing import Self

#=EndNotes##pinning=
class PermutationSpace(dict[Pile, LeafSpace]):
	"""Representation of `Pile: LeafSpace` for all `Pile` in `pilesTotal`, and methods to validly alter `PermutationSpace`."""

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
		return PermutationSpace(self)

	#============== New methods and attributes ====================================================

	def atPileExcludeLeaf(self, pile: Pile, leaf: Leaf) -> PermutationSpace:
		"""Exclude `leaf` from `pile` in this `PermutationSpace`.

		(AI generated docstring)

		Remove `leaf` from `ChoicesLeaf` at `pile` when `pile` contains a `ChoicesLeaf`, or if `leaf`
		is already pinned at `pile`, mark this `PermutationSpace` as invalid.

		Parameters
		----------
		pile : Pile
			Index of the pile whose candidate set will be modified. The method ensures a `LeafSpace`
			exists for this pile before attempting the exclusion.
		leaf : Leaf
			Leaf index to exclude from the pile's candidate set.

		Returns
		-------
		permutationSpace : PermutationSpace
			The same `PermutationSpace` instance (self), mutated in-place.

		Side effects
		------------
		- Mutates self by updating self[pile].
		- Calls self._solidifyLeafSpaceAtPile(pile) to normalize internal state after mutation.
		- Sets self.valid = False if attempting to exclude a leaf that is already pinned at pile.
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
		"""DANGEROUSLY create a new `PermutationSpace` with `leaf` pinned at `pile` without modifying `permutationSpace`.

		Danger: Corrupted `PermutationSpace`
		------------------------------------
		If you overwrite a different `Leaf` pinned at `pile`, it will corrupt the `PermutationSpace`.
		If `leaf` is already pinned at a different `Pile`, but you pin `leaf` at `pile`, it will
		corrupt the `PermutationSpace`.

		Nevertheless, this method _assumes_ either 1. a. `leaf` is not pinned and b. `pile` is open or
		2. `leaf` is already pinned at `pile`.

		Danger: Corrupted Collection of `PermutationSpace`
		--------------------------------------------------
		If any `PermutationSpace` in your collection overlaps with any other `PermutationSpace` in
		your collection, it will corrupt your collection. This method creates a new `PermutationSpace`
		that _almost completely overlaps_ the original `PermutationSpace`. Ensure your logic never
		puts both versions in your collection.

		Parameters
		----------
		pile : int
			`pile` at which to pin `leaf`.
		leaf : int
			`leaf` to pin.

		Returns
		-------
		dictionaryPermutationSpace : PermutationSpace
			New dictionary with `pile` mapped to `leaf`.

		Example
		-------
		Overwriting the original `PermutationSpace` avoids corruption.
		```python
			ImaPermutationSpace = ImaPermutationSpace.atPilePinLeaf(pile, leaf)
		```
		"""
		return PermutationSpace(associateKeyValue(self, pile, leaf, PermutationSpace))

	# TODO reconsider the role, necessity, and location of this function.
	def atPilePinLeafSafetyFilter(self, pile: Pile, leaf: Leaf) -> bool:
		"""Return `True` if it is safe to call `permutationSpace.atPilePinLeaf(pile, leaf)`.

		For performance, you probably can and probably *should* create a set of filters for your
		circumstances.

		Parameters
		----------
		pile : int
			`pile` at which to pin.
		leaf : int
			`leaf` to pin.

		Returns
		-------
		isSafeToPin : bool
			True if it is safe to pin `leaf` at `pile` in `permutationSpace`.
		"""
		return self.leafPinnedAtPile吗(leaf, pile) or (self.pileUndetermined吗(pile) and self.leafNotPinned吗(leaf))

	def bifurcate(self) -> tuple[PinnedLeaves, UndeterminedPiles]:
		"""Split a `PermutationSpace` into `PinnedLeaves` and `UndeterminedPiles`.

		Returns
		-------
		leavesPinned : PinnedLeaves
			Dictionary of `Pile` to pinned `Leaf` mappings.
		pilesUndetermined : UndeterminedPiles
			Dictionary of `Pile` to `ChoicesLeaf` domain mappings.
		"""
		leavesPinned: PinnedLeaves = self.pinnedLeaves()
		#=SIN= `cast`: type checkers cannot infer that partitioning `PermutationSpace` preserves `UndeterminedPiles`.
		return (leavesPinned, cast('UndeterminedPiles', dissociatePile(self, *DOTkeys(leavesPinned))))

	def deconstructDomainOfLeaf(self, leaf: Leaf, leafDomain: Iterable[Pile]) -> list[PermutationSpace]:
		"""Pin `leaf` at each open `pile` in the domain of `leaf`.

		Return a `list` containing this `PermutationSpace` if `leaf` is already pinned, or one
		`PermutationSpace` for each open `pile` in `leafDomain` with `leaf` pinned at `pile`.

		Parameters
		----------
		leaf : int
			`leaf` to pin.
		leafDomain : Iterable[int]
			Domain of `pile` indices for `leaf`.

		Returns
		-------
		deconstructedPermutationSpace : list[PermutationSpace]
			List of `PermutationSpace` dictionaries with `leaf` pinned at each open `pile` in
			`leafDomain`.
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
		"""Pin several leaves across matching pile-domain tuples.

		Parameters
		----------
		leaves : Sequence[int]
			Leaves to pin.
		leavesDomain : Iterable[Sequence[int]]
			Candidate pile tuples whose positions correspond to `leaves`.

		Returns
		-------
		deconstructedPermutationSpace : list[PermutationSpace]
			List of `PermutationSpace` dictionaries with the requested leaves pinned across
			compatible pile tuples.
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
				For each leaf in leaves, I filter out occupied piles, so I will not overwrite any pinned leaves--that would invalidate the permutation space.
				I apply filters that prevent pinning the same leaf twice.
				Therefore, for each domain in `leavesDomain`, I can safely pin `leaves[次]` at `boxOfPiles[次]` without corrupting the permutation space."""
				permutationSpace: PermutationSpace = self.copy()
				for 次 in range(len(leaves)):
					permutationSpace = permutationSpace.atPilePinLeaf(boxOfPiles[次], leaves[次])
				deconstructedPermutationSpace.append(permutationSpace)
		else:
			deconstructedPermutationSpace.append(self)

		return deconstructedPermutationSpace

	def deconstructPile(self, pile: Pile | None = None, leavesToPin: Iterable[Leaf] = ()) -> Iterable[PermutationSpace]:
		"""Create alternative `PermutationSpace` branches from `leavesToPin` candidates at `pile`.

		(AI generated docstring)

		You can use this method to replace one `PermutationSpace` with alternatives that pin one
		candidate `Leaf` at `pile`. When `pile` is `None`, this method selects the first pile whose
		value is a `ChoicesLeaf`. When `leavesToPin` is false-valued, this method uses every `Leaf`
		represented by the selected `ChoicesLeaf`. Candidates already pinned in `self` are omitted.

		If `pile` does not contain a `ChoicesLeaf`, this method returns an iterable containing `self`
		without copying `self`.

		Parameters
		----------
		pile : Pile | None = None
			`Pile` at which to pin each candidate `Leaf`. A value of `None` selects the first pile
			whose value is a `ChoicesLeaf`.
		leavesToPin : Iterable[Leaf] = ()
			Candidate `Leaf` values. A false-valued `leavesToPin` selects every `Leaf` represented by
			the selected `ChoicesLeaf`.

		Returns
		-------
		deconstructed : Iterable[PermutationSpace]
			Iterable that yields one new `PermutationSpace` for each candidate `Leaf` that is not
			already pinned, or yields `self` once when `pile` does not contain a `ChoicesLeaf`.

		Collection Integrity
		--------------------
		Do not retain `self` in the same collection as any returned `PermutationSpace` that you
		consume. Each new branch overlaps with `self`, and the fallback result contains the exact
		`self` object rather than a copy.
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
		"""Retrieve a pinned `Leaf` from `permutationSpace` at `pile`, or return a default value.

		Parameters
		----------
		pile : Pile
			`Pile` index to look up in `permutationSpace`.
		default : Leaf | None = None
			Value to return when `permutationSpace[pile]` is not a `Leaf`.

		Returns
		-------
		leafOrDefault : Leaf | None
			The `Leaf` at `permutationSpace[pile]` if `permutationSpace[pile]` is a `Leaf`, otherwise
			`default`.
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
		"""Read `permutationSpace[pile]` only when `permutationSpace[pile]` is a `ChoicesLeaf`.

		Parameters
		----------
		pile : Pile
			`Pile` index to look up in `permutationSpace`.
		default : ChoicesLeaf | None = None
			Value to return when `permutationSpace[pile]` is not a `ChoicesLeaf`.

		Returns
		-------
		choicesLeafOrNone : ChoicesLeaf | None
			`ChoicesLeaf` value from `permutationSpace[pile]`, or `default`.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		ImaChoicesLeaf: LeafSpace = self[pile]
		if choicesLeaf吗(ImaChoicesLeaf):
			return ImaChoicesLeaf
		return default

	@property
	def leafCount(self) -> int:
		"""Count of `Leaf` indices that are pinned in this `PermutationSpace`.

		Returns
		-------
		leafCount : int
			Count of `Leaf` indices that are pinned in this `PermutationSpace`.
		"""
		return sum(map(leaf吗, self.values()))

	def leafNotPinned吗(self, leaf: Leaf) -> bool:
		"""Return `True` if `leaf` is not presently pinned in this `PermutationSpace`.

		Parameters
		----------
		leaf : Leaf
			`Leaf` index.

		Returns
		-------
		leafIsNotPinned : bool
			`True` if this `PermutationSpace` does not include `leaf`.
		"""
		return leaf not in self.values()

	def leafPinned吗(self, leaf: Leaf) -> bool:
		"""Return `True` if `leaf` is pinned in this `PermutationSpace`.

		Parameters
		----------
		leaf : Leaf
			`Leaf` index.

		Returns
		-------
		leafIsPinned : bool
			`True` if this `PermutationSpace` includes `leaf`.
		"""
		return leaf in self.values()

	def leafPinnedAtPile吗(self, leaf: Leaf, pile: Pile) -> bool:
		"""Return `True` if `leaf` is pinned at `pile` in this `PermutationSpace`.

		Parameters
		----------
		leaf : Leaf
			`Leaf` whose presence at `pile` is being checked.
		pile : Pile
			`Pile` index.

		Returns
		-------
		leafIsPinnedAtPile : bool
			`True` if this `PermutationSpace` includes `pile: leaf`.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		return leaf == self[pile]

	# TODO Consider implementing another method to make a `Folding` or _maybe_ cleverly overloading
	# this method (I'm deeply skeptical that overload is a good idea). `makeFolding` handles _my_
	# current needs. If I had to create ONE `makeFolding` function/method with the most utility,
	# however, it would NOT look like this function. 2026 July 10: off the top of my head, passing
	# `boxOfPileLeaf: Sequence[tuple[Pile, Leaf]]` would be better than the current function and is
	# probably close to the ideal generalized function.
	def makeFolding(self, leavesToInsert: Sequence[Leaf] = ()) -> Folding:
		"""Complete this `PermutationSpace` as a `Folding`.

		(AI generated docstring)

		This method pairs each item in `leavesToInsert` with an undetermined `Pile`. The first item
		corresponds to the smallest undetermined `Pile`, the second item to the next-smallest
		undetermined `Pile`, and so on. Existing pinned `Leaf` values keep their pile positions.

		Parameters
		----------
		leavesToInsert : Sequence[Leaf]
			One `Leaf` for each undetermined `Pile`, ordered by ascending `Pile`.

		Returns
		-------
		folding : Folding
			Every pinned or inserted `Leaf`, ordered by ascending `Pile`.
		"""
		pilesToInsert: Iterator[Pile] = DOTkeys(self.undeterminedPiles())
		#=SIN= `cast` because the type checkers cannot possible know that the prior logic leads to all int.
		# TODO Think about: I _feel_ like this logic could be more efficient. This
		# `tuple(DOTvalues(dict(sorted(DOTitems` has THREE constructors (`sorted` is a stealth `list`
		# constructor) or FIVE constructors if `Iterator` is a constructor (`DOTitems` and
		# `DOTvalues`), so I _feel_ it would be faster if I could change the values without
		# ping-ponging from `dict` to `list` to `dict` to `tuple`.
		return tuple(DOTvalues(dict(sorted(DOTitems(cast('PinnedLeaves', merge(self, dict(zip(pilesToInsert, leavesToInsert, strict=True)), factory=PermutationSpace)))))))

	def pilePinned吗(self, pile: Pile) -> bool:
		"""Determine whether `pile` has a pinned `Leaf`.

		Use this method when control flow concerns the assignment state of a `Pile` in this
		`PermutationSpace`. Use `isLeaf吗` when the logic already has a `LeafSpace` value and needs
		Python type narrowing.

		Parameters
		----------
		pile : Pile
			`Pile` index.

		Returns
		-------
		pileIsPinned : bool
			`True` if this `PermutationSpace` contains a `Leaf` at `pile`.

		See Also
		--------
		`pileUndetermined吗`
			Determine whether a `Pile` still requires a `Leaf` assignment.
		`mapFolding._e.filters.isLeaf吗`
			Narrow an existing `LeafSpace` value to `Leaf`.
		`mapFolding._e.filters.isChoicesLeaf吗`
			Narrow an existing `LeafSpace` value to `ChoicesLeaf`.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		return leaf吗(self[pile])

	def pileUndetermined吗(self, pile: Pile) -> bool:
		"""Determine whether `pile` still requires a `Leaf` assignment.

		Use this method when control flow concerns whether a `Pile` still requires a `Leaf`
		assignment. Use `isChoicesLeaf吗` when the logic already has a `LeafSpace` value and needs
		Python type narrowing.

		Parameters
		----------
		pile : Pile
			`Pile` index.

		Returns
		-------
		pileIsUndetermined : bool
			`True` if this `PermutationSpace` contains `ChoicesLeaf` at `pile`.

		See Also
		--------
		`pilePinned吗`
			Determine whether a `Pile` already has a pinned `Leaf`.
		`mapFolding._e.filters.isChoicesLeaf吗`
			Narrow an existing `LeafSpace` value to `ChoicesLeaf`.
		`mapFolding._e.filters.isLeaf吗`
			Narrow an existing `LeafSpace` value to `Leaf`.
		"""
		self._solidifyLeafSpaceAtPile(pile)
		return choicesLeaf吗(self[pile])

	def _solidifyLeafSpace(self) -> None:
		count: int = self.leafCount
		if count < len(self):
			tuple(map(self._solidifyLeafSpaceAtPile, self))
			leavesPinned: PinnedLeaves = self.pinnedLeaves()
			antiChoicesLeaf: ChoicesLeaf = _e.makeAntiChoicesLeaf(len(self), leavesPinned.values())
			for pile in filterfalse[Pile](leavesPinned.__contains__, self):
				self[pile] &= antiChoicesLeaf
			tuple(map(self._solidifyLeafSpaceAtPile, leavesPinned))
			if count < self.leafCount:
				self._solidifyLeafSpace()

	def _solidifyLeafSpaceAtPile(self, pile: Pile) -> None:
		rangeOfPile: LeafSpace | None = self[pile]
		if choicesLeaf吗(rangeOfPile):
			# If the range size of `pile` is 0 or 1, convert to None or `Leaf`.
			rangeOfPile = _e.choicesLeafLeafNone(rangeOfPile)
			if choicesLeaf吗(rangeOfPile):
				self[pile] = rangeOfPile
			elif rangeOfPile is None:
				self.valid = False

	# TODO Does it matter whether this is a property or a method?
	def pinnedLeaves(self) -> PinnedLeaves:
		"""Create a dictionary *unsorted* by `pile` of only `pile: leaf` without `pile: choicesLeaf`.

		Returns
		-------
		dictionaryOfPileLeaf : dict[int, int]
			Dictionary of `pile` with pinned `leaf`, if a `leaf` is pinned at `pile`.
		"""
		return filterLeaf(leaf吗, self)

	def undeterminedPiles(self) -> UndeterminedPiles:
		"""Create a dictionary *unsorted* by `pile` of all `pile: choicesLeaf` in `PermutationSpace`.

		Returns
		-------
		pilesUndetermined : dict[int, ChoicesLeaf]
			Dictionary of `pile: choicesLeaf`, if a `choicesLeaf` is defined at `pile`.
		"""
		return filterLeaf(choicesLeaf吗, self)

	def updatePilesMissing(self, missing: PermutationSpace | UndeterminedPiles | PinnedLeaves) -> PermutationSpace:
		"""Update missing `Pile: LeafSpace` items with the items from `missing`.

		This will not overwrite any existing `Pile: LeafSpace` items in `permutationSpace` because
		that would corrupt the `PermutationSpace`.

		Parameters
		----------
		missing : PermutationSpace | UndeterminedPiles | PinnedLeaves
			`Pile: LeafSpace` in `missing`.

		Returns
		-------
		permutationSpace : PermutationSpace
			New `PermutationSpace` and modifies `PermutationSpace` in place.
		"""
		#=EndNotes##sorted=
		self = PermutationSpace(sorted(DOTitems(merge(missing, self))))
		#=Wrong= It is necessary to assign to self before returning.
		return self  # ruff: ignore[unnecessary-assign]

	valid: bool = True

@dataclasses.dataclass(slots=True)
class StateElimination:
	"""Computational state for algorithms that compute `totalFolds` by elimination.

	This data basket stores both mutable workbench fields (which change during the search) and
	precomputed constants derived from `mapShape` (which do not change after `__post_init__`).

	Attributes
	----------
	mapShape : tuple[int, ...]
		Dimension lengths of the map being analyzed.
	groupsOfFolds : int = 0
		Count of distinct `Folding` pattern groups found so far.
	boxOfFolding : list[`Folding`]
		List of `Folding` patterns found.
	boxOfPermutationSpace : list[`PermutationSpace`]
		List of exclusive `PermutationSpace` dictionaries.
	pile : `Pile` = -1
		The current `pile` on the workbench.
	permutationSpace : `PermutationSpace`
		The current `PermutationSpace` dictionary on the workbench.
	Theorem2Multiplier : int = 1
		Multiplier applied by Theorem 2 optimizations.
	Theorem3Multiplier : int = 1
		Multiplier applied by Theorem 3 optimizations.
	Theorem4Multiplier : int = 1
		Multiplier applied by Theorem 4 optimizations.
	totalDimensions : int
		Unchanging total number of axes in `mapShape`.
	foldingCheckSum : int
		Unchanging triangular-number check-sum for a valid `Folding`.
	leafLast : `Leaf`
		Unchanging 0-indexed largest `leaf` value.
	totalLeaves : int
		Unchanging total number of leaves in the map.
	pileLast : `Pile`
		Unchanging 0-indexed largest `pile` value.
	pilesTotal : int
		Unchanging total number of piles in the map.
	mapShapeProducts : tuple[int, ...]
		Unchanging products of dimension lengths, from the empty product through all dimensions.
	mapShapeProductsSums : tuple[int, ...]
		Unchanging sums of `mapShapeProducts` from the head.
	mapShape首ProductsSums : tuple[int, ...]
		Unchanging sums of `mapShapeProducts` from the head `首`.
	首 : int
		Unchanging single-base positional-numeral value of the first out-of-bounds Cartesian
		coordinate.

	Notes
	-----
	The computed `totalFolds` is `groupsOfFolds * totalLeaves * Theorem2Multiplier *
	Theorem3Multiplier * Theorem4Multiplier`.

	"""

	mapShape: tuple[int, ...] = dataclasses.field(init=True)
	"""Dimensions of the map being analyzed for folding patterns."""

	boxOfPermutationSpace: list[PermutationSpace] = dataclasses.field(default_factory=list[PermutationSpace], init=True)
	"""A list of dictionaries (`{pile: leaf or possible leaves}`) that each define an exclusive permutation space: no overlap between dictionaries."""

	permutationSpace: PermutationSpace = dataclasses.field(default_factory=PermutationSpace, init=True)
	"""The `permutationSpace` dictionary (`{pile: leaf or possible leaves}`) on the workbench."""

	boxOfFunctionsReduction: Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]] = dataclasses.field(default_factory=list[Callable[['StateElimination', PermutationSpace], PermutationSpace]], init=True)

	groupsOfFolds: int = 0
	"""`totalFolds` is divisible by `totalLeaves`; the algorithm counts each `Folding` that represents a group of `totalLeaves`-many foldings."""

	boxOfFolding: list[Folding] = dataclasses.field(default_factory=list[Folding], init=True)
	"""A list of `Folding` patterns found."""
	pile: Pile = -1
	"""The `pile` on the workbench."""

	Theorem2aMultiplier: int = 1
	Theorem2Multiplier: int = 1
	Theorem3Multiplier: int = 1
	Theorem4Multiplier: int = 1

	totalDimensions: int = dataclasses.field(init=False)
	"""Unchanging total number of dimensions in the map."""
	foldingCheckSum: int = dataclasses.field(init=False)
	"""Unchanging triangular number check-sum for a valid `Folding`, https://en.wikipedia.org/wiki/Triangular_number."""
	leafLast: Leaf = dataclasses.field(init=False)
	"""Unchanging 0-indexed largest `leaf` in a `Folding`."""
	totalLeaves: int = dataclasses.field(init=False)
	"""Unchanging total number of leaves in the map."""
	pileLast: Pile = dataclasses.field(init=False)
	"""Unchanging 0-indexed final `pile` in a `Folding`."""
	pilesTotal: int = dataclasses.field(init=False)
	"""Unchanging total number of piles in the map."""
	mapShapeProducts: tuple[int, ...] = dataclasses.field(init=False)
	"""Unchanging list of products of map dimensions from the product of no dimensions, `[0]`, to the product of all dimensions, `[totalDimensions + inclusive]`."""
	mapShapeProductsSums: tuple[int, ...] = dataclasses.field(init=False)
	"""Unchanging list of sums of products of map dimensions from the sum of no products, `[0]`, to the sum of all products, `[len(mapShapeProducts) + inclusive]`."""
	mapShape首ProductsSums: tuple[int, ...] = dataclasses.field(init=False)
	"""Unchanging list of sums of products of map dimensions starting from the head `首`, from the sum of no products, `[0]`, to the sum of all products, `[len(mapShapeProducts) + inclusive]`."""
	首: int = dataclasses.field(init=False)
	"""Unchanging single-base positional-numeral value of the Cartesian coordinates that are the first to be _out-of-bounds_ for the `mapShape`."""

	@property
	def totalFolds(self) -> int:
		"""The computed number of distinct `Folding` patterns for this `mapShape`."""
		return prod(
			(self.groupsOfFolds, self.Theorem2aMultiplier, self.Theorem2Multiplier, self.Theorem3Multiplier, self.Theorem4Multiplier)
		)

	def moveToBoxOfFolding(self) -> Self:
		foldingGroup吗: dict[bool, list[PermutationSpace]] = toolz_groupby(
			compose(self.totalLeaves.__eq__, attrgetter('leafCount')), self.boxOfPermutationSpace
		)
		self.boxOfPermutationSpace = list(foldingGroup吗.get(False, ()))
		self.boxOfFolding.extend(map(methodcaller('makeFolding'), foldingGroup吗.get(True, ())))
		return self

	def permutationSpaceCreaseViolation吗(self, permutationSpace: PermutationSpace) -> bool:
		"""You can detect forbidden crease crossings inside `state.permutationSpace`.

		`permutationSpaceCreaseViolation吗` is a pruning predicate used before counting or expanding a
		candidate `PermutationSpace`. `removeCreaseViolationsFromEliminationState` uses
		`permutationSpaceCreaseViolation吗` to filter `state.boxOfPermutationSpace` [5], and a caller
		such as `mapFolding._e.pin2上nDimensions` uses `removeCreaseViolationsFromEliminationState`
		[6] as part of building a reduced search space.

		Algorithm Details
		-----------------
		`permutationSpaceCreaseViolation吗` interprets `state.permutationSpace` as a partial mapping
		from `Pile` to `Leaf`. The pinned leaves extracted by `PermutationSpace.extractPinnedLeaves`
		[1] are inverted to a `Leaf`-to-`Pile` mapping so crease-post leaves can be looked up by
		`Leaf` index.

		`permutationSpaceCreaseViolation吗` filters candidate assignments with `between` [2] to skip
		leaves that cannot have a crease-post leaf in a selected dimension.

		For each `dimension`, `permutationSpaceCreaseViolation吗`:

		- enumerates each `(pile, leaf)` assignment that can have a crease-post leaf,
		- derives the crease-post leaf using `getCreasePost` [4],
		- looks up the crease-post leaf pile using pinned assignments,
		- groups crease pairs by parity using `ImaOddLeaf`,
		- checks each pair of crease pairs with `creaseViolation吗` [3].

		Parameters
		----------
		permutationSpace : PermutationSpace
			A permutation space that provides `permutationSpace.extractPinnedLeaves()` and bounds
			such as `permutationSpace.leafLast`.

		Returns
		-------
		hasViolation : bool
			`True` when at least one forbidden crease crossing is detected.

		References
		----------
		[1] mapFolding._e.dataBaskets.PermutationSpace.extractPinnedLeaves

		[2] mapFolding._e.filters.between

		[3] mapFolding._e.algorithms.iff.creaseViolation吗

		[4] mapFolding._e.algorithms.iff.getCreasePost

		[5] mapFolding._e.algorithms.iff.removeCreaseViolationsFromEliminationState

		[6] mapFolding._e.pin2上nDimensions
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
					for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2)):
					return True
		return False

	def pinAt_pile吗(self, leaf: Leaf) -> bool:
		return all((
			self.permutationSpace.leafNotPinned吗(leaf)
			, self.permutationSpace.pileUndetermined吗(self.pile)
			, self.pile in _e.getDomainLeaf(self, leaf)
		))

	def reduceAllPermutationSpace(self, boxOfFunctionsReduction: Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]] | None = None) -> Self:
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
		"""You can filter `state.boxOfPermutationSpace` by removing crease-crossing candidates.

		(AI generated docstring)

		`removePermutationSpaceViolations` is a mutating filter step that keeps only those
		`PermutationSpace` values that satisfy `permutationSpaceHasIFFViolation(self) == False` [1].
		This function is used by pinning flows that enumerate multiple candidate permutation spaces
		and then prune candidate permutation spaces before deeper elimination work. A caller such as
		`mapFolding._e.pin2上nDimensions` uses this function [2].

		Parameters
		----------
		self : Self
			The instance of the class.

		Returns
		-------
		self : Self
			The same instance with `self.boxOfPermutationSpace` filtered.

		References
		----------
		[1] mapFolding._e.algorithms.iff.permutationSpaceHasIFFViolation

		[2] mapFolding._e.pin2上nDimensions
		"""
		boxOfPermutationSpace: list[PermutationSpace] = self.boxOfPermutationSpace.copy()
		self.boxOfPermutationSpace = []
		self.boxOfPermutationSpace.extend(filterfalse(self.permutationSpaceCreaseViolation吗, boxOfPermutationSpace))

		return self

	def __post_init__(self) -> None:
		"""One-time computation of unchanging values."""
		self.mapShape = validateMapShape(self.mapShape)
		self.totalDimensions = len(self.mapShape)
		self.totalLeaves = getTotalLeaves(self.mapShape)
		if 0 < self.totalLeaves:
			self.Theorem2aMultiplier = self.totalLeaves
		self.leafLast = self.totalLeaves - 1
		self.foldingCheckSum = self.leafLast * self.totalLeaves // 2
		self.pilesTotal = self.totalLeaves
		self.pileLast = self.pilesTotal - 1
		self.首 = self.totalLeaves
		self.mapShapeProducts = _e.getMapShapeProducts(self.mapShape)
		self.mapShapeProductsSums = _e.getMapShapeProductsSums(self.mapShape)
		self.mapShape首ProductsSums = _e.getMapShape首ProductsSums(
			self.mapShapeProducts, self.totalDimensions, self.totalDimensions
		)
