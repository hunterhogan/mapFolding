# TODO idk enough to choose between `UserDict` and subclassing `dict`.
# ruff: noqa: FURB189
"""Use data baskets to easily move data, including values that affect computations: don't limit yourself to one data basket per algorithm."""
from __future__ import annotations

from collections import deque
# TODO `partial` vs `humpy_cytoolz.functoolz.curry`: which is better?
from functools import partial
from gmpy2 import bit_mask
# SEMIOTICS `associate`, `associateItem`, or something else?
from humpy_cytoolz import assoc as associate, compose, dissoc as dissociatePile, merge, valfilter as filterLeaf, valmap as mapLeaf
from hunterMakesPy import raiseIfNone
from mapFolding._e import getProductsOfDimensions, getSumsOfProductsOfDimensions, getSumsOfProductsOfDimensionsNearest首, JeanValjean
from mapFolding._e.filters import isLeafOptions吗, isLeaf吗, leafInLeafOptions吗
from mapFolding._e.theTypes import Folding, LeafSpace, Pile, UndeterminedPiles
from mapFolding.beDRY import getLeavesTotal
from mapFolding.genericNeedsNewHome import DOTitems, DOTkeys, DOTvalues
from math import prod
from typing import cast, TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding._e.theTypes import Leaf, LeafOptions, PinnedLeaves

# TODO Probably create a `Property` to report the number of `Leaf` objects. Use case example: `sum首:
# int = sum(map(dimensionNearest首, permutationSpace.values()))`.
class PermutationSpace(dict[Pile, LeafSpace]):
	"""Representation of `Pile: LeafSpace` for all `Pile` in `pilesTotal`, and methods to validly alter `PermutationSpace`."""

	def addMissingLeafOptions(self, dictionaryLeafOptions: UndeterminedPiles) -> PermutationSpace:
		"""Return a new `PermutationSpace` with default NO! `LeafOptions` for missing piles.

		TODO "default" is not the right word.

		Parameters
		----------
		dictionaryLeafOptions : UndeterminedPiles
			Default `LeafOptions` by `Pile`.

		Returns
		-------
		permutationSpace : PermutationSpace
			New `PermutationSpace` with current entries overriding default `LeafOptions`.
		"""
		# NOTE `sorted` overrides the insertion order and sorts based on `Pile` index. This is
		# partially "defensive" in the sense that it is a consistent, logical, expected order, and may
		# prevent odd results if another subroutine didn't guarantee the order when it ought to have.
		# I'm hoping it improves efficiency, too.
		return PermutationSpace(sorted(DOTitems(merge(mapLeaf(compose(raiseIfNone, JeanValjean), dictionaryLeafOptions), self, factory=PermutationSpace))))

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
		return PermutationSpace(associate(self, pile, leaf, PermutationSpace))

	# TODO reconsider the role, necessity, and location of this function.
	def atPilePinLeafSafetyFilter(self, pile: Pile, leaf: Leaf) -> bool:
		"""Return `True` if it is safe to call `permutationSpace.atPilePinLeaf(pile, leaf)`.

		For performance, you probably can and probably *should* create a set of filters for your circumstances.

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
		return self.leafPinnedAtPile吗(leaf, pile) or (self.pileOpen吗(pile) and self.leafNotPinned吗(leaf))

	def bifurcate(self) -> tuple[PinnedLeaves, UndeterminedPiles]:
		"""Split a `PermutationSpace` into `PinnedLeaves` and `UndeterminedPiles`.

		Returns
		-------
		leavesPinned : PinnedLeaves
			Dictionary of `Pile` to pinned `Leaf` mappings.
		pilesUndetermined : UndeterminedPiles
			Dictionary of `Pile` to `LeafOptions` domain mappings.
		"""
		leavesPinned: PinnedLeaves = self.extractPinnedLeaves()
		# TODO Create new comment marker to signal "deviations" from the code style, or code that doesn't "conform" to the rules.
		# NOTE `cast` because type checkers don't know `PermutationSpace` - `PinnedLeaves` = `UndeterminedPiles`.
		return (leavesPinned, cast("UndeterminedPiles", dissociatePile(self, *DOTkeys(leavesPinned))))

	def copy(self) -> PermutationSpace:
		return PermutationSpace(self)

	def deconstructAtPile(self, pile: Pile, leavesToPin: Iterable[Leaf]) -> dict[Leaf, PermutationSpace]:
		"""Deconstruct an open `pile` to the `leaf` range of `pile`.

		Return a dictionary containing this `PermutationSpace` if `pile` already has a
		pinned `leaf`, or one `PermutationSpace` for each unpinned `leaf` in
		`leavesToPin` pinned at `pile`.

		Parameters
		----------
		pile : int
			`pile` at which to pin a `leaf`.
		leavesToPin : list[int]
			List of `leaves` to pin at `pile`.

		Returns
		-------
		deconstructedPermutationSpace : dict[int, PermutationSpace]
			Dictionary mapping from `leaf` pinned at `pile` to the `PermutationSpace`
			dictionary with the `leaf` pinned at `pile`.
		"""
		if (leaf := self.getLeaf(pile)) is not None:
			deconstructedPermutationSpace: dict[Leaf, PermutationSpace] = {leaf: self}
		else:
			pin: Callable[[Leaf], PermutationSpace] = partial(self.atPilePinLeaf, pile)
			deconstructedPermutationSpace = {leaf: pin(leaf) for leaf in filter(self.leafNotPinned吗, leavesToPin)}
		return deconstructedPermutationSpace

	def deconstructByDomainOfLeaf(self, leaf: Leaf, leafDomain: Iterable[Pile]) -> deque[PermutationSpace]:
		"""Pin `leaf` at each open `pile` in the domain of `leaf`.

		Return a `deque` containing this `PermutationSpace` if `leaf` is already
		pinned, or one `PermutationSpace` for each open `pile` in `leafDomain` with
		`leaf` pinned at `pile`.

		Parameters
		----------
		leaf : int
			`leaf` to pin.
		leafDomain : Iterable[int]
			Domain of `pile` indices for `leaf`.

		Returns
		-------
		deconstructedPermutationSpace : deque[PermutationSpace]
			Deque of `PermutationSpace` dictionaries with `leaf` pinned at each open
			`pile` in `leafDomain`.
		"""
		deconstructedPermutationSpace: deque[PermutationSpace] = deque()
		if self.leafNotPinned吗(leaf):
			leafInPileRange: Callable[[int], bool] = compose(leafInLeafOptions吗(leaf), partial(self.getLeafOptions, default=bit_mask(len(self))))
			pinLeafAt: Callable[[int], PermutationSpace] = partial(self.atPilePinLeaf, leaf=leaf)
			deconstructedPermutationSpace.extend(map(pinLeafAt, filter(leafInPileRange, filter(self.pileOpen吗, leafDomain))))
		else:
			deconstructedPermutationSpace.append(self)
		return deconstructedPermutationSpace

	def deconstructByDomainsCombined(self, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]]) -> deque[PermutationSpace]:
		"""Pin several leaves across matching pile-domain tuples.

		Parameters
		----------
		leaves : Sequence[int]
			Leaves to pin.
		leavesDomain : Iterable[Sequence[int]]
			Candidate pile tuples whose positions correspond to `leaves`.

		Returns
		-------
		deconstructedPermutationSpace : deque[PermutationSpace]
			Deque of `PermutationSpace` dictionaries with the requested leaves pinned
			across compatible pile tuples.
		"""
		deconstructedPermutationSpace: deque[PermutationSpace] = deque()

		def pileOpenByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:
			def workhorse(domain: Sequence[Pile]) -> bool:
				return self.pileOpen吗(domain[index])
			return workhorse

		def leafInPileRangeByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:
			def workhorse(domain: Sequence[Pile]) -> bool:
				leafOptions: LeafOptions = raiseIfNone(self.getLeafOptions(domain[index], default=bit_mask(len(self))))
				return leafInLeafOptions吗(leaves[index], leafOptions)
			return workhorse

		def isPinnedAtPileByIndex(leaf: Leaf, index: int) -> CallableFunction[[Sequence[Pile]], bool]:
			def workhorse(domain: Sequence[Pile]) -> bool:
				return self.leafPinnedAtPile吗(leaf, domain[index])
			return workhorse

		if any(map(self.leafNotPinned吗, leaves)):
			for index in range(len(leaves)):
				"""Redefine leavesDomain by filtering out domains that are not possible with the current `PermutationSpace`."""
				if self.leafNotPinned吗(leaves[index]):
					"""`leaves[index]` is not pinned, so it needs a pile.
					In each iteration of `leavesDomain`, `listOfPiles`, the pile it needs is `listOfPiles[index]`.
					Therefore, if `listOfPiles[index]` is open, filter in the iteration. If `listOfPiles[index]` is occupied, filter out the iteration."""
					leavesDomain = filter(pileOpenByIndex(index), leavesDomain)
					"""`leaves[index]` is not pinned, it wants `listOfPiles[index]`, and `listOfPiles[index]` is open.
					Is `leaves[index]` in the pile-range of `listOfPiles[index]`?"""
					leavesDomain = filter(leafInPileRangeByIndex(index), leavesDomain)
				else:
					"""`leaves[index]` is pinned.
					In each iteration of `leavesDomain`, `listOfPiles`, the pile in which `leaves[index]` is pinned must match `listOfPiles[index]`.
					Therefore, if the pile in which `leaves[index]` is pinned matches `listOfPiles[index]`, filter in the iteration. Otherwise, filter out the iteration."""
					leavesDomain = filter(isPinnedAtPileByIndex(leaves[index], index), leavesDomain)

			for listOfPiles in leavesDomain:
				"""Properly and safely deconstruct `permutationSpace` by the combined domain of leaves.
				The parameter `leavesDomain` is the full domain of the leaves, so deconstructing with `leavesDomain` preserves the permutation space.
				For each leaf in leaves, I filter out occupied piles, so I will not overwrite any pinned leaves--that would invalidate the permutation space.
				I apply filters that prevent pinning the same leaf twice.
				Therefore, for each domain in `leavesDomain`, I can safely pin `leaves[index]` at `listOfPiles[index]` without corrupting the permutation space."""
				permutationSpaceForListOfPiles: PermutationSpace = self.copy()
				for index in range(len(leaves)):
					permutationSpaceForListOfPiles = permutationSpaceForListOfPiles.atPilePinLeaf(listOfPiles[index], leaves[index])
				deconstructedPermutationSpace.append(permutationSpaceForListOfPiles)
		else:
			deconstructedPermutationSpace.append(self)

		return deconstructedPermutationSpace

	def extractPinnedLeaves(self) -> PinnedLeaves:
		"""Create a dictionary *sorted* by `pile` of only `pile: leaf` without `pile: leafOptions`.

		Returns
		-------
		dictionaryOfPileLeaf : dict[int, int]
			Dictionary of `pile` with pinned `leaf`, if a `leaf` is pinned at `pile`.
		"""
		return dict(sorted(DOTitems(filterLeaf(isLeaf吗, self))))

	def extractUndeterminedPiles(self) -> UndeterminedPiles:
		"""Return a dictionary *sorted* by `pile` of all `pile: leafOptions` in `PermutationSpace`.

		Returns
		-------
		pilesUndetermined : dict[int, LeafOptions]
			Dictionary of `pile: leafOptions`, if a `leafOptions` is defined at `pile`.
		"""
		return dict(sorted(DOTitems(filterLeaf(isLeafOptions吗, self))))

	# TODO `getLeaf` is modeled directly on `.get()`. Therefore, ought `default: Leaf | None` be so
	# restrictive? I cannot think of any use case for a `default` that is not a `Leaf` or `None`, but
	# I cannot think of a reason to restrict it.
	# TODO `isLeaf吗` returns `TypeIs`, and I almost wrote that I should change the return type of
	# `getLeaf`. Therefore, the REAL todo is: I need to stop coding and go eat.
	def getLeaf(self, pile: Pile, default: Leaf | None = None) -> Leaf | None:
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
			The `Leaf` at `permutationSpace[pile]` if `permutationSpace[pile]` is a `Leaf`,
			otherwise `default`.
		"""
		ImaLeaf: LeafSpace | None = self.get(pile)
		if isLeaf吗(ImaLeaf):
			return ImaLeaf
		return default

	def getLeafOptions(self, pile: Pile, default: LeafOptions | None = None) -> LeafOptions | None:
		"""Read `permutationSpace[pile]` only when `permutationSpace[pile]` is a `LeafOptions`.

		Parameters
		----------
		pile : Pile
			`Pile` index to look up in `permutationSpace`.
		default : LeafOptions | None = None
			Value to return when `permutationSpace[pile]` is not a `LeafOptions`.

		Returns
		-------
		leafOptionsOrNone : LeafOptions | None
			`LeafOptions` value from `permutationSpace[pile]`, or `default`.
		"""
		ImaLeafOptions: LeafSpace | None = self.get(pile)
		if isLeafOptions吗(ImaLeafOptions):
			return ImaLeafOptions
		return default

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

	# TODO Learn how to use caching for a method. Once a `Leaf` is pinned, it will always be pinned in
	# this `PermutationSpace`. Is it possible to conditionally cache? I don't want to cache `False`
	# because that could change.
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
		return leaf == self.get(pile)

	# TODO Consider implementing another method to make a `Folding` or _maybe_ cleverly overloading
	# this method (I'm deeply skeptical that overload is a good idea). `makeFolding` handles _my_
	# current needs. If I had to create ONE `makeFolding` function/method with the most utility,
	# however, it would NOT look like this function. 2026 July 10: off the top of my head, passing
	# `listPileLeaf: Sequence[tuple[Pile, Leaf]]` would be better than the current function and is
	# probably close to the ideal generalized function.
	def makeFolding(self, leavesToInsert: Sequence[Leaf]) -> Folding:
		# DOCUMENT `pilesToInsert` is sorted from smallest to largest Pile. `leavesToInsert` must be ordered with that in mind.
		pilesToInsert: Iterator[Pile] = DOTkeys(self.extractUndeterminedPiles())
		# NOTE `cast` because the type checkers cannot possible know that the prior logic leads to all int.
		# TODO Think about: I _feel_ like this logic could be more efficient. This
		# `tuple(DOTvalues(dict(sorted(DOTitems` has THREE constructors (`sorted` is a stealth `list`
		# constructor) or FIVE constructors if `Iterator` is a constructor (`DOTitems` and
		# `DOTvalues`), so I _feel_ it would be faster if I could change the values without
		# ping-ponging from `dict` to `list` to `dict` to `tuple`.
		return tuple(DOTvalues(dict(sorted(DOTitems(cast("PinnedLeaves", merge(self, dict(zip(pilesToInsert, leavesToInsert, strict=True)), factory=PermutationSpace)))))))

	def pileNotOpen吗(self, pile: Pile) -> bool:
		"""Return `True` if a `Leaf` is pinned at `pile` in this `PermutationSpace`.

		Parameters
		----------
		pile : Pile
			`Pile` index.

		Returns
		-------
		pileIsNotOpen : bool
			`True` if this `PermutationSpace` contains a `Leaf` at `pile`.
		"""
		return isLeaf吗(self[pile])

	# DOCUMENT "Do you want to know if the pile is open or do you really want to know the Python `type` of the value at that key?"
	# SEMIOTICS `pileUndetermined吗` to mirror `UndeterminedPiles`? The pile isn't exactly "open".
	def pileOpen吗(self, pile: Pile) -> bool:
		"""Return `True` if `pile` is open in this `PermutationSpace`.

		Parameters
		----------
		pile : Pile
			`Pile` index.

		Returns
		-------
		pileIsOpen : bool
			`True` if this `PermutationSpace` contains `LeafOptions` at `pile`.
		"""
		return not isLeaf吗(self[pile])

@dataclasses.dataclass(slots=True)
class EliminationState:
	"""Computational state for algorithms that compute `foldsTotal` by elimination.

	This data basket stores both mutable workbench fields (which change during the search) and
	precomputed constants derived from `mapShape` (which do not change after `__post_init__`).

	Attributes
	----------
	mapShape : tuple[int, ...]
		Dimension lengths of the map being analyzed.
	groupsOfFolds : int = 0
		Count of distinct `Folding` pattern groups found so far.
	listFolding : list[`Folding`]
		List of `Folding` patterns found.
	listPermutationSpace : list[`PermutationSpace`]
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
	dimensionsTotal : int
		Unchanging total number of axes in `mapShape`.
	foldingCheckSum : int
		Unchanging triangular-number check-sum for a valid `Folding`.
	leafLast : `Leaf`
		Unchanging 0-indexed largest `leaf` value.
	leavesTotal : int
		Unchanging total number of leaves in the map.
	pileLast : `Pile`
		Unchanging 0-indexed largest `pile` value.
	pilesTotal : int
		Unchanging total number of piles in the map.
	productsOfDimensions : tuple[int, ...]
		Unchanging products of dimension lengths, from the empty product through all dimensions.
	sumsOfProductsOfDimensions : tuple[int, ...]
		Unchanging sums of `productsOfDimensions` from the head.
	sumsOfProductsOfDimensionsNearest首 : tuple[int, ...]
		Unchanging sums of `productsOfDimensions` from the head `首`.
	首 : int
		Unchanging single-base positional-numeral value of the first out-of-bounds Cartesian coordinate.

	Notes
	-----
	The computed `foldsTotal` is `groupsOfFolds * leavesTotal * Theorem2Multiplier * Theorem3Multiplier * Theorem4Multiplier`.

	"""

	mapShape: tuple[int, ...] = dataclasses.field(init=True)
	"""Dimensions of the map being analyzed for folding patterns."""

	groupsOfFolds: int = 0
	"""`foldsTotal` is divisible by `leavesTotal`; the algorithm counts each `Folding` that represents a group of `leavesTotal`-many foldings."""

	listFolding: deque[Folding] = dataclasses.field(default_factory=deque[Folding], init=True)
	"""A list of `Folding` patterns found."""
	listPermutationSpace: deque[PermutationSpace] = dataclasses.field(default_factory=deque[PermutationSpace], init=True)
	"""A list of dictionaries (`{pile: leaf or possible leaves}`) that each define an exclusive permutation space: no overlap between dictionaries."""

	pile: Pile = -1
	"""The `pile` on the workbench."""
	permutationSpace: PermutationSpace = dataclasses.field(default_factory=PermutationSpace, init=True)
	"""The `permutationSpace` dictionary (`{pile: leaf or possible leaves}`) on the workbench."""

	Theorem2aMultiplier: int = 1
	Theorem2Multiplier: int = 1
	Theorem3Multiplier: int = 1
	Theorem4Multiplier: int = 1

	dimensionsTotal: int = dataclasses.field(init=False)
	"""Unchanging total number of dimensions in the map."""
	foldingCheckSum: int = dataclasses.field(init=False)
	"""Unchanging triangular number check-sum for a valid `Folding`."""
	leafLast: Leaf = dataclasses.field(init=False)
	"""Unchanging 0-indexed largest `leaf` in a `Folding`."""
	leavesTotal: int = dataclasses.field(init=False)
	"""Unchanging total number of leaves in the map."""
	pileLast: Pile = dataclasses.field(init=False)
	"""Unchanging 0-indexed final `pile` in a `Folding`."""
	pilesTotal: int = dataclasses.field(init=False)
	"""Unchanging total number of piles in the map."""
	productsOfDimensions: tuple[int, ...] = dataclasses.field(init=False)
	"""Unchanging list of products of map dimensions from the product of no dimensions, `[0]`, to the product of all dimensions, `[dimensionsTotal + inclusive]`."""
	sumsOfProductsOfDimensions: tuple[int, ...] = dataclasses.field(init=False)
	"""Unchanging list of sums of products of map dimensions from the sum of no products, `[0]`, to the sum of all products, `[len(productsOfDimensions) + inclusive]`."""
	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = dataclasses.field(init=False)
	"""Unchanging list of sums of products of map dimensions starting from the head `首`, from the sum of no products, `[0]`, to the sum of all products, `[len(productsOfDimensions) + inclusive]`."""
	首: int = dataclasses.field(init=False)
	"""Unchanging single-base positional-numeral value of the Cartesian coordinates that are the first to be _out-of-bounds_ for the `mapShape`."""

	@property
	def foldsTotal(self) -> int:
		"""The computed number of distinct `Folding` patterns for this `mapShape`."""
		return prod((self.groupsOfFolds, self.Theorem2aMultiplier, self.Theorem2Multiplier, self.Theorem3Multiplier, self.Theorem4Multiplier))

	def __post_init__(self) -> None:
		"""One-time computation of unchanging values."""
		self.dimensionsTotal = len(self.mapShape)
		self.leavesTotal = getLeavesTotal(self.mapShape)
		if 0 < self.leavesTotal:
			self.Theorem2aMultiplier = self.leavesTotal
		self.leafLast = self.leavesTotal - 1
		self.foldingCheckSum = self.leafLast * self.leavesTotal // 2  # https://en.wikipedia.org/wiki/Triangular_number
		self.pilesTotal = self.leavesTotal
		self.pileLast = self.pilesTotal - 1
		self.首 = self.leavesTotal
		self.productsOfDimensions = getProductsOfDimensions(self.mapShape)
		self.sumsOfProductsOfDimensions = getSumsOfProductsOfDimensions(self.mapShape)
		self.sumsOfProductsOfDimensionsNearest首 = getSumsOfProductsOfDimensionsNearest首(self.productsOfDimensions, self.dimensionsTotal, self.dimensionsTotal)
