"""Use data baskets to easily move data, including values that affect computations: don't limit yourself to one data basket per algorithm."""
from __future__ import annotations

from collections import deque
from functools import partial
from humpy_cytoolz import assoc as associate, compose, dissoc as dissociatePile, merge, valfilter as filterLeaf, valmap as mapLeaf
from hunterMakesPy import raiseIfNone
from mapFolding._e import getProductsOfDimensions, getSumsOfProductsOfDimensions, getSumsOfProductsOfDimensionsNearest首, JeanValjean
from mapFolding._e.filters import isLeafOptions吗, isLeaf吗, leafNotPinned吗, leafPinnedAtPile吗, pileOpen吗
from mapFolding._e.theTypes import Folding, LeafSpace, Pile, UndeterminedPiles
from mapFolding.beDRY import getLeavesTotal
from mapFolding.genericNeedsNewHome import DOTitems, DOTkeys, DOTvalues
from math import prod
from typing import cast, TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from mapFolding._e.theTypes import Leaf, LeafOptions, PinnedLeaves


class PermutationSpace(dict[Pile, LeafSpace]):  # noqa: FURB189
	"""Represent `pile: leaf` and `pile: leafOptions` mappings with pinning helper methods."""

	def copy(self) -> PermutationSpace:
		return PermutationSpace(self)

	def addMissingLeafOptions(self, dictionaryLeafOptions: UndeterminedPiles) -> PermutationSpace:
		"""Return a new `PermutationSpace` with default `LeafOptions` for missing piles.

		Parameters
		----------
		dictionaryLeafOptions : UndeterminedPiles
			Default `LeafOptions` by `Pile`.

		Returns
		-------
		permutationSpace : PermutationSpace
			New `PermutationSpace` with current entries overriding default `LeafOptions`.
		"""
		return PermutationSpace(merge(mapLeaf(compose(raiseIfNone, JeanValjean), dictionaryLeafOptions), self))

	def extractPinnedLeaves(self) -> PinnedLeaves:
		"""Create a dictionary *sorted* by `pile` of only `pile: leaf` without `pile: leafOptions`.

		Returns
		-------
		dictionaryOfPileLeaf : dict[int, int]
			Dictionary of `pile` with pinned `leaf`, if a `leaf` is pinned at `pile`.
		"""
		return dict(sorted(DOTitems(filterLeaf(isLeaf吗, self))))

	def extractUndeterminedPiles(self) -> UndeterminedPiles:
		"""Return a dictionary of all pile-ranges of leaves in `permutationSpace`.

		Returns
		-------
		pilesUndetermined : dict[int, LeafOptions]
			Dictionary of `pile: leafOptions`, if a `leafOptions` is defined at `pile`.
		"""
		return filterLeaf(isLeafOptions吗, self)

	def DOTgetPileIfLeaf(self, pile: Pile, default: Leaf | None = None) -> Leaf | None:
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

	def DOTgetPileIfLeafOptions(self, pile: Pile, default: LeafOptions | None = None) -> LeafOptions | None:
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

	def atPilePinLeaf(self, pile: Pile, leaf: Leaf) -> PermutationSpace:
		"""Return a new `PermutationSpace` with `leaf` pinned at `pile` without modifying `permutationSpace`.

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
		"""
		return PermutationSpace(associate(self, pile, cast("LeafSpace", leaf)))

	def atPilePinLeafSafetyFilter(self, pile: Pile, leaf: Leaf) -> bool:
		"""Return `True` if it is safe to call `permutationSpace.atPilePinLeaf(pile, leaf)`.

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
		return leafPinnedAtPile吗(self, leaf, pile) or (pileOpen吗(self, pile) and leafNotPinned吗(self, leaf))

	def deconstructPermutationSpaceAtPile(self, pile: Pile, leavesToPin: Iterable[Leaf]) -> dict[Leaf, PermutationSpace]:
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
		if (leaf := self.DOTgetPileIfLeaf(pile)) is not None:
			deconstructedPermutationSpace: dict[Leaf, PermutationSpace] = {leaf: self}
		else:
			pin: Callable[[Leaf], PermutationSpace] = partial(self.atPilePinLeaf, pile)
			leafCanBePinned: Callable[[Leaf], bool] = leafNotPinned吗(self)
			deconstructedPermutationSpace = {leaf: pin(leaf) for leaf in filter(leafCanBePinned, leavesToPin)}
		return deconstructedPermutationSpace

	def bifurcatePermutationSpace(self) -> tuple[PinnedLeaves, UndeterminedPiles]:
		"""Split a `PermutationSpace` into `PinnedLeaves` and `UndeterminedPiles`.

		Returns
		-------
		leavesPinned : PinnedLeaves
			Dictionary of `Pile` to pinned `Leaf` mappings.
		pilesUndetermined : UndeterminedPiles
			Dictionary of `Pile` to `LeafOptions` domain mappings.
		"""
		leavesPinned: PinnedLeaves = self.extractPinnedLeaves()
		# NOTE `cast` because type checkers don't know `PermutationSpace` - `PinnedLeaves` = `UndeterminedPiles`.
		return (leavesPinned, cast("UndeterminedPiles", dissociatePile(self, *DOTkeys(leavesPinned))))

	def makeFolding(self, leavesToInsert: Sequence[Leaf]) -> Folding:
		pilesToInsert: Iterator[Pile] = DOTkeys(self.extractUndeterminedPiles())
		# NOTE `cast` because the type checkers cannot possible know that the prior logic leads to all int.
		return tuple(DOTvalues(dict(sorted(DOTitems(cast("PinnedLeaves", merge(self, cast("dict[Pile, LeafSpace]", dict(zip(pilesToInsert, leavesToInsert, strict=True))))))))))

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

#==== PermutationSpace(dict) ====
# TODO Make `PermutationSpace` a subclass of `dict` so I can add methods.
# NOTE I REFUSE TO BE AN OBJECT-ORIENTED PROGRAMMER!!! But, I'll use some OOP if it makes sense.
# Goals: DRY code, useful code, a useful PermutationSpace `object`, EFFICIENCY, seamless integration with a strongly functional paradigm.
# On EFFICIENCY: this object will help enumerate ~362794844160000 permutations for A001417(8), for
# example. One extra clock cycle on one oft-called operation can add days to a multi-week computation.

#---- method (only?) ------------
# addMissingLeafOptionsToPermutationSpace
	# def addMissingLeafOptions(self, dictionaryLeafOptions: UndeterminedPiles):
	# 	# Incomplete prototype.
	# 	self.permutationSpace = merge(mapLeaf(compose(raiseIfNone, JeanValjean), dictionaryLeafOptions), self.permutationSpace)  # noqa: ERA001
	# 	return self  # noqa: ERA001
# atPilePinLeaf
# atPilePinLeafSafetyFilter
# bifurcatePermutationSpace
# DOTgetPileIfLeaf
# DOTgetPileIfLeafOptions
# extractPinnedLeaves
# extractUndeterminedPiles
# makeFolding

#---- method and function (?) ---
# NOTE Remember the goals when deciding method, function, or both. When implementing both, DRYer code helps
#	 to ensure that behavior is consistent between the method and the function.
# deconstructPermutationSpaceAtPile
# deconstructPermutationSpaceByDomainOfLeaf
# deconstructPermutationSpaceByDomainsCombined
# leafNotPinned吗
# leafPinned吗
# leafPinnedAtPile吗
# pileNotOpen吗
# pileOpen吗
