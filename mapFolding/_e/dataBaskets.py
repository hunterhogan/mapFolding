# ruff:file-ignore[subclass-builtin]
# TODO idk enough to choose between `UserDict` and subclassing `dict`.
"""Use data baskets to easily move data, including values that affect computations: don't limit yourself to one data basket per algorithm."""

from __future__ import annotations

from collections import deque
# TODO `partial` vs `humpy_cytoolz.functoolz.curry`: which is better?
from functools import partial
from gmpy2 import bit_mask
from humpy_cytoolz import (
	assoc as associateKeyValue, compose, dissoc as dissociatePile, first, groupby as toolz_groupby, merge, valfilter as filterLeaf)
from hunterMakesPy import raiseIfNone
from itertools import combinations, filterfalse
from mapFolding._e import (
	getIteratorOfLeaves, getLeafDomain, getProductsOfDimensions, getSumsOfProductsOfDimensions, getSumsOfProductsOfDimensionsNearest首)
from mapFolding._e.algorithms.iff import creaseViolation吗, getCreasePost, oddLeaf吗
from mapFolding._e.filters import isLeafOptions吗, isLeaf吗, leafInLeafOptions吗
from mapFolding._e.theTypes import Folding, LeafOptions, LeafSpace, Pile
from mapFolding.beDRY import getLeavesTotal
from math import prod
from operator import attrgetter, methodcaller
from typing import cast, overload, TYPE_CHECKING, TypeIs
from Z0Z_tools import DOTitems, DOTkeys, DOTvalues
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding._e.theTypes import Leaf, PinnedLeaves, UndeterminedPiles
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

	def addMissingPileLeafSpace(self, missing: PermutationSpace | UndeterminedPiles | PinnedLeaves) -> PermutationSpace:
		"""Update missing `Pile: LeafSpace` items with the items from `missing`.

		This will not overwrite any existing `Pile: LeafSpace` items in `permutationSpace` because
		that would corrupt the `PermutationSpace`.

		Parameters
		----------
		missing : PermutationSpace | UndeterminedPiles | PinnedLeaves
			`LeafSpace` by `Pile` in `missing`.

		Returns
		-------
		permutationSpace : PermutationSpace
			New `PermutationSpace` and modifies `PermutationSpace` in place.
		"""
		#=EndNotes##sorted=
		self = PermutationSpace(sorted(DOTitems(merge(missing, self, factory=PermutationSpace))))
		return self.copy()

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
			Dictionary of `Pile` to `LeafOptions` domain mappings.
		"""
		leavesPinned: PinnedLeaves = self.extractPinnedLeaves()
		#=SIN= `cast`: type checkers cannot infer that partitioning `PermutationSpace` preserves `UndeterminedPiles`.
		return (leavesPinned, cast('UndeterminedPiles', dissociatePile(self, *DOTkeys(leavesPinned))))

	def deconstructAtPile(self, pile: Pile | None = None, leavesToPin: Iterable[Leaf] = ()) -> Iterable[PermutationSpace]:
		# DOCUMENT Do not keep the original if you use any part of the return.
		if pile is None:
			pile = first(filterLeaf(isLeafOptions吗, self))
		if (leafOptions := self.getLeafOptions(pile)) is None:
			deconstructed: Iterable[PermutationSpace] = deque([self])
		else:
			leavesToPin = leavesToPin or getIteratorOfLeaves(leafOptions)
			deconstructed = map(partial(self.atPilePinLeaf, pile), filter(self.leafNotPinned吗, leavesToPin))
		return deconstructed

	def deconstructByDomainOfLeaf(self, leaf: Leaf, leafDomain: Iterable[Pile]) -> deque[PermutationSpace]:
		"""Pin `leaf` at each open `pile` in the domain of `leaf`.

		Return a `deque` containing this `PermutationSpace` if `leaf` is already pinned, or one
		`PermutationSpace` for each open `pile` in `leafDomain` with `leaf` pinned at `pile`.

		Parameters
		----------
		leaf : int
			`leaf` to pin.
		leafDomain : Iterable[int]
			Domain of `pile` indices for `leaf`.

		Returns
		-------
		deconstructedPermutationSpace : deque[PermutationSpace]
			Deque of `PermutationSpace` dictionaries with `leaf` pinned at each open `pile` in
			`leafDomain`.
		"""
		deconstructedPermutationSpace: deque[PermutationSpace] = deque()
		if self.leafNotPinned吗(leaf):
			leafInPileRange: Callable[[int], bool] = compose(
				leafInLeafOptions吗(leaf), partial(self.getLeafOptions, default=bit_mask(len(self)))
			)
			pinLeafAt: Callable[[int], PermutationSpace] = partial(self.atPilePinLeaf, leaf=leaf)
			deconstructedPermutationSpace.extend(map(pinLeafAt, filter(leafInPileRange, filter(self.pileUndetermined吗, leafDomain))))
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
			Deque of `PermutationSpace` dictionaries with the requested leaves pinned across
			compatible pile tuples.
		"""
		deconstructedPermutationSpace: deque[PermutationSpace] = deque()

		def pileOpenByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:
			def workhorse(domain: Sequence[Pile]) -> bool:
				return self.pileUndetermined吗(domain[index])

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
		return filterLeaf(isLeafOptions吗, self)

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
		ImaLeaf: LeafSpace | None = self.get(pile)
		if isLeaf吗(ImaLeaf):
			return ImaLeaf
		return default

	@overload
	def getLeafOptions(self, pile: Pile, default: None = None) -> LeafOptions | None: ...
	@overload
	def getLeafOptions(self, pile: Pile, default: LeafOptions) -> LeafOptions: ...
	@overload
	def getLeafOptions[个](self, pile: Pile, default: 个) -> LeafOptions | 个: ...
	def getLeafOptions[个](self, pile: Pile, default: LeafOptions | 个 | None = None) -> LeafOptions | 个 | None:
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

	@property
	def leafCount(self) -> int:
		"""Count of `Leaf` indices that are pinned in this `PermutationSpace`.

		Returns
		-------
		leafCount : int
			Count of `Leaf` indices that are pinned in this `PermutationSpace`.
		"""
		return sum(map(isLeaf吗, self.values()))

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
		pilesToInsert: Iterator[Pile] = DOTkeys(self.extractUndeterminedPiles())
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
		`mapFolding._e.filters.isLeafOptions吗`
			Narrow an existing `LeafSpace` value to `LeafOptions`.
		"""
		return isLeaf吗(self[pile])

	def pileUndetermined吗(self, pile: Pile) -> bool:
		"""Determine whether `pile` still requires a `Leaf` assignment.

		Use this method when control flow concerns whether a `Pile` still requires a `Leaf`
		assignment. Use `isLeafOptions吗` when the logic already has a `LeafSpace` value and needs
		Python type narrowing.

		Parameters
		----------
		pile : Pile
			`Pile` index.

		Returns
		-------
		pileIsUndetermined : bool
			`True` if this `PermutationSpace` contains `LeafOptions` at `pile`.

		See Also
		--------
		`pilePinned吗`
			Determine whether a `Pile` already has a pinned `Leaf`.
		`mapFolding._e.filters.isLeafOptions吗`
			Narrow an existing `LeafSpace` value to `LeafOptions`.
		`mapFolding._e.filters.isLeaf吗`
			Narrow an existing `LeafSpace` value to `Leaf`.
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
		Unchanging single-base positional-numeral value of the first out-of-bounds Cartesian
		coordinate.

	Notes
	-----
	The computed `foldsTotal` is `groupsOfFolds * leavesTotal * Theorem2Multiplier *
	Theorem3Multiplier * Theorem4Multiplier`.

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
		return prod(
			(self.groupsOfFolds, self.Theorem2aMultiplier, self.Theorem2Multiplier, self.Theorem3Multiplier, self.Theorem4Multiplier)
		)

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
		self.sumsOfProductsOfDimensionsNearest首 = getSumsOfProductsOfDimensionsNearest首(
			self.productsOfDimensions, self.dimensionsTotal, self.dimensionsTotal
		)

	def moveToListFolding(self) -> Self:
		foldingGroup吗: dict[bool, list[PermutationSpace]] = toolz_groupby(
			compose(self.leavesTotal.__eq__, attrgetter('leafCount')), self.listPermutationSpace
		)
		self.listPermutationSpace = deque(foldingGroup吗.get(False, ()))
		self.listFolding.extend(map(methodcaller('makeFolding'), foldingGroup吗.get(True, ())))
		return self

	def permutationSpaceCreaseViolation吗(self, permutationSpace: PermutationSpace) -> bool:
		"""You can detect forbidden crease crossings inside `state.permutationSpace`.

		`permutationSpaceCreaseViolation吗` is a pruning predicate used before counting or expanding a
		candidate `PermutationSpace`. `removeCreaseViolationsFromEliminationState` uses
		`permutationSpaceCreaseViolation吗` to filter `state.listPermutationSpace` [5], and a caller
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
		leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(permutationSpace.extractPinnedLeaves())}

		for dimension in range(self.dimensionsTotal):
			listPileCreaseByParity: list[list[tuple[Pile, Pile]]] = [[], []]
			for pile, leaf in permutationSpace.extractPinnedLeaves().items():
				crease: int | None = getCreasePost(self.mapShape, leaf, dimension)
				if crease:
					pileCrease: int | None = leafToPile.get(crease)
					if pileCrease:
						listPileCreaseByParity[oddLeaf吗(self.mapShape, leaf, dimension)].append((pile, pileCrease))
			for groupedParity in listPileCreaseByParity:
				if any(creaseViolation吗(pile, pileComparand, pileCrease, pileComparandCrease)
					for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2)):
						return True
		return False

	def pinAt_pile吗(self, leaf: Leaf) -> bool:
		return all((
			self.permutationSpace.leafNotPinned吗(leaf)
			, self.permutationSpace.pileUndetermined吗(self.pile)
			, self.pile in getLeafDomain(self, leaf)
		))

	def reduceAllPermutationSpace(
		self, listFunctionsReduction: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]]
	) -> Self:
		listPermutationSpace: deque[PermutationSpace] = self.listPermutationSpace
		self.listPermutationSpace = deque()
		listPermutationSpaceIrreducible: deque[PermutationSpace] = deque()

		while listPermutationSpace:
			#------------ Initialize `permutationSpace` ------------------------------
			permutationSpace: PermutationSpace | None = listPermutationSpace.pop()
			sumPermutationSpace: Leaf | LeafOptions = sum(permutationSpace.values())
			functionsReduction: deque[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = deque(
				listFunctionsReduction
			)
			keepGoing: bool = True

			while keepGoing:
				reducePermutationSpace: Callable[[EliminationState, PermutationSpace], PermutationSpace | None] = (
					functionsReduction.popleft()
				)
				permutationSpace = reducePermutationSpace(self, raiseIfNone(permutationSpace))

				if not permutationSpace:
					keepGoing = False
				elif sumPermutationSpace != sum(permutationSpace.values()):
					functionsReduction = deque(listFunctionsReduction)
					sumPermutationSpace = sum(permutationSpace.values())
				elif not functionsReduction:
					listPermutationSpaceIrreducible.append(permutationSpace)
					keepGoing = False

		else:
			self.listPermutationSpace.extend(listPermutationSpaceIrreducible)

		return self

	def removeCreaseViolations(self) -> Self:
		"""You can filter `state.listPermutationSpace` by removing crease-crossing candidates.

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
			The same instance with `self.listPermutationSpace` filtered.

		References
		----------
		[1] mapFolding._e.algorithms.iff.permutationSpaceHasIFFViolation

		[2] mapFolding._e.pin2上nDimensions
		"""
		listPermutationSpace: deque[PermutationSpace] = self.listPermutationSpace.copy()
		self.listPermutationSpace = deque()
		self.listPermutationSpace.extend(filterfalse(self.permutationSpaceCreaseViolation吗, listPermutationSpace))

		return self
