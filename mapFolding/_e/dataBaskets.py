"""Use data baskets to easily move data, including values that affect computations: don't limit yourself to one data basket per algorithm."""
from __future__ import annotations

from collections import deque
from humpy_cytoolz import compose, merge, valmap as mapLeaf
from hunterMakesPy import raiseIfNone
from mapFolding._e import getProductsOfDimensions, getSumsOfProductsOfDimensions, getSumsOfProductsOfDimensionsNearest首, JeanValjean
from mapFolding._e.theTypes import Folding, LeafSpace, PermutationSpace, Pile, UndeterminedPiles
from mapFolding.beDRY import getLeavesTotal
from math import prod
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from mapFolding._e.theTypes import Leaf

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
	permutationSpace: PermutationSpace = dataclasses.field(default_factory=dict[Pile, LeafSpace], init=True)
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
	# 	self.permutationSpace = merge(mapLeaf(compose(raiseIfNone, JeanValjean), dictionaryLeafOptions), self.permutationSpace)
	# 	return self
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
