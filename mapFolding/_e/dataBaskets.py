"""Use data baskets to easily move data, including values that affect computations: don't limit yourself to one data basket per algorithm."""
from mapFolding import getLeavesTotal
from mapFolding._e import (
	Folding, getProductsOfDimensions, getSumsOfProductsOfDimensions, getSumsOfProductsOfDimensionsNearest首, Leaf,
	LeafOrPileRangeOfLeaves, PermutationSpace, Pile)
from math import prod
import dataclasses

@dataclasses.dataclass(slots=True)
class EliminationState:
	"""Computational state for algorithms to compute foldsTotal by elimination.

	Attributes
	----------
	mapShape : tuple[int, ...]
		Dimension lengths of the map being analyzed for folding patterns.
	groupsOfFolds : int = 0
		Current count of distinct folding pattern groups: each group has `leavesTotal`-many foldings.
	listPermutationSpace : list[dict[int, int]]
		A list of dictionaries that each define an exclusive permutation space: no overlap between dictionaries.
	pile : int = -1
		The `pile` on the workbench.
	permutationSpace : dict[int, int]
		The `permutationSpace` dictionary on the workbench.
	Theorem2Multiplier : int = 1
		Multiplier for Theorem 2 optimizations.
	Theorem3Multiplier : int = 1
		Multiplier for Theorem 3 optimizations.
	Theorem4Multiplier : int = 1
		Multiplier for Theorem 4 optimizations.
	dimensionsTotal : int
		Unchanging total number of dimensions in the map.
	leafLast : int
		Unchanging 0-indexed final `leaf` of the map.
	leavesTotal : int
		Unchanging total number of leaves of the map.
	pileLast : int
		Unchanging 0-indexed final `pile` in a `Folding`.
	productsOfDimensions : list[int]
		Unchanging list of products of map dimensions from the product of no dimensions to the product of all dimensions.
	"""

	mapShape: tuple[int, ...] = dataclasses.field(init=True)
	"""Dimensions of the map being analyzed for folding patterns."""

	groupsOfFolds: int = 0
	"""`foldsTotal` is divisible by `leavesTotal`; the algorithm counts each `Folding` that represents a group of `leavesTotal`-many foldings."""

	listFolding: list[Folding] = dataclasses.field(default_factory=list[Folding], init=True)
	"""A list of `Folding` patterns found."""
	listPermutationSpace: list[PermutationSpace] = dataclasses.field(default_factory=list[PermutationSpace], init=True)
	"""A list of dictionaries (`{pile: leaf or possible leaves}`) that each define an exclusive permutation space: no overlap between dictionaries."""

	pile: Pile = -1
	"""The `pile` on the workbench."""
	permutationSpace: PermutationSpace = dataclasses.field(default_factory=dict[Pile, LeafOrPileRangeOfLeaves], init=True)
	"""The `permutationSpace` dictionary (`{pile: leaf or possible leaves}`) on the workbench."""

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
		return prod((self.groupsOfFolds, self.leavesTotal, self.Theorem2Multiplier, self.Theorem3Multiplier, self.Theorem4Multiplier))

	def __post_init__(self) -> None:
		"""One-time computation of unchanging values."""
		self.dimensionsTotal = len(self.mapShape)
		self.leavesTotal = getLeavesTotal(self.mapShape)
		self.leafLast = self.leavesTotal - 1
		self.foldingCheckSum = self.leafLast * self.leavesTotal // 2 # https://en.wikipedia.org/wiki/Triangular_number
		self.pilesTotal = self.leavesTotal
		self.pileLast = self.pilesTotal - 1
		self.首 = self.leavesTotal
		self.productsOfDimensions = getProductsOfDimensions(self.mapShape)
		self.sumsOfProductsOfDimensions = getSumsOfProductsOfDimensions(self.mapShape)
		self.sumsOfProductsOfDimensionsNearest首 = getSumsOfProductsOfDimensionsNearest首(self.productsOfDimensions, self.dimensionsTotal, self.dimensionsTotal)
