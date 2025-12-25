from mapFolding import getLeavesTotal, inclusive
from mapFolding._e import LeafOrPileRangeOfLeaves, PermutationSpace
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
	leavesPinned : dict[int, int]
		The `leavesPinned` dictionary on the workbench.
	Theorem2Multiplier : int = 1
		Multiplier for Theorem 2 optimizations.
	Theorem3Multiplier : int = 1
		Multiplier for Theorem 3 optimizations.
	Theorem4Multiplier : int = 1
		Multiplier for Theorem 4 optimizations.
	dimensionsTotal : int
		Unchanging total number of dimensions in the map.
	leafLast : int
		Unchanging 0-indexed final `leaf` in a `folding`.
	leavesTotal : int
		Unchanging total number of leaves in the map.
	pileLast : int
		Unchanging 0-indexed final `pile` in a `folding`.
	productsOfDimensions : list[int]
		Unchanging list of products of map dimensions from the product of no dimensions to the product of all dimensions.

	Notes
	-----
	Fine control over integer bit-widths is not currently part of these algorithms, so removing `TypeAlias` reduces complexity.
	(And will hopefully stop the intermittent false-positive Pylance diagnostic, `expected 0 positional arguments`.)
	"""

	mapShape: tuple[int, ...] = dataclasses.field(init=True)
	"""Dimensions of the map being analyzed for folding patterns."""

	groupsOfFolds: int = 0
	"""`foldsTotal` is divisible by `leavesTotal`; the algorithm counts each `folding` that represents a group of `leavesTotal`-many foldings."""

	listPermutationSpace: list[PermutationSpace] = dataclasses.field(default_factory=list[PermutationSpace], init=True)
	"""A list of dictionaries (`{pile: leaf or possible leaves}`) that each define an exclusive permutation space: no overlap between dictionaries."""

	pile: int = -1
	"""The `pile` on the workbench."""
	leavesPinned: PermutationSpace = dataclasses.field(default_factory=dict[int, LeafOrPileRangeOfLeaves], init=True)
	"""The `leavesPinned` dictionary (`{pile: leaf or possible leaves}`) on the workbench."""

	Theorem2Multiplier: int = 1
	Theorem3Multiplier: int = 1
	Theorem4Multiplier: int = 1

	dimensionsTotal: int = dataclasses.field(init=False)
	"""Unchanging total number of dimensions in the map."""
	leafLast: int = dataclasses.field(init=False)
	"""Unchanging 0-indexed final `leaf` in a `folding`."""
	leavesTotal: int = dataclasses.field(init=False)
	"""Unchanging total number of leaves in the map."""
	pileLast: int = dataclasses.field(init=False)
	"""Unchanging 0-indexed final `pile` in a `folding`."""
	productsOfDimensions: tuple[int, ...] = dataclasses.field(init=False)
	"""Unchanging list of products of map dimensions from the product of no dimensions, `[0]`, to the product of all dimensions, `[dimensionsTotal + inclusive]`."""
	sumsOfProductsOfDimensions: tuple[int, ...] = dataclasses.field(init=False)
	"""Unchanging list of sums of products of map dimensions from the sum of no products, `[0]`, to the sum of all products, `[dimensionsTotal + 1 + inclusive]`."""

	@property
	def foldsTotal(self) -> int:
		"""The computed number of distinct `folding` patterns for this `mapShape`."""
		return prod((self.groupsOfFolds, self.leavesTotal, self.Theorem2Multiplier, self.Theorem3Multiplier, self.Theorem4Multiplier))

	def __post_init__(self) -> None:
		"""One-time computation of unchanging values."""
		self.dimensionsTotal = len(self.mapShape)
		self.leavesTotal = getLeavesTotal(self.mapShape)
		self.pileLast = self.leavesTotal - 1
		self.leafLast = self.leavesTotal - 1
		self.productsOfDimensions = tuple(prod(self.mapShape[0:dimension], start=1) for dimension in range(self.dimensionsTotal + inclusive))
		self.sumsOfProductsOfDimensions = tuple(sum(self.productsOfDimensions[0:aProduct], start=0) for aProduct in range(self.dimensionsTotal + 1 + inclusive))
