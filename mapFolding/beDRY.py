"""Access shared `ChoicesLeaf` operations and map-shape arithmetic without state imports.

(AI generated docstring)

You can construct and inspect leaf sets and compute dimension products and sums. Pass values directly
to use these operations in modules that define elimination state without introducing circular imports.

Contents
--------
ChoicesLeaf functions
	choicesLeafAND
		Intersect two `ChoicesLeaf` values.
	distinguishChoicesLeaf
		Distinguish `ChoicesLeaf`, its only `Leaf`, or `None`.
	getIteratorOfLeaves
		Enumerate leaf indices from a sentinel-bearing `ChoicesLeaf`.
	lengthChoicesLeaf
		Count the possible leaves in `ChoicesLeaf`.
	makeAntiChoicesLeaf
		Build a `ChoicesLeaf` by excluding the specified leaf indices.
	makeChoicesLeaf
		Build a `ChoicesLeaf` from the specified leaf indices.
Be DRY functions
	getMapShapeProducts
		Compute cumulative products of the dimension lengths in a map shape.
	getMapShapeProductsSums
		Compute cumulative sums of a map shape's dimension products.
	getMapShape首ProductsSums
		Compute cumulative dimension-product sums from the head of a map.
Flow control
	mapShapeLengthsAreEqual
		Group dimension indices that share a repeated length greater than one.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import cache, partial, reduce
from gmpy2 import bit_clear, bit_mask, bit_set, xmpz
from humpy_cytoolz import unique
from hunterMakesPy import inclusive, raiseIfNone, zeroIndexed
from hunterMakesPy.parseParameters import defineConcurrencyLimit, intInnit
from itertools import accumulate
from mapFolding.filters import choicesLeaf吗
from more_itertools import iter_index
from numba import get_num_threads, set_num_threads
from operator import add, mul
from sys import maxsize as sysMaxsize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import ChoicesLeaf, Leaf

#======== `ChoicesLeaf` functions ================================================

def distinguishChoicesLeaf(choicesLeaf: ChoicesLeaf, /) -> ChoicesLeaf | Leaf | None:
	"""Distinguish `ChoicesLeaf`, its only `Leaf`, or `None`.

	(AI generated docstring)

	You can distinguish an empty leaf set from a forced leaf assignment. You receive a `ChoicesLeaf` when
	several leaves remain, or the unchanged input when its value is not an instance of the `ChoicesLeaf` type.

	Distinguishing leaf choices
	--------------------------
	You can inspect the set-bit count and first set bit with `gmpy2.mpz.bit_count` and
	`gmpy2.mpz.bit_scan1` [1]. Two set bits identify one leaf plus the sentinel. One set bit identifies
	the sentinel alone.

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` with a sentinel above all leaf bits.

	Returns
	-------
	leafSpaceOrNone : ChoicesLeaf | Leaf | None
		The only remaining leaf, `None` for a sentinel-only `ChoicesLeaf`, or the unchanged value.

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	whoAmI: ChoicesLeaf | Leaf | None = choicesLeaf
	if choicesLeaf吗(choicesLeaf):
		if choicesLeaf.bit_count() == 2:
			whoAmI = raiseIfNone(choicesLeaf.bit_scan1())
		elif choicesLeaf.bit_count() == 1:
			whoAmI = None
	return whoAmI

def choicesLeafAND(choicesLeafDISPOSABLE: ChoicesLeaf, choicesLeaf: ChoicesLeaf) -> ChoicesLeaf:
	"""Intersect two `ChoicesLeaf` values.

	(AI generated docstring)

	You can keep leaf indices present in both `choicesLeafDISPOSABLE` and `choicesLeaf`. You receive a
	new `ChoicesLeaf` with both arguments unchanged.

	Parameters
	----------
	choicesLeafDISPOSABLE : ChoicesLeaf
		Leaf mask containing allowed leaf bits and the sentinel.
	choicesLeaf : ChoicesLeaf
		Leaf set to restrict.

	Returns
	-------
	maskedChoicesLeaf : ChoicesLeaf
		Intersection retaining the sentinel when both inputs contain the sentinel.
	"""
	return choicesLeaf & choicesLeafDISPOSABLE

def getIteratorOfLeaves(choicesLeaf: ChoicesLeaf) -> Iterator[Leaf]:
	"""Enumerate leaf indices from a sentinel-bearing `ChoicesLeaf`.

	(AI generated docstring)

	You can enumerate the leaves represented by `choicesLeaf` in ascending order without changing the
	input `ChoicesLeaf`. You iterate over the remaining set bits after copying `choicesLeaf` to `gmpy2.xmpz` [1]
	and clearing the highest bit as the sentinel.

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		Nonnegative `ChoicesLeaf` whose highest set bit is the sentinel.

	Returns
	-------
	iteratorOfLeaves : Iterator[Leaf]
		Iterator of leaf indices, excluding the sentinel.

	Examples
	--------
	The constraint solver builds allowed assignments from a leaf set.

		```python
		model.add_allowed_assignments(
			[boxOfLeavesInPileOrder[aPile]], list(zip(getIteratorOfLeaves(choicesLeaf)))
		)
		```

	References
	----------
	[1] gmpy2 mutable integers and set-bit iteration.
		https://gmpy2.readthedocs.io/en/latest/advmpz.html
	"""
	iteratorOfLeaves: xmpz = xmpz(choicesLeaf)
	iteratorOfLeaves[-1] = 0
	return iteratorOfLeaves.iter_set()

def lengthChoicesLeaf(choicesLeaf: ChoicesLeaf) -> int:
	"""Count the possible leaves in `ChoicesLeaf`.

	(AI generated docstring)

	You can count the leaf indices represented by `choicesLeaf`, excluding the sentinel from the set-bit
	count provided by `gmpy2.mpz.bit_count` [1].

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` with one sentinel bit above all leaf bits.

	Returns
	-------
	leavesCount : int
		Number of possible leaves, excluding the sentinel.

	Examples
	--------
	The subset reducer compares leaf count with the number of matching piles.

		```python
		if mapFolding.beDRYElimination.lengthChoicesLeaf(choicesLeaf) == len(boxOfPiles):
			permutationSpace = reduceLeafSpace(
				permutationSpace, DOTitems(dissoc(pilesUndetermined, *boxOfPiles)),
				mapFolding.beDRYElimination.makeAntiChoicesLeaf(state.totalLeaves, mapFolding.beDRYElimination.getIteratorOfLeaves(choicesLeaf))
			)
		```

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return choicesLeaf.bit_count() - 1

def makeAntiChoicesLeaf(totalLeaves: int, leaves: Iterable[Leaf]) -> ChoicesLeaf:
	"""Build a `ChoicesLeaf` by excluding the specified leaf indices.

	(AI generated docstring)

	You can obtain a `ChoicesLeaf` containing every leaf index below `totalLeaves` except the indices in
	`leaves`. You retain the sentinel while clearing excluded bits with `gmpy2.bit_clear` [1].

	Parameters
	----------
	totalLeaves : int
		Total number of leaves and index of the sentinel bit.
	leaves : Iterable[Leaf]
		Leaf indices to exclude. Each index must be below `totalLeaves`.

	Returns
	-------
	antiChoicesLeaf : ChoicesLeaf
		`ChoicesLeaf` containing the remaining leaves and the sentinel.

	Examples
	--------
	State normalization excludes leaves already pinned in the permutation space.

		```python
		antiChoicesLeaf = mapFolding.beDRYElimination.makeAntiChoicesLeaf(len(self), leavesPinned.values())
		```

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return reduce(bit_clear, leaves, bit_mask(totalLeaves + inclusive))

def makeChoicesLeaf(totalLeaves: int, leaves: Iterable[Leaf]) -> ChoicesLeaf:
	"""Build a `ChoicesLeaf` from the specified leaf indices.

	(AI generated docstring)

	You can encode `leaves` as a `ChoicesLeaf` with a sentinel at `totalLeaves`. You set the leaf bits with
	`gmpy2.bit_set` [1].

	Parameters
	----------
	totalLeaves : int
		Total number of leaves and index of the sentinel bit.
	leaves : Iterable[Leaf]
		Leaf indices to include. Each index must be below `totalLeaves`.

	Returns
	-------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` containing the requested leaves and the sentinel.

	Examples
	--------
	The pile-options implementation encodes the leaves that pass its filters.

		```python
		return makeChoicesLeaf(totalLeaves, choicesLeaf)
		```

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return reduce(bit_set, leaves, bit_set(0, totalLeaves))

#======== maps and leaves ================================================

def getMapShapeProducts(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""Compute cumulative products of the dimension lengths in a map shape.

	(AI generated docstring)

	You can convert `mapShape` into the strides used to identify leaves across dimensions. You receive a
	tuple that includes the initial empty product and the final product of every dimension.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths in their original order.

	Returns
	-------
	mapShapeProducts : tuple[int, ...]
		Cumulative products beginning with `1`, with length `len(mapShape) + 1`.

	Examples
	--------
	Elimination-state initialization stores the products.

		```python
		self.mapShapeProducts = mapFolding.beDRYElimination.getMapShapeProducts(self.mapShape)
		```
	"""
	return tuple(accumulate(mapShape, mul, initial=1))

def getMapShapeProductsSums(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""Compute cumulative sums of a map shape's dimension products.

	(AI generated docstring)

	You can sum the cumulative products from `getMapShapeProducts` [1]. You receive a tuple that includes
	the initial empty sum and the sum including the total leaf count.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths in their original order.

	Returns
	-------
	mapShapeProductsSums : tuple[int, ...]
		Cumulative sums beginning with `0`, with length `len(mapShape) + 2`.

	References
	----------
	[1] `getMapShapeProducts`
	"""
	return tuple(accumulate(getMapShapeProducts(mapShape), add, initial=0))

def getMapShape首ProductsSums(mapShapeProducts: tuple[int, ...], totalDimensions: int | None = None, dimensionFrom首: int | None = None) -> tuple[int, ...]:
	"""Compute cumulative dimension-product sums from the head of a map.

	(AI generated docstring)

	You can use `mapShapeProducts` to compute bounds measured from the largest dimension stride toward
	the smallest dimension stride. You receive a tuple beginning with the empty sum.

	Product selection
	-----------------
	The slice endpoint is `dimensionFrom首 - (totalDimensions + 1)`. With the default dimension count, the
	slice retains `mapShapeProducts[:dimensionFrom首]`. The total leaf-count product is excluded at the
	default endpoint.

	Parameters
	----------
	mapShapeProducts : tuple[int, ...]
		Cumulative dimension products, including the initial empty product and total leaf count.
	totalDimensions : int | None = None
		Dimension count. A false-valued value infers `len(mapShapeProducts) - 1`.
	dimensionFrom首 : int | None = None
		Number of leading dimension products to retain. `None` uses `totalDimensions`.

	Returns
	-------
	mapShape首ProductsSums : tuple[int, ...]
		Cumulative sums of the retained products in reverse order, beginning with `0`.

	Examples
	--------
	Elimination-state initialization requests all dimension strides.

		```python
		self.mapShape首ProductsSums = mapFolding.beDRYElimination.getMapShape首ProductsSums(
			self.mapShapeProducts, self.totalDimensions, self.totalDimensions
		)
		```
	"""
	totalDimensions = totalDimensions or len(mapShapeProducts) - 1
	dimensionFrom首 = dimensionFrom首 or totalDimensions
	mapShapeProductsTruncator: int = dimensionFrom首 - (totalDimensions + zeroIndexed)
	mapShapeProductsFrom首: tuple[int, ...] = mapShapeProducts[0:mapShapeProductsTruncator][::-1]
	return tuple(accumulate(mapShapeProductsFrom首, add, initial=0))

@cache
def getTotalLeaves(mapShape: tuple[int, ...]) -> int:
	"""The definitive calculation of the total number of leaves in a map with the given dimensions.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers with the length of each dimension of the map.

	Returns
	-------
	totalLeaves : int
		The definitive total number of leaves in the map.

	Raises
	------
	OverflowError
		If the product of dimensions would exceed the system's maximum integer size. This check
		prevents silent numeric overflow issues that could lead to incorrect results.

	Notes
	-----
	It is impossible to overstate the importance of `totalLeaves` in every algorithm for counting
	folds. Therefore, in this package, this function is the ***only*** permissible way to compute
	`totalLeaves`.

	The total number of leaves is the product of all dimensions in `mapShape`.
	"""
	productDimensions = 1
	for dimension in mapShape:
		#=EndNotes##absurd=
		if dimension > sysMaxsize // productDimensions:
			message: str = f"I received `{dimension = }` in `{mapShape = }`, but the product of the dimensions exceeds the maximum size of an integer on this system."
			raise OverflowError(message)
		productDimensions *= dimension
	return productDimensions

#======== Flow control ================================================

def mapShapeIs2上nDimensions(mapShape: tuple[int, ...], *, youMustBeDimensionsTallToRideThis: int = 3) -> bool:
	"""Check that `mapShape` has enough dimensions and every length is two.

	(AI generated docstring)

	You can use this predicate to select algorithms that require at least a specified number of
	dimensions, with exactly two leaves along every dimension.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths to inspect.
	youMustBeDimensionsTallToRideThis : int = 3
		Minimum number of dimensions required for a match.

	Returns
	-------
	is2上nDimensions : bool
		`True` when the dimension count meets the minimum and every length equals two.

	Examples
	--------
	`mapFolding.basecamp.countFolds` [1] uses a minimum of two dimensions during algorithm selection.

		```python
		mapShapeIs2上nDimensions(mapShape, youMustBeDimensionsTallToRideThis=2)
		```

	References
	----------
	[1] `mapFolding.basecamp.countFolds`

	"""
	return (youMustBeDimensionsTallToRideThis <= len(mapShape)) and all(map((2).__eq__, mapShape))

def mapShapeLengthsAreEqual(mapShape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
	"""Group dimension indices that share a repeated length greater than one.

	(AI generated docstring)

	You can identify dimension groups that can share an ordering restriction. You receive groups in the
	order of the first occurrence of each length in `mapShape`, excluding dimension lengths of one.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths in their original order.

	Returns
	-------
	indicesSameDimensionLength : Iterator[tuple[int, ...]]
		Iterator of ascending index tuples, each containing at least two indices.

	Examples
	--------
	The elimination algorithm applies symmetry restrictions to repeated dimension lengths.

		```python
		for indicesSameDimensionLength in mapShapeLengthsAreEqual(state.mapShape):
			state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
		```
	"""
	return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(filter((1).__lt__, mapShape)))))

#======== Parse parameters ======================================

def defineProcessorLimit(CPUlimit: ConcurrencyLimit, concurrencyPackage: str | None = None) -> int:
	"""Resolve the processor count from `CPUlimit` and the concurrency backend.

	(AI generated docstring)

	You can use this function to convert a processor-usage preference into a worker count. This
	function delegates limit interpretation to `hunterMakesPy.parseParameters.defineConcurrencyLimit` [1]. Selecting `'numba'` also changes the active Numba thread limit [2].

	Processor Bounds
	----------------
	The delegated resolver enforces at least one worker and at most 61 workers on Windows [1].
	An explicit positive count can exceed the available processor count. With `'numba'`, an
	unsupported thread count propagates `ValueError` from the Numba setter [2]. Repeated Numba
	calls use the currently active thread count as the basis for fractions and reservations.

	Parameters
	----------
	CPUlimit : bool | float | int | None
		Please see the documentation in `countFolds` for details. I know it is annoying, but I want to
		be sure you have the most accurate information.
	concurrencyPackage : str | None = None
		`'numba'` uses the current Numba thread count and applies the result globally. Every other
		value uses the default processor count without changing the Numba thread limit.

	Returns
	-------
	concurrencyLimit : int
		Resolved worker count, or the active thread count after applying the Numba limit.

	Examples
	--------
	`mapFolding.basecamp.countFolds` [3] resolves resources before task division.

		```python
		concurrencyLimit = defineProcessorLimit(CPUlimit, settingsPackage.concurrencyPackage)
		```

	References
	----------
	[1] hunterMakesPy `defineConcurrencyLimit`.
		https://context7.com/hunterhogan/huntermakespy
	[2] Numba thread-count controls.
		https://numba.readthedocs.io/en/stable/user/threading-layer.html
	[3] `mapFolding.basecamp.countFolds`

	"""
	match concurrencyPackage:
		case 'numba':
			concurrencyLimit: int = defineConcurrencyLimit(limit=CPUlimit, cpuTotal=get_num_threads())
			set_num_threads(concurrencyLimit)
			concurrencyLimit = get_num_threads()
		case 'multiprocessing' | _:
			concurrencyLimit = defineConcurrencyLimit(limit=CPUlimit)
	return concurrencyLimit

def getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, totalLeaves: int) -> int:
	"""Resolve computation task divisions from `computationDivisions`.

	(AI generated docstring)

	You can use this function to translate a task-count preference into a nonnegative count bounded
	by `totalLeaves`. A result of zero means the caller should run one undivided computation.

	Task Counting
	-------------
	Counts greater than `totalLeaves` would count some foldings more than once.
	`mapFolding.basecamp.countFolds` [1] uses parallel execution only when the resolved count exceeds
	one.

	Parameters
	----------
	computationDivisions : int | str | None
		`None`, zero, and `False` select zero divisions. An `int` selects that count, with negative
		values reduced to zero. `'maximum'` selects `totalLeaves`. `'cpu'` selects the smaller of
		`concurrencyLimit` and `totalLeaves`. String matching ignores capitalization.
	concurrencyLimit : int
		Available worker count used by `'cpu'`.
	totalLeaves : int
		Maximum permitted number of tasks.

	Returns
	-------
	taskDivisions : int
		Resolved nonnegative count. `True` resolves to one through the integer case.

	Raises
	------
	ValueError
		If `computationDivisions` has an unsupported value or type, or the selected count exceeds
		`totalLeaves`.

	References
	----------
	[1] `mapFolding.basecamp.countFolds`

	"""
	taskDivisions = 0
	match computationDivisions:
		case None | 0 | False:
			pass
		case int() as intComputationDivisions:
			taskDivisions = intComputationDivisions
		case str() as strComputationDivisions:
			strComputationDivisions = strComputationDivisions.lower()
			match strComputationDivisions:
				case 'maximum':
					taskDivisions: int = totalLeaves
				case 'cpu':
					taskDivisions = min(concurrencyLimit, totalLeaves)
				case _:
					message: str = f"I received '{strComputationDivisions}' for the parameter, `computationDivisions`, but the string value is not supported."
					raise ValueError(message)
		case _:
			message = f"I received {computationDivisions} for the parameter, `computationDivisions`, but the type {type(computationDivisions).__name__} is not supported."
			raise ValueError(message)

	if taskDivisions > totalLeaves:
		message = (
			f"I derived `{taskDivisions = }`, which is greater than `{totalLeaves = }`, but task divisions cannot exceed the map's "
			"total leaves because that would count folds more than once."
		)
		raise ValueError(message)
	return int(max(0, taskDivisions))

def validateMapShape(mapShape: Sequence[int]) -> tuple[int, ...]:
	"""Convert `mapShape` to ordered, nonnegative integer dimensions.

	(AI generated docstring)

	You can use this function to normalize dimension values before counting foldings. This function
	returns a nonempty tuple in the supplied order and rejects negative dimensions. Zero dimensions
	are accepted for the special case handled by `mapFolding.basecamp.countFolds` [1].

	Dimension Order
	---------------
	This function does not sort the dimensions. Algorithms can depend on the original dimension
	order.

	Parameters
	----------
	mapShape : Sequence[int]
		Dimension lengths. Integer-compatible values are converted by
		`hunterMakesPy.parseParameters.intInnit` [2]; boolean elements are rejected.

	Returns
	-------
	mapShapeValidated : tuple[int, ...]
		Converted dimensions in their original order, including any zero values.

	Raises
	------
	ValueError
		If `mapShape` is empty, contains negative dimensions, or cannot be converted to integers.

	Examples
	--------
	`mapFolding.basecamp.countFolds` [1] validates the supplied dimensions before choosing an algorithm.

		```python
		mapShape = validateMapShape(mapShape)
		```

	References
	----------
	[1] `mapFolding.basecamp.countFolds`

	[2] hunterMakesPy `intInnit`.
		https://context7.com/hunterhogan/huntermakespy
	"""
	mapShapeAsList: list[int] = intInnit(mapShape, 'mapShape', Sequence[int])
	if not mapShapeAsList or any(map((0).__gt__, mapShapeAsList)):
		message: str = f"I received `{mapShape = }`, but I need at least one positive integer dimension."
		raise ValueError(message)

	#=EndNotes##sortingDimensions=
	#Do NOT sort the dimensions.
	return tuple(mapShapeAsList)
