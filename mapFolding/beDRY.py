"""Compute Catalan numbers, dot products, and map-shape arithmetic.

(AI generated docstring)

You can compute Catalan numbers and dot products, including weighted sums of Catalan numbers. You
can compute dimension products and sums for a map shape. You can also validate map dimensions and
resolve processor limits and task divisions without importing computation-state containers.

Contents
--------
Catalan numbers and dot products
	catalan
		Compute the Catalan number at an ordinal as a Python integer.
	dotProduct
		Sum the products of paired integers from two iterables.
	dotProductCatalan
		Sum integer multipliers against Catalan numbers in decreasing ordinal order.
	工boxOfBasisCatalan
		Iterate over Catalan numbers at ordinals selected with range bounds.

Maps and leaves
	getMapShapeProducts
		Compute cumulative products of the dimension lengths in a map shape.
	getMapShapeProductsSums
		Compute cumulative sums of a map shape's dimension products.
	getMapShape首ProductsSums
		Compute cumulative dimension-product sums from the head of a map.
	getTotalLeaves
		Compute the definitive total number of leaves for a map shape.

Flow control
	mapShapeIs2上nDimensions
		Check that a map has enough dimensions and every dimension length is two.
	mapShapeLengthsAreEqual
		Group dimension indices that share a repeated length greater than one.

Parse parameters
	defineProcessorLimit
		Resolve the processor count from a resource preference and concurrency backend.
	getTaskDivisions
		Resolve the number of computation tasks within the total leaf count.
	validateMapShape
		Convert supplied dimensions to ordered, nonnegative integers.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache, partial
from humpy_cytoolz import unique
from hunterMakesPy import decreasing, zeroIndexed
from hunterMakesPy.parseParameters import defineConcurrencyLimit, intInnit
from itertools import accumulate
from more_itertools import iter_index
from numba import get_num_threads, set_num_threads
from operator import add, mul
from sympy import catalan as sympy_catalan
from sys import maxsize as sysMaxsize
from typing import SupportsIndex, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterator
	from hunterMakesPy import ConcurrencyLimit

def 工boxOfBasisCatalan(start: SupportsIndex, stop: SupportsIndex | None = None, step: SupportsIndex = 1) -> Iterable[int]:
	"""Iterate over Catalan numbers at ordinals selected with range bounds.

	(AI generated docstring)

	You can obtain a Catalan basis for a dot product. The ordinal bounds follow `range`: one bound
	selects ordinals from zero up to that bound, excluding the bound. Catalan numbers are evaluated as
	you consume the returned iterator.

	Parameters
	----------
	start : SupportsIndex
		First ordinal when `stop` is supplied; otherwise, the exclusive bound after a start of zero.
	stop : SupportsIndex | None = None
		Exclusive ordinal bound. `None` uses `start` as the bound and begins at zero.
	step : SupportsIndex = 1
		Nonzero ordinal increment. A negative increment selects ordinals in decreasing order.

	Returns
	-------
	boxOfBasisCatalan : Iterable[int]
		Single-use iterator of Python integers. Bounds that select no ordinals give an empty iterator.

	See Also
	--------
	`catalan`
		Evaluate one Catalan number.
	`dotProductCatalan`
		Sum multipliers against Catalan numbers in decreasing ordinal order.

	Examples
	--------
	`mapFolding.algorithms.matrixMeandersShare` weights the Catalan numbers at ordinals one through
	`boundary + 1`.

		```python
		totalStatesMaximum: int = dotProduct(
			(1 + min(k - (boundary + 1) % 2, boundary + 1 - k) for k in range(1, boundary + 2))
			, 工boxOfBasisCatalan(1, boundary + 2), strict=True)
		```
	"""
	if stop is None:
		stop = start
		start = 0
	return map(catalan, range(start, stop, step))

def catalan(n: int) -> int:
	"""Compute the Catalan number at ordinal `n` as a Python integer.

	(AI generated docstring)

	You can use any integer `n`. Negative ordinals give zero.

	Mathematics
	-----------
	Catalan number : equation

	```math
		Let n ≜ `n`
		n ∈ ℤ

		C(n) = (2n)! / ((n + 1)! n!)    n ≥ 0
		C(n) = 0                        n < 0

		`catalan` ≜ C(n)
	```

	Parameters
	----------
	n : int
		Ordinal of the Catalan number.

	Returns
	-------
	catalan : int
		Exact Catalan number converted to a Python integer.

	See Also
	--------
	`工boxOfBasisCatalan`
		Iterate over Catalan numbers at ordinals selected with range bounds.

	Examples
	--------
	`mapFolding.algorithms.matrixMeandersShare` subtracts a Catalan count from a bucket bound.

		```python
		totalBuckets -= 4 * catalan(distance // 2)
		```
	"""
	return int(sympy_catalan(n))

def dotProduct(boxOfMultiplicands: Iterable[int], boxOfMultipliers: Iterable[int], *, strict: bool = False) -> int:
	"""Sum the products of paired integers from two iterables.

	(AI generated docstring)

	You can compute a dot product of two integer sequences. The first multiplicand pairs with the
	first multiplier, the second multiplicand pairs with the second multiplier, and so on.

	Parameters
	----------
	boxOfMultiplicands : Iterable[int]
		Integers to multiply, in pairing order.
	boxOfMultipliers : Iterable[int]
		Integer weights, in pairing order.
	strict : bool = False
		`True` raises `ValueError` when the iterables have different lengths. `False` ignores the
		unpaired values of the longer iterable.

	Returns
	-------
	dotProduct : int
		Sum of the paired products, or zero when either iterable is empty and `strict` is `False`.

	See Also
	--------
	`dotProductCatalan`
		Sum multipliers against Catalan numbers in decreasing ordinal order.

	Examples
	--------
	`mapFolding.algorithms.matrixMeandersShare` evaluates a linear formula in `state.n`.

		```python
		totalBucketsKnown = dotProduct(coefficientsBuckets, (state.n // 2, 1, state.n % 2), strict=True)
		```
	"""
	return sum(map(mul, boxOfMultiplicands, boxOfMultipliers, strict=strict))

def dotProductCatalan(ordinal: int, boxOfMultipliers: Sequence[int], shift: int = 0, *, strict: bool = False) -> int:
	"""Sum integer multipliers against Catalan numbers in decreasing ordinal order.

	(AI generated docstring)

	You can evaluate a weighted Catalan sum. The first multiplier pairs with the Catalan number at
	`ordinal + shift`, and each subsequent multiplier pairs with the next lower ordinal. Ordinals
	below zero contribute zero.

	Mathematics
	-----------
	Weighted Catalan sum : equation

	```math
		Let o ≜ `ordinal`,  s ≜ `shift`,  (b₀, …, bₘ₋₁) ≜ `boxOfMultipliers`
		o ∈ ℤ,  s ∈ ℕ₀,  b ∈ ℤᵐ

		C(n) = (2n)! / ((n + 1)! n!)    n ≥ 0
		C(n) = 0                        n < 0
		S = Σⱼ₌₀ᵐ⁻¹ bⱼ · C(o + s − j)

		`dotProductCatalan` ≜ S
	```

	Parameters
	----------
	ordinal : int
		Catalan ordinal paired with the first multiplier when `shift` is zero.
	boxOfMultipliers : Sequence[int]
		Integer weights in the order to pair with decreasing Catalan ordinals.
	shift : int = 0
		Offset added to every Catalan ordinal. A negative `shift` leaves the final `-shift`
		multipliers unpaired.
	strict : bool = False
		`True` raises `ValueError` when `shift` is not zero. `False` permits every `shift`.

	Returns
	-------
	dotProductCatalan : int
		Weighted sum S.

	See Also
	--------
	`dotProduct`
		Sum the products of paired integers from two iterables.
	`工boxOfBasisCatalan`
		Select Catalan ordinals with explicit range bounds.

	Examples
	--------
	`Z0Z_research.catalanArches.formulasParents` counts a triangle cell from its row coefficients.

		```python
		return dotProductCatalan(row_n, boxOfCoefficients, strict=True)
		```
	"""
	return dotProduct(工boxOfBasisCatalan(ordinal + shift, ordinal - len(boxOfMultipliers), decreasing), boxOfMultipliers, strict=strict)

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
		self.mapShapeProducts = mapFolding.beDRY.getMapShapeProducts(self.mapShape)
		```
	"""
	return tuple(accumulate(mapShape, mul, initial=1))

def getMapShapeProductsSums(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""Compute cumulative sums of a map shape's dimension products.

	(AI generated docstring)

	You can sum the cumulative products from `getMapShapeProducts`. You receive a tuple that includes
	the initial empty sum and the sum including the total leaf count.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths in their original order.

	Returns
	-------
	mapShapeProductsSums : tuple[int, ...]
		Cumulative sums beginning with `0`, with length `len(mapShape) + 2`.

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
		self.mapShape首ProductsSums = mapFolding.beDRY.getMapShape首ProductsSums(
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
	`mapFolding.basecamp.countFolds` uses a minimum of two dimensions during algorithm selection.

		```python
		mapShapeIs2上nDimensions(mapShape, youMustBeDimensionsTallToRideThis=2)
		```

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
	function delegates limit interpretation to `hunterMakesPy.parseParameters.defineConcurrencyLimit`.
	Selecting `'numba'` also changes the active Numba thread limit [1].

	Processor Bounds
	----------------
	The delegated resolver enforces at least one worker and at most 61 workers on Windows [2].
	An explicit positive count can exceed the available processor count. With `'numba'`, an
	unsupported thread count propagates `ValueError` from the Numba setter [1]. Repeated Numba
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
	`mapFolding.basecamp.countFolds` resolves resources before task division.

		```python
		concurrencyLimit = defineProcessorLimit(CPUlimit, settingsPackage.concurrencyPackage)
		```

	References
	----------
	[1] Numba thread-count controls: Setting the Number of Threads.
		https://numba.readthedocs.io/en/stable/user/threading-layer.html#setting-the-number-of-threads
	[2] Python `concurrent.futures.ProcessPoolExecutor`, Windows worker limit.
		https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor

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
	`mapFolding.basecamp.countFolds` uses parallel execution only when the resolved count exceeds
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
	are accepted for the special case handled by `mapFolding.basecamp.countFolds`.

	Dimension Order
	---------------
	This function does not sort the dimensions. Algorithms can depend on the original dimension
	order.

	Parameters
	----------
	mapShape : Sequence[int]
		Dimension lengths. Integer-compatible values are converted by
		`hunterMakesPy.parseParameters.intInnit`; boolean elements are rejected.

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
	`mapFolding.basecamp.countFolds` validates the supplied dimensions before choosing an algorithm.

		```python
		mapShape = validateMapShape(mapShape)
		```

	"""
	mapShapeAsList: list[int] = intInnit(mapShape, 'mapShape', Sequence[int])
	if not mapShapeAsList or any(map((0).__gt__, mapShapeAsList)):
		message: str = f"I received `{mapShape = }`, but I need at least one positive integer dimension."
		raise ValueError(message)

	#=EndNotes##sortingDimensions=
	#Do NOT sort the dimensions.
	return tuple(mapShapeAsList)
