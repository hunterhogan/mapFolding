"""Oft-needed computations or actions, especially for multi-dimensional map folding."""
from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from hunterMakesPy.parseParameters import defineConcurrencyLimit, intInnit
#=SIN= Incomplete typing in `numba`.
from numba import get_num_threads, set_num_threads  # pyright: ignore[reportUnknownVariableType]
from sys import maxsize as sysMaxsize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit

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

#======== map folding ===================================

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
