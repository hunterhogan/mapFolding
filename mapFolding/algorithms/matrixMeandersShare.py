from __future__ import annotations

from hunterMakesPy import raiseIfNone
from mapFolding.dataBaskets import StateMeanders
from mapFolding.oeis.A400429 import boxOfDiagonals
from mapFolding.synthesized.matrixMeanders.matrixMeandersShare import walkDyckPath
from mapFolding.theTypes import 形ArcCode
from numba import int64, vectorize  # pyright: ignore[reportUnknownVariableType]
from typing import overload, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable
	from mapFolding.theTypes import 形ArrayArcCode, 形ArrayInteger
	from numpy import int64 as numpy_int64
	from typing import Any, Literal, LiteralString
	import pandas

"""Matrix Meanders Goals:
- Extreme abstraction.
- Find operations with latent intermediate arrays and make the intermediate array explicit.
- Reduce or eliminate intermediate arrays and selector arrays.
- Write formulas in prefix notation.
- For each formula, find an equivalent prefix notation formula that never uses the same variable as input more than once: that
	would allow the evaluation of the expression with only a single stack, which saves memory.
- Standardize code as much as possible to create duplicate code.
- Convert duplicate code to procedures.
"""

def integersWide吗(state: StateMeanders, *, arrayMeanders: 形ArrayArcCode | None = None, dataframe: pandas.DataFrame | None = None, fixedSizeMAXIMUMarcCode: bool = False) -> bool:
	"""Check if the largest values are wider than the maximum limits.

	Parameters
	----------
	state : StateMeanders
		The current state of the computation, including `lookupMeanders`.
	dataframe : pandas.DataFrame | None = None
		DataFrame containing 'analyzed' and 'meanders' columns. If provided, use this instead of
		`state.lookupMeanders`.
	fixedSizeMAXIMUMarcCode : bool = False
		Set this to `True` if you cast `state.arcCodeMAXIMUM` to the same fixed size integer type as
		`dtypeArcCode`.

	Returns
	-------
	wider : bool
		True if at least one integer is wider than the fixed-size integers.

	Notes
	-----
	Casting `state.arcCodeMAXIMUM` to a fixed-size 64-bit unsigned integer might cause the flow to be
	a little more complicated because `arcCodeMAXIMUM` is usually 1-bit larger than the `max(arcCode)`
	value.

	If you start the algorithm with very large `arcCode` in your `lookupMeanders` (*i.e.,*
	semi), then the flow will go to a function that does not use fixed size integers. When the
	integers are below the limits (*e.g.,* `bitWidthArcCodeMaximum`), the flow will go to a function
	with fixed size integers. In that case, casting `arcCodeMAXIMUM` to a fixed size merely delays the
	transition from one function to the other by one iteration.

	If you start with small values in `lookupMeanders`, however, then the flow goes to the
	function with fixed size integers and usually stays there until `meanders` is huge, which is near
	the end of the computation. If you cast `arcCodeMAXIMUM` into a 64-bit unsigned integer, however,
	then around `state.boundary == 28`, the bit width of `arcCodeMAXIMUM` might exceed the limit. That
	will cause the flow to go to the function that does not have fixed size integers for a few
	iterations before returning to the function with fixed size integers.
	"""
	if dataframe is not None:
		arcCodeWidest = int(dataframe['analyzed'].max()).bit_length()
		meandersWidest = int(dataframe['meanders'].max()).bit_length()
	elif arrayMeanders is not None:
		arcCodeWidest = int(arrayMeanders.max()).bit_length()
		meandersWidest = int(arrayMeanders.max()).bit_length()
	else:
		arcCodeWidest: int = max(state.lookupMeanders.keys()).bit_length()
		meandersWidest: int = max(state.lookupMeanders.values()).bit_length()

	arcCodeMAXIMUM: int = 0
	if fixedSizeMAXIMUMarcCode:
		arcCodeMAXIMUM = state.arcCodeMAXIMUM

	return (raiseIfNone(state.bitWidthLimitArcCode) < arcCodeWidest
		or raiseIfNone(state.bitWidthLimitMeanders) < meandersWidest
		or raiseIfNone(state.bitWidthLimitArcCode) < arcCodeMAXIMUM
		)

def makeLookupMeanders(kind: Literal['closed', 'meanders', 'semi'] | LiteralString, n: int, boundary: int = 0) -> dict[int, int]:
	"""Create the starting `lookupMeanders` for a matrix meander count.

	You can use this function to build the initial `arcCode` → total meanders count mapping. For `kind
	== 'semi'`, the dictionary has two `arcCode` keys based on the parity of `n` and the magnitude of
	`boundary`. For `kind == 'meanders'`, the dictionary has one `arcCode` based on the parity of `n`.

	Parameters
	----------
	kind : Literal['semi', 'meanders'] | LiteralString
		Which family of meander count to initialize.
	n : int
		The number of times the meander crosses the road.
	boundary : int = `n` - 1
		The boundary from which to start counting.

	Returns
	-------
	lookupMeanders : dict[int, int]
		A dictionary whose keys are `arcCode` and whose values are `1`.

	Raises
	------
	ValueError
		Raised when `kind` is not `'semi'` or `'meanders'`.
	"""
	if not boundary:
		boundary = n - 1

	if kind == 'closed':
		lookupMeanders = {0b1111: 1}  # 0xf
	elif kind == 'meanders':
		if n & 0b1:
			lookupMeanders = {0b1111: 1}  # 0xf
		else:
			lookupMeanders = {0b10110: 1}
	elif kind == 'semi':
		if n == 1:
			lookupMeanders: dict[int, int] = {0b1: 1}
		else:
			if n & 0b1:
				arcCode: int = 0b101
			else:
				arcCode = 0b1
			boxOfArcCodes: list[int] = [(arcCode << 1) | arcCode]
#											   0b 1010 | 0b 0101 = 0b 1111, or 0xf
#											   0b   10 | 0b   01 = 0b   11, or 0x3

			arcCodeMAXIMUM: int = 1 << (2 * boundary + 4)
			while boxOfArcCodes[-1] < (arcCodeMAXIMUM >> 8):
				arcCode = (arcCode << 4) | 0b0101  # e.g., 0b 10000 | 0b 0101 = 0b 10101
				boxOfArcCodes.append((arcCode << 1) | arcCode)  # e.g., 0b 101010 | 0b 1010101 = 0b 111111 = 0x3f
				# Thereafter, append 0b1111 or 0xf, so, e.g., 0x3f, 0x3ff, 0x3fff, 0x3ffff, ...
				# See "research/matrixMeanders/A000682facts.py"
			lookupMeanders: dict[int, int] = dict.fromkeys(boxOfArcCodes, 1)

	else:
		message: str = f"I received `{kind = }` for meander computation, but I don't know that kind."
		raise ValueError(message)

	return lookupMeanders

#================== Dyck Path =====================================================================

@overload
def flipTheExtra_0b1(intWithExtra_0b1: numpy_int64) -> numpy_int64: ...
@overload
def flipTheExtra_0b1(intWithExtra_0b1: 形ArcCode) -> 形ArcCode: ...
@overload
def flipTheExtra_0b1(intWithExtra_0b1: 形ArrayArcCode) -> 形ArrayArcCode: ...
@overload
def flipTheExtra_0b1(intWithExtra_0b1: 形ArrayInteger) -> 形ArrayInteger: ...
@overload
def flipTheExtra_0b1(intWithExtra_0b1: pandas.Series[Any]) -> pandas.Series[Any]: ...
@vectorize([int64(int64), f"{形ArcCode.__name__}({形ArcCode.__name__})"], cache=True, nopython=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def flipTheExtra_0b1(intWithExtra_0b1: int) -> int:  # pyright: ignore[reportInconsistentOverload]
	"""Flip a bit based on Dyck path with a Numba-generated universal function [1].

	You can call `flipTheExtra_0b1` with a `numpy.uint64`, a `numpy.ndarray` [2], or a
	`pandas.Series` [3] that contains the fixed-width arc-code representation.

	Warning
	-------
	The function will loop infinitely if _any_ element does not have a bit that needs flipping.

	Parameters
	----------
	intWithExtra_0b1 : numpy.uint64 | numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] | pandas.Series
		One arc code or a container of arc codes with unbalanced closures.

	Returns
	-------
	flipped : numpy.uint64 | numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] | pandas.Series
		The same scalar or container representation with one bit flipped in each arc code.

	References
	----------
	[1] Numba - Creating NumPy universal functions
		https://numba.readthedocs.io/en/stable/user/vectorize.html
	[2] NumPy - Universal functions
		https://numpy.org/doc/stable/reference/ufuncs.html
	[3] pandas.Series
		https://pandas.pydata.org/docs/reference/api/pandas.Series.html
	"""
	return intWithExtra_0b1 ^ walkDyckPath(intWithExtra_0b1)

#================== Buckets =======================================================================

def getTotalBuckets(state: StateMeanders, totalArcCodes: int = 0) -> int:
	"""Return the allocation length for one transfer-matrix step.

	Parameters
	----------
	state : StateMeanders
		The current transfer-matrix state.
	totalArcCodes : int
		Number of unique input `arcCode` at the current boundary.

	Returns
	-------
	totalBuckets : int
		The exact or estimated number of non-unique `arcCode` rows to allocate.

	Notes
	-----
	TODO remake `getTotalBuckets` from scratch.

	Factors:
		- The starting quantity of `arcCode`.
		- The value(s) of the starting `arcCode`.
		- n
		- parity of n
		- boundary
		- parity of boundary
		- Whether this totalBuckets is increasing, as compared to all of the prior totalBuckets.
		- If increasing, is it exponential or logarithmic?
		- The maximum value.
		- If decreasing, I don't really know the factors.
		- If I know the actual value or if I must estimate it.
		- Intentionally incomplete arcCode to total meanders dictionaries sometimes require more buckets. Crazy, but true.

	Figure out an intelligent flow for so many factors.
	"""
	if state.boundary <= state.n * 2 // 3:
		totalBuckets: int = (355 * totalArcCodes + 99) // 100
	else:
		totalBuckets = totalArcCodes * 2

	return max(totalBuckets, 3000000)

def makeLookupDiagonal(n: int, 次diagonal: int) -> StateMeanders:
	arcCode: int = (1 << (2 * (n - 2 * 次diagonal) + 2)) - 1
	return StateMeanders(n, 'semi', lookupMeanders={arcCode: 1})

def prune(state: StateMeanders) -> StateMeanders:
	"""Accumulate final contributions for dictionary states with known completion counts.

	(AI generated docstring)

	You can use this function between counting iterations to remove entries whose remaining work
	can be evaluated by `mapFolding.oeis.A400429.boxOfDiagonals` [1]. The function preserves each
	removed entry's final contribution in `state.countAddend` [2] and returns the updated state.

	Parameters
	----------
	state : StateMeanders
		Current computation state with arc codes and their subtotals in `state.lookupMeanders`.

	Returns
	-------
	state : StateMeanders
		The same state, with recognized entries removed and their final contributions accumulated.

	See Also
	--------
	`pruneDataFrame`
		Prune an active table stored as a dataframe instead of a dictionary [3].

	Diagonal Selection
	------------------
	The local `boundary` is `state.boundary + 1`. Successive functions in `boxOfDiagonals` match
	diagonals numbered from one through `boundary // 2`, stopping when the formula list ends.
	For diagonal `次diagonal`, the selected key is
	`(1 << (2 * (boundary - 2 * 次diagonal) + 2)) - 1`.

	The function removes the selected key, multiplies the removed subtotal by the diagonal formula
	evaluated at `boundary`, and adds the product to `state.countAddend`. A missing key contributes
	zero. If no dictionary entries remain, the function sets `state.boundary` to zero so iteration
	can stop. Other boundary-derived fields are not refreshed.

	Examples
	--------
	`mapFolding.synthesized.matrixMeanders.prune.count` [4] prunes before iteration and after each
	transfer step.

		```python
		state = prune(state)
		```

	References
	----------
	[1] `mapFolding.oeis.A400429.boxOfDiagonals`

	[2] `mapFolding.dataBaskets.StateMeanders.countAddend`

	[3] `pruneDataFrame`

	[4] `mapFolding.synthesized.matrixMeanders.prune.count`

	"""
	boundary: int = state.boundary + 1

	def removeKnownValue(次diagonal: int, 工: Callable[[int], int]) -> int:
		arcCode: int = (1 << (2 * (boundary - 2 * 次diagonal) + 2)) - 1
		subtotalMeanders: int = state.lookupMeanders.pop(arcCode, 0)
		return 工(boundary) * subtotalMeanders

	state.countAddend += sum(map(removeKnownValue, range(1, boundary // 2 + 1), boxOfDiagonals))
	if not state.lookupMeanders:
		state.boundary = 0
	return state

def pruneDataFrame(state: StateMeanders, dataframeAnalyzed: pandas.DataFrame) -> tuple[StateMeanders, pandas.DataFrame]:
	"""Accumulate final contributions for dataframe rows with known completion counts.

	(AI generated docstring)

	You can use this function after a counting step to remove rows whose remaining contribution
	can be evaluated by `mapFolding.oeis.A400429.boxOfDiagonals` [1]. The function adds those final
	contributions to `state.countAddend` [2] and returns the state with the remaining dataframe.

	Parameters
	----------
	state : StateMeanders
		Current computation state supplying the boundary and accumulating completed contributions.
	dataframeAnalyzed : pandas.DataFrame
		Active rows with an `'analyzed'` column of arc codes and a `'meanders'` column of integer
		subtotals. Repeated arc codes are allowed and their subtotals are combined when removed.

	Returns
	-------
	state : StateMeanders
		The same state with completed contributions added to `state.countAddend` and
		`state.boundary` set to zero if no rows remain.
	dataframeRemaining : pandas.DataFrame
		Unrecognized rows in their original order, with a fresh consecutive index starting at zero.

	See Also
	--------
	`prune`
		Prune active entries stored in `state.lookupMeanders` instead of a dataframe [3].

	Row Removal
	-----------
	The function uses the same diagonal selection as `prune` [3], evaluated at
	`state.boundary + 1`. For each selected arc code, the function converts matching subtotals to
	Python `int`, sums the subtotals, and multiplies the sum by the corresponding diagonal formula.
	A missing arc code contributes zero.

	Boolean selection through `pandas.DataFrame.loc` [4] creates the filtered dataframe. The
	function returns that dataframe after `reset_index(drop=True)` [5], so callers must retain the
	returned dataframe. The function mutates `state.countAddend` and, when empty, `state.boundary`;
	`state.lookupMeanders` and the other boundary-derived fields retain their existing values.

	Examples
	--------
	`mapFolding.synthesized.matrixMeanders.prunePandas.count` [6] retains both returned objects
	after aggregating a transfer step.

		```python
		state, dataframeAnalyzed = pruneDataFrame(state, dataframeAnalyzed)
		```

	References
	----------
	[1] `mapFolding.oeis.A400429.boxOfDiagonals`

	[2] `mapFolding.dataBaskets.StateMeanders.countAddend`

	[3] `prune`

	[4] pandas `DataFrame.loc`.
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
	[5] pandas `DataFrame.reset_index`.
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
	[6] `mapFolding.synthesized.matrixMeanders.prunePandas.count`

	"""
	boundary: int = state.boundary + 1

	def removeKnownValue(次diagonal: int, 工: Callable[[int], int]) -> int:
		nonlocal dataframeAnalyzed
		arcCode: int = (1 << (2 * (boundary - 2 * 次diagonal) + 2)) - 1
		selectorKnown: pandas.Series[bool] = dataframeAnalyzed['analyzed'].eq(arcCode)
		subtotalMeanders: int = sum(map(int, dataframeAnalyzed.loc[selectorKnown, 'meanders']))
		dataframeAnalyzed = dataframeAnalyzed.loc[~selectorKnown]
		return 工(boundary) * subtotalMeanders

	state.countAddend += sum(map(removeKnownValue, range(1, boundary // 2 + 1), boxOfDiagonals))
	if not len(dataframeAnalyzed.index):
		state.boundary = 0
	return state, dataframeAnalyzed.reset_index(drop=True)
