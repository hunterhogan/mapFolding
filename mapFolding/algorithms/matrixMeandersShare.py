from __future__ import annotations

from mapFolding.dataBaskets import StateMeanders
from mapFolding.oeis.A400429oeisByFormula import boxOfDiagonals
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from typing import Literal, LiteralString

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

def integersWide吗(state: StateMeanders) -> bool:
	"""Determine whether the next transfer step exceeds the fixed-width integer capacity.

	(AI generated docstring)

	The current NumPy and pandas formulas need up to four bits beyond the widest `arcCode`.
	Each input produces at most four contributions, so four times the sum of active counts
	bounds every aggregated count in the next step. This bound can select Python integers
	earlier than an exact analysis of each output bucket would.

	`state.bitWidthLimit` counts usable nonnegative value bits. For example, unsigned 64-bit
	integers provide 64 value bits and signed 128-bit integers provide 127. The boundary
	threshold `state.arcCodeMaximum` remains a Python integer and consumes no buffer capacity.
	"""
	bitWidthArcCode: int = max(state.lookupMeanders, default=0).bit_length() + 4
	bitWidthMeanders: int = (4 * sum(state.lookupMeanders.values())).bit_length()
	return state.bitWidthLimit < max(bitWidthArcCode, bitWidthMeanders)

def makeLookupMeanders(kind: Literal['closed', 'meanders', 'semi'] | LiteralString, n: int, boundary: int = 0) -> dict[int, int]:
	"""Create the starting `lookupMeanders` for a matrix meander count.

	(AI generated docstring)

	You can use this function to build the initial mapping from arch codes to counts. For
	`kind == 'semi'`, the number and values of the starting codes depend on `n` and `boundary`.
	For `kind == 'meanders'`, one starting code is selected using the parity of `n`.
	For `kind == 'closed'`, one fixed starting code is used.

	Parameters
	----------
	kind : Literal['closed', 'meanders', 'semi'] | LiteralString
		Which family of meander count to initialize.
	n : int
		The number of times the meander crosses the road.
	boundary : int = 0
		The boundary from which to start counting. Zero selects `n - 1`.

	Returns
	-------
	lookupMeanders : dict[int, int]
		A dictionary whose keys are `arcCode` and whose values are `1`.

	Raises
	------
	ValueError
		Raised when `kind` is not `'closed'`, `'semi'`, or `'meanders'`.
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

			arcCodeMaximum: int = 1 << (2 * boundary + 4)
			while boxOfArcCodes[-1] < (arcCodeMaximum >> 8):
				arcCode = (arcCode << 4) | 0b0101  # e.g., 0b 10000 | 0b 0101 = 0b 10101
				boxOfArcCodes.append((arcCode << 1) | arcCode)  # e.g., 0b 101010 | 0b 1010101 = 0b 111111 = 0x3f
				# Thereafter, append 0b1111 or 0xf, so, e.g., 0x3f, 0x3ff, 0x3fff, 0x3ffff, ...
				# See "research/matrixMeanders/A000682facts.py"
			lookupMeanders: dict[int, int] = dict.fromkeys(boxOfArcCodes, 1)

	else:
		message: str = f"I received `{kind = }` for meander computation, but I don't know that kind."
		raise ValueError(message)

	return lookupMeanders

#================== Buckets =======================================================================

totalBucketsMaximumBy_boundary: dict[int, int] = {1: 3, 2: 12, 3: 40, 4: 125, 5: 392, 6: 1254
	, 7: 4087, 8: 13623, 9: 46181, 10: 159137, 11: 555469, 12: 1961369, 13: 6991893, 14: 25134208}

def getTotalBuckets(state: StateMeanders, totalArcCodes: int = 0) -> int:
	"""Estimate the output allocation length for one transfer-matrix step.

	(AI generated docstring)

	You can use this function to choose an initial output-buffer length from the number of input
	records and the transfer boundary. The estimate can be smaller than the required allocation;
	callers that write additional records must allow the buffer to grow.

	Parameters
	----------
	state : StateMeanders
		The current transfer-matrix state.
	totalArcCodes : int = 0
		Number of unique input `arcCode` at the current boundary.

	Returns
	-------
	totalBuckets : int
		The exact or estimated number of non-unique `arcCode` rows to allocate.

	Notes
	-----
	TODO remake `getTotalBuckets` from scratch.

	Factors:
		- [ ] Intentionally incomplete arcCode to total meanders dictionaries sometimes require more
			buckets. Crazy, but true. Can `research/matrixMeanders/fixed_deficit_semimeanders.py` help
			me find a formula for this?
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

	Figure out an intelligent flow for so many factors.
	"""
	totalBuckets: int = (355 * totalArcCodes + 99) // 100
	return max(totalBuckets, totalBucketsMaximumBy_boundary.get(state.boundary, totalBuckets))

def makeLookupDiagonal(n: int, 次diagonal: int) -> StateMeanders:
	"""Prepare a semi-meander state for counting diagonal `次diagonal` at row `n`.

	(AI generated docstring)

	You can use this function to initialize the transfer calculation for one triangle cell.
	The returned `StateMeanders` starts with one arch code and count one. Counting has not
	yet run.

	Parameters
	----------
	n : int
		Semi-meander row number in the domain of the selected diagonal.
	次diagonal : int
		Diagonal offset from the complete right edge, starting at one.

	Returns
	-------
	state : StateMeanders
		Initial state for the selected diagonal term, ready for
		`mapFolding.basecamp.countMeanders`.

	See Also
	--------
	`mapFolding.basecamp.countMeanders`
		Run the prepared diagonal state's counting calculation.

	Examples
	--------
	`easyRun/triangleSemiMake.py` counts a diagonal term before saving the total.

		```python
		state = countMeanders(state=makeLookupDiagonal(n, 次diagonal))
		```

	"""
	arcCode: int = (1 << (2 * (n - 2 * 次diagonal) + 2)) - 1
	return StateMeanders(n, 'semi', lookupMeanders={arcCode: 1})

def prune(state: StateMeanders) -> StateMeanders:
	"""Accumulate final contributions for dictionary states with known completion counts.

	(AI generated docstring)

	You can use this function between counting iterations to remove entries whose remaining work
	can be evaluated by `mapFolding.oeis.A400429oeisByFormula.boxOfDiagonals`. The function preserves
	each removed entry's final contribution in `state.countAddend` and returns the updated state.

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
	`mapFolding.oeis.A400429oeisByFormula.boxOfDiagonals`
		Inspect the known diagonal formulas used to complete selected dictionary entries.
	`mapFolding.synthesized.matrixMeanders.matrixMeandersShare.pruneDataFrame`
		Prune an active table stored as a dataframe instead of a dictionary.

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
	`mapFolding.synthesized.matrixMeanders.prune.count` prunes before iteration and after each
	transfer step.

		```python
		state = prune(state)
		```

	"""
	boundary: int = state.boundary + 1

	次diagonal: int = 1
	while 次diagonal <= min(boundary // 2, len(boxOfDiagonals)):
		arcCode: int = (1 << (2 * (boundary - 2 * 次diagonal) + 2)) - 1
		subtotalMeanders: int = state.lookupMeanders.pop(arcCode, 0)
		state.countAddend += boxOfDiagonals[次diagonal - 1](boundary) * subtotalMeanders
		次diagonal += 1
	if not state.lookupMeanders:
		state.boundary = 0
	return state
