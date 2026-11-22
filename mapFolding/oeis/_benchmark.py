from __future__ import annotations

from hunterMakesPy import errorL33T
from mapFolding.kitTerminal import printBenchmark, printBenchmarkTriangle
from mapFolding.oeis._metadata import getTriangle, getValuesKnown
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence
	from mapFolding.oeis._theTypes import OEISid

def printEasyRunBenchmark(oeisID: str, n: int, computed: int, timeStart: float, *, ratio: int = 0) -> None:
	"""Print a benchmark comparison line for an OEIS sequence value.

	(AI generated docstring)

	You can use this function to compare `computed` with the stored value at index `n` from
	`mapFolding.oeis.getValuesKnown`. This function writes the index, a colored match flag,
	elapsed seconds, computed value, known value, and optional ratio to standard output.

	Parameters
	----------
	oeisID : str
		Sequence identifier, normalized by the known-value lookup.
	n : int
		Index of the sequence term to compare.
	computed : int
		The computed value to compare against the known OEIS value.
	timeStart : float
		Start time from `time.perf_counter` for elapsed time calculation.
	ratio : int = 0
		Number of decimal places for the known-to-computed ratio. Zero hides the ratio.

	Output Fields
	-------------
	The index precedes a space; subsequent fields are separated by tabs. Matching values use
	`hunterMakesPy.ansiColor.GreenOnBlack`, and mismatches use `YellowOnRed`. An absent term
	uses `-errorL33T` as the known value. When requested, the ratio uses the match color if the
	float result is an integer and the mismatch color otherwise.

	Examples
	--------
	`easyRun/countFolds.py` prints a comparison after counting folds for a sequence term.

		```python
		from mapFolding.oeis import printEasyRunBenchmark

		printEasyRunBenchmark(oeisID, n, totalFolds, timeStart, ratio=0)
		```
	"""
	printBenchmark(n, computed, getValuesKnown(oeisID).get(n, -errorL33T), timeStart, ratio=ratio)

def printEasyRunTriangle(oeisID: OEISid, n: int, computed: Sequence[int], timeStart: float, *, ratio: int = 0) -> None:
	"""Compare the computed triangle row `n` with known sequence cells.

	(AI generated docstring)

	You can use this function to inspect a row computation with its elapsed time. This function
	uses row `n` of `mapFolding.oeis.getTriangle` as the reference cells. The function prints the
	comparison and returns `None`.

	Parameters
	----------
	oeisID : OEISid
		Registered triangle sequence whose metadata defines the row layout.
	n : int
		Row number in the sequence's metadata layout.
	computed : Sequence[int]
		Computed cells in column order, starting with the first column.
	timeStart : float
		Starting value from `time.perf_counter` for the elapsed-time display.
	ratio : int = 0
		Decimal places for known-to-computed ratios on a differing row. Zero hides ratios.

	See Also
	--------
	`printEasyRunBenchmark`
		Compare a single computed sequence term with its known value.

	Output Layout
	-------------
	The first line contains the row number, an exact-match flag, elapsed seconds, and `computed`.
	A match requires equal lengths and equal corresponding values. A differing row prints the
	known cells on a second aligned line. Missing positions in the shorter row display `'?'`.
	An unknown row has no reference cells.

	Ratios are printed only for differing rows and only for paired cells. Zero computed cells
	display `'-'`; integer ratios use the success color even when the ratio differs from
	one. This function does not calculate sequence terms or save results.
	"""
	printBenchmarkTriangle(n, computed, getTriangle(oeisID).get(n, ()), timeStart, ratio=ratio)
