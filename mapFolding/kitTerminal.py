"""Print benchmark comparisons and labels with reproducible terminal colors.

(AI generated docstring)

You can use this module to compare a computed value with a known value, one term or one row at a
time, and to label a benchmark run. Colors come from `hunterMakesPy.ansiColor`.

Contents
--------
Functions
	formatBenchmarkRatio
		Format the ratio of a known value to a computed value with a match color.
	printBenchmark
		Print computed and known values with elapsed time and an optional ratio.
	printBenchmarkTriangle
		Compare a computed row with known cells and align differing rows.
	printEasyRunHeader
		Print two labels with reproducible terminal colors.
"""
from __future__ import annotations

from functools import partial
from hunterMakesPy import ansiColor, ansiColorReset
from itertools import chain, repeat
from operator import eq
from typing import TYPE_CHECKING
import anyascii
import sys
import time

if TYPE_CHECKING:
	from collections.abc import Sequence

def formatBenchmarkRatio(known: int, computed: int, ratio: int) -> str:
	"""Format `known / computed` with `ratio` decimal places and a color that shows whether the quotient is an integer.

	(AI generated docstring)

	Parameters
	----------
	known : int
		The known value, the numerator.
	computed : int
		The computed value, the denominator.
	ratio : int
		Number of decimal places. Zero formats nothing.

	Returns
	-------
	formatted : str
		An empty string when `ratio` is zero, `'-'` when `computed` is zero, and otherwise the quotient.
		An integer quotient uses `hunterMakesPy.ansiColor.GreenOnBlack`, and any other quotient uses `YellowOnRed`.
	"""
	formatted: str = ''
	if ratio:
		formatted = '-'
		if computed:
			quotient: float = known / computed
			formatted = f"{(ansiColor.YellowOnRed, ansiColor.GreenOnBlack)[quotient.is_integer()]}{quotient:.{ratio}f}{ansiColorReset}"
	return formatted

def printBenchmark(n: int, computed: int, known: int, timeStart: float, *, ratio: int = 0) -> None:
	"""Print a benchmark comparison line for the term `n`.

	(AI generated docstring)

	You can use this function to compare `computed` with `known`. This function writes the index, a
	colored match flag, elapsed seconds, computed value, known value, and optional ratio to standard
	output.

	Parameters
	----------
	n : int
		Index of the term to compare.
	computed : int
		The computed value.
	known : int
		The known value to compare with `computed`.
	timeStart : float
		Start time from `time.perf_counter` for elapsed time calculation.
	ratio : int = 0
		Number of decimal places for the known-to-computed ratio. Zero hides the ratio.

	Output Fields
	-------------
	The index precedes a space; subsequent fields are separated by tabs. Matching values use
	`hunterMakesPy.ansiColor.GreenOnBlack`, and mismatches use `YellowOnRed`. When requested,
	the ratio uses the match color if the float result is an integer and the mismatch color
	otherwise.
	"""
	match: bool = computed == known
	sys.stdout.write(
		f"{n:2} "
		f"{(ansiColor.YellowOnRed, ansiColor.GreenOnBlack)[match]}{match}{ansiColorReset}\t"
		f"{time.perf_counter() - timeStart:5.2f}\t"
		f"{computed}\t{known}\t"
	)
	if ratio and computed:
		sys.stdout.write(formatBenchmarkRatio(known, computed, ratio) + '\t')
	sys.stdout.write(f"{ansiColorReset}\n")

def printBenchmarkTriangle(n: int, computed: Sequence[int], known: Sequence[int], timeStart: float, *, ratio: int = 0) -> None:
	"""Compare the computed row `n` with the known cells of row `n`.

	(AI generated docstring)

	You can use this function to inspect a row computation with its elapsed time. This function prints
	the comparison and returns `None`.

	Parameters
	----------
	n : int
		Row number.
	computed : Sequence[int]
		Computed cells in column order, starting with the first column.
	known : Sequence[int]
		Known cells in column order, starting with the first column. An unknown row has no known cells.
	timeStart : float
		Starting value from `time.perf_counter` for the elapsed-time display.
	ratio : int = 0
		Decimal places for known-to-computed ratios on a differing row. Zero hides ratios.

	See Also
	--------
	`printBenchmark`
		Compare a single computed term with its known value.

	Output Layout
	-------------
	The first line contains the row number, an exact-match flag, elapsed seconds, and `computed`.
	A match requires equal lengths and equal corresponding values. A differing row prints the
	known cells on a second aligned line. Missing positions in the shorter row display `'?'`.

	Ratios are printed only for differing rows and only for paired cells. Zero computed cells
	display `'-'`; integer ratios use the success color even when the ratio differs from
	one. This function does not calculate sequence terms or save results.
	"""
	match: bool = len(computed) == len(known) and all(map(eq, computed, known, strict=True))
	timeElapsed: float = time.perf_counter() - timeStart

	sys.stdout.write(f"{n:2} {(ansiColor.YellowOnRed, ansiColor.GreenOnBlack)[match]}{match}{ansiColorReset}\t{timeElapsed:5.2f}\t")

	width: int = max(map(len, map(str, chain(computed, known))), default=1)
	sys.stdout.write(' '.join(map(str.rjust, chain(map(str, computed), repeat('?', max(0, len(known) - len(computed)))), repeat(width))))

	if not match:
		if ratio:
			sys.stdout.write('\t' + ' '.join(map(partial(formatBenchmarkRatio, ratio=ratio), known, computed, strict=False)))
		sys.stdout.write('\n' + ' ' * len(f"{n:2} {match}\t{timeElapsed:5.2f}\t".expandtabs()))
		sys.stdout.write(' '.join(map(str.rjust, chain(map(str, known), repeat('?', max(0, len(computed) - len(known)))), repeat(width))))

	sys.stdout.write(f"{ansiColorReset}\n")

def printEasyRunHeader(title: str | int, subtitle: str | int) -> None:
	"""Print a title and subtitle with reproducible terminal colors.

	(AI generated docstring)

	You can use this function to label a benchmark run with `title` and `subtitle`. This function
	writes the original labels to standard output and selects each color from the corresponding
	ASCII transliteration supplied by `anyascii`.

	Parameters
	----------
	title : str | int
		First label. Its ASCII transliteration must be accepted as a base-36 integer.
	subtitle : str | int
		Second label, subject to the same base-36 conversion requirement as `title`.

	Label Conversion
	----------------
	This function propagates `ValueError` if either label's ASCII transliteration cannot be
	converted to a base-36 integer.

	Color Selection
	---------------
	Each transliterated label is converted with `int` using base 36. The result modulo the
	number of entries in `hunterMakesPy.ansiColor` selects the color. Labels can select the same
	color. This function resets terminal colors after printing both labels and a newline.

	Examples
	--------
	`easyRun/countFolds.py` labels each run with the sequence identifier and algorithm selector.

		```python
		from mapFolding.kitTerminal import printEasyRunHeader

		printEasyRunHeader(oeisID, flow)
		```
	"""
	sys.stdout.write(f"{ansiColor[int(anyascii.anyascii(str(title)), 36) % len(ansiColor)]}{title} ")
	sys.stdout.write(f"{ansiColor[int(anyascii.anyascii(str(subtitle)), 36) % len(ansiColor)]} {subtitle} ")
	sys.stdout.write(ansiColorReset + '\n')
