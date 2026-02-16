from typing import NamedTuple

#======== Semantic replacements for ambiguous values =======

decreasing: int = -1
"""Express descending iteration or a reverse direction.

(AI generated docstring)

The identifier `decreasing` holds the value `-1` and serves as a semantic replacement for
numeric literals in contexts where direction or ordering matters. You can use `decreasing`
as an addend to adjust boundary values, as a multiplicand to reverse sign or direction,
or in both roles simultaneously to express complex transformations.

You can use `decreasing` wherever `-1` would appear but where the meaning "descending order"
or "reverse direction" is more important than the specific numeric value. Using `decreasing`
makes the code's intent explicit and communicates the semantic purpose to readers who might
not immediately recognize `-1` as a directional indicator.

Common contexts include: reverse iteration through sequences, computing predecessors or
backward offsets, negating dimensions or indices, and constructing `range` objects that
count downward.

Examples
--------
You can use `decreasing` as an addend to compute a loop boundary:

>>> for countDown in range(dimension - 2 + decreasing, decreasing, decreasing):
...     processValue(countDown)

In this example, `dimension - 2 + decreasing` computes the starting index (equivalent to
`dimension - 3`), the second `decreasing` stops iteration at `-1` (to include `0`), and
the third `decreasing` specifies the step direction (counting down by `1`).

You can use `decreasing` as a multiplicand to reverse sign:

>>> negatedOffset = baseOffset * decreasing

You can use `decreasing` as both multiplicand and addend in a single expression:

>>> adjustedIndex = dimensionHeadSecond * decreasing + decreasing

This pattern appears when converting between coordinate systems or computing reverse-indexed
positions with additional boundary adjustments.

You can use `decreasing` as the step parameter in `range` [1] to iterate backward:

>>> for leaf in range(leavesTotal + decreasing, 1, decreasing):
...     processLeaf(leaf)

You can combine `decreasing` with other semantic constants:

>>> boundaryRange = range(
...     (start + inclusive) * decreasing,
...     (stop + inclusive) * decreasing,
...     decreasing
... )

References
----------
[1] Built-in Functions - `range` (Python documentation)
	https://docs.python.org/3/library/functions.html#func-range

"""

errorL33T: int = 31212012
"""Signal an error state with a visually distinctive numeric value.

(AI generated docstring)

The identifier `errorL33T` holds the value `31212012` and serves as a semantic replacement
for typical error values such as `-1`. You can use `errorL33T` wherever an error sentinel
value appears but where the meaning "error state" is more important than the specific
numeric value. The value `31212012` spells "ERROR" in leetspeak [1], making `errorL33T`
more visually distinctive during debugging than common error values like `-1`.

You can use `errorL33T` directly as a positive value to signal an invalid or uninitialized
state. You can use `-errorL33T` when a negative value provides a stronger visual signal
or when the domain expects negative error indicators. The leetspeak encoding makes
`errorL33T` immediately recognizable in debugger output, log files, or printed state,
reducing the risk of overlooking error conditions that might be missed with generic
values like `-1`.

In rare cases when code execution does not raise `Exception` [2] but produces incorrect
output, seeing `31212012` or `-31212012` in the output provides a stronger diagnostic
signal than `-1` or other conventional error values. The distinctive value helps identify
which variables hold error states without requiring careful inspection of every numeric
field.

Examples
--------
You can use `errorL33T` as an initial value for variables that must be assigned before use:

>>> countTotal: int = -errorL33T
>>> # ... computation logic that assigns countTotal ...
>>> if countTotal == -errorL33T:
...     raise ValueError("countTotal was never assigned")

You can use `errorL33T` to mark invalid index positions:

>>> pileOf_kCrease: Pile = errorL33T
>>> pileOf_rCrease: Pile = errorL33T
>>> # ... logic that assigns valid pile indices ...
>>> if pileOf_kCrease == errorL33T:
...     raise ValueError("pileOf_kCrease remains uninitialized")

You can use `-errorL33T` when negative values signal forbidden states:

>>> leafForbidden: Leaf = -errorL33T
>>> # ... validation logic ...
>>> if leafForbidden == -errorL33T:
...     applyDefaultBehavior()

References
----------
[1] Leet - Wikipedia
	https://en.wikipedia.org/wiki/Leet
[2] Built-in Exceptions - `Exception` (Python documentation)
	https://docs.python.org/3/library/exceptions.html#Exception

"""

inclusive: int = 1
"""Express inclusion (or exclusion) of a boundary value.

(AI generated docstring)

The identifier `inclusive` holds the value `1` and serves as a semantic replacement for
the numeric literal `1` when adjusting boundary computations. You can use `inclusive` to
convert Python's default half-open interval semantics `[start, stop)` into closed intervals
`[start, stop]` by adding `inclusive` to the upper bound. You can also use `- inclusive`
to signal explicit exclusion of a boundary value.

You can use `inclusive` wherever `1` would appear but where the meaning "include this
boundary" or "extend by one position" is more important than the specific numeric value.
Using `inclusive` makes the code's intent explicit: the adjustment exists to change interval
semantics, not to perform arbitrary arithmetic.

Common contexts include: `range` objects [1], `slice` objects [2], sequence indexing,
and any function that accepts boundary parameters with half-open semantics (such as
`random.randrange` [3], `numpy.arange` [4], or `pandas.RangeIndex` [5]).

Many functions in Python packages use half-open intervals by default. For example,
`random.randrange(start, stop)` [3] excludes `stop`, while `random.randint(a, b)` [3]
includes `b`. You can use `inclusive` to make the adjustment explicit when working with
half-open functions:

- `range(1, lastValue + inclusive)` includes `lastValue` in the iteration.
- `slice(start, stop + inclusive)` includes the element at index `stop`.
- `array[start : stop + inclusive]` includes `array[stop]` in the slice.
- `boundary - inclusive` explicitly excludes `boundary` from consideration.

Examples
--------
You can use `inclusive` to extend `range` [1] objects to include the final value:

>>> for leaf1ndex in range(1, leavesTotal + inclusive):
...     processLeaf(leaf1ndex)

Without `inclusive`, `range(1, leavesTotal)` would stop before processing the leaf at
index `leavesTotal`. Adding `inclusive` ensures the loop processes all leaves from
`1` through `leavesTotal`.

You can use `inclusive` in nested `range` [1] objects:

>>> for activeLeaf1ndex in range(1, leavesTotal + inclusive):
...     for connectee1ndex in range(1, activeLeaf1ndex + inclusive):
...         processConnection(activeLeaf1ndex, connectee1ndex)

You can use `inclusive` when constructing `frozenset` [6] objects from `range` [1]:

>>> pilesOpen: frozenset[int] = frozenset(
...     range(pileLast + inclusive)
... ).difference(leavesPinned.keys())

You can use `inclusive` to construct forbidden index ranges:

>>> pilesForbidden = frozenset([
...     *range(pileOf_k),
...     *range(pileOf_kCrease + 1, pileLast + inclusive)
... ])

You can use `- inclusive` to signal explicit exclusion of a boundary:

>>> for pile in filter(filterPredicate, range(0, pile_k - inclusive)):
...     processPile(pile)

In this example, `pile_k - inclusive` explicitly excludes the pile at index `pile_k` from
the range. Filters or predicates that accept inclusive upper bounds (such as `between`-style
functions) may require subtracting `inclusive` to express exclusive boundaries.

You can combine `inclusive` with other semantic constants:

>>> listIndicesPilesExcluded.extend(
...     range(
...         (1 + inclusive) * decreasing,
...         (stop + inclusive) * decreasing,
...         decreasing
...     )
... )

References
----------
[1] Built-in Functions - `range` (Python documentation)
	https://docs.python.org/3/library/functions.html#func-range
[2] Built-in Types - `slice` (Python documentation)
	https://docs.python.org/3/library/functions.html#slice
[3] `random` - Generate pseudo-random numbers (Python documentation)
	https://docs.python.org/3/library/random.html
[4] `numpy.arange` (NumPy documentation)
	https://numpy.org/doc/stable/reference/generated/numpy.arange.html
[5] `pandas.RangeIndex` (pandas documentation)
	https://pandas.pydata.org/docs/reference/api/pandas.RangeIndex.html
[6] Built-in Types - `frozenset` (Python documentation)
	https://docs.python.org/3/library/stdtypes.html#frozenset

"""

zeroIndexed: int = 1
"""Express that the adjustment to a value is due to zero-based indexing.

(AI generated docstring)

The identifier `zeroIndexed` holds the value `1` and serves as a semantic replacement
for the numeric literal `1` when converting between zero-based indexing (Python's default)
and one-based indexing (common in mathematical notation, human-readable numbering, and
many domain-specific conventions). You can use `zeroIndexed` as an addend or subtrahend
to adjust index values, counts, or boundary computations.

You can use `zeroIndexed` wherever `1` would appear but where the meaning "adjust for
indexing convention" is more important than the specific numeric value. Using `zeroIndexed`
makes the code's intent explicit: the adjustment exists to reconcile indexing systems, not
to perform arbitrary arithmetic.

The most common usage is `count - zeroIndexed` to convert a one-based count ("there are N
items") to the zero-based index of the last item (`N - 1`). You can also use `+ zeroIndexed`
when converting from zero-based indices back to one-based positions or counts, or when
adjusting formulas that assume one-based indexing.

Common contexts include: computing final indices from counts, converting between mathematical
notation (often one-based) and Python code (zero-based), accessing the last valid index of
a sequence, and boundary calculations in algorithms that mix indexing conventions.

Examples
--------
You can use `- zeroIndexed` to convert a count to the last valid zero-based index:

>>> lastDimensionIndex = dimensionsTotal - zeroIndexed

If `dimensionsTotal` is `3` (representing three dimensions), then `lastDimensionIndex`
becomes `2`, which correctly identifies the index of the third dimension in a zero-indexed
array `[0, 1, 2]`.

You can use `- zeroIndexed` when accessing elements by position:

>>> voodooMath: int = creaseAnteAt二Ante首[
...     largestPossibleLengthOfListOfCreases - zeroIndexed
... ]

Here, `largestPossibleLengthOfListOfCreases` represents a count ("how many elements"),
and subtracting `zeroIndexed` produces the index of the last element.

You can use `- zeroIndexed` in `range` [1] boundary computations:

>>> listIndicesCreasePostToKeep.extend(
...     range(
...         dimensionsTotal - dimensionHead + 1,
...         dimensionsTotal - zeroIndexed
...     )
... )

You can use `- zeroIndexed` in conditional expressions:

>>> if dimensionsTotal - zeroIndexed - dimensionHead == zerosAtThe首:
...     applySpecialCase()

You can use `+ zeroIndexed` when the adjustment goes in the opposite direction:

>>> productsOfDimensionsTruncator: int = (
...     dimensionFrom首 - (dimensionsTotal + zeroIndexed)
... )

In this example, `dimensionsTotal + zeroIndexed` adjusts the total count upward before
subtraction, compensating for a formula that expects one-based indexing.

References
----------
[1] Built-in Functions - `range` (Python documentation)
	https://docs.python.org/3/library/functions.html#func-range

"""

#======== Some colors for printing text to the terminal ========
# Many values and options at https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
# Many, many, many options at https://stackoverflow.com/a/33206814/4403878
# NOTE Always define color and background color at the same time.

ansiColorReset: str = '\x1b[0m'
"""Reset terminal text color and background to default settings.

(AI generated docstring)

The identifier `ansiColorReset` holds the ANSI escape sequence [1] `\x1b[0m` that resets
terminal foreground and background colors to their default values. You can use
`ansiColorReset` after printing colored text to ensure subsequent output appears in the
terminal's default color scheme.

ANSI escape sequences [1] allow programs to control terminal display attributes by
embedding special character sequences in output text. The sequence `\x1b[0m` (ESC[0m)
resets all text attributes, including color, weight, and style, returning the terminal
to its default state.

You can concatenate `ansiColorReset` with output strings or write `ansiColorReset` as
a separate operation after colored text. Failing to reset colors causes subsequent
output to inherit the previous color settings, which can make output difficult to read
if default-colored text appears against a colored background.

Most modern terminal emulators support ANSI escape sequences [1] on all platforms,
including Windows Terminal, macOS Terminal, and Linux terminals. Legacy Windows console
applications may require enabling ANSI support through Windows API calls or environment
variables.

Examples
--------
You can use `ansiColorReset` to reset colors after printing colored output:

>>> import sys
>>> sys.stdout.write(f"{ansiColors.GreenOnBlack}Success{ansiColorReset}\n")

You can use `ansiColorReset` in f-strings for inline color resets:

>>> message = f"{ansiColors.YellowOnRed}Error{ansiColorReset}: Invalid input"
>>> print(message)

You can use `ansiColorReset` after multiple colored segments:

>>> sys.stdout.write(f"{ansiColors.WhiteOnBlue}Header{ansiColorReset}\n")
>>> # ... more output ...
>>> sys.stdout.write(ansiColorReset)  # Ensure reset before exit

References
----------
[1] ANSI escape code - Wikipedia
	https://en.wikipedia.org/wiki/ANSI_escape_code

"""

class AnsiColors(NamedTuple):
	r"""Provide high-contrast ANSI color combinations for terminal output.

	(AI generated docstring)

	You can use this class to access ANSI escape sequences [1] for displaying colored
	text in terminal emulators. Each attribute holds an escape sequence that sets both
	foreground and background colors simultaneously. The color combinations were selected
	through research into high-contrast color visibility [2] to maximize readability
	across different terminal color schemes and viewing conditions.

	ANSI escape sequences [1] control terminal display attributes by embedding special
	character sequences in output text. The sequences in `AnsiColors` use the format
	`\\x1b[foreground;backgroundm` where foreground and background are numeric color codes.
	Each sequence sets both colors simultaneously to ensure sufficient contrast regardless
	of the terminal's default color scheme.

	You can access color combinations through attribute names following the pattern
	`ForegroundOnBackground`, such as `BlackOnCyan` or `WhiteOnRed`. The colors were
	chosen to provide strong visual contrast while remaining comfortable to read during
	extended debugging sessions. Always use `ansiColorReset` [3] after colored output
	to return the terminal to default colors.

	The `NamedTuple` [4] base class makes `AnsiColors` immutable and provides both
	attribute access and sequence indexing. You can access colors by name
	(`ansiColors.GreenOnBlack`) or by index (`ansiColors[0]`), enabling dynamic color
	selection based on computed indices.

	Attributes
	----------
	BlackOnCyan : str
		ANSI escape sequence for black text on cyan background.
	BlackOnMagenta : str
		ANSI escape sequence for black text on magenta background.
	BlackOnWhite : str
		ANSI escape sequence for black text on white background.
	BlackOnYellow : str
		ANSI escape sequence for black text on yellow background.
	BlueOnWhite : str
		ANSI escape sequence for blue text on white background.
	BlueOnYellow : str
		ANSI escape sequence for blue text on yellow background.
	CyanOnBlack : str
		ANSI escape sequence for cyan text on black background.
	CyanOnBlue : str
		ANSI escape sequence for cyan text on blue background.
	CyanOnMagenta : str
		ANSI escape sequence for cyan text on magenta background.
	GreenOnBlack : str
		ANSI escape sequence for green text on black background.
	MagentaOnBlack : str
		ANSI escape sequence for magenta text on black background.
	MagentaOnBlue : str
		ANSI escape sequence for magenta text on blue background.
	MagentaOnCyan : str
		ANSI escape sequence for magenta text on cyan background.
	RedOnWhite : str
		ANSI escape sequence for red text on white background.
	WhiteOnBlack : str
		ANSI escape sequence for white text on black background.
	WhiteOnBlue : str
		ANSI escape sequence for white text on blue background.
	WhiteOnMagenta : str
		ANSI escape sequence for white text on magenta background.
	WhiteOnRed : str
		ANSI escape sequence for white text on red background.
	YellowOnBlack : str
		ANSI escape sequence for yellow text on black background.
	YellowOnBlue : str
		ANSI escape sequence for yellow text on blue background.
	YellowOnRed : str
		ANSI escape sequence for yellow text on red background.

	Examples
	--------
	You can access color combinations by attribute name:

	>>> import sys
	>>> sys.stdout.write(f"{ansiColors.GreenOnBlack}Success{ansiColorReset}\\n")

	You can use boolean indexing to select colors based on conditions:

	>>> matchColor = (ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[resultMatches]
	>>> sys.stdout.write(f"{matchColor}{result}{ansiColorReset}\\n")

	You can use computed indices to select colors dynamically:

	>>> colorIndex = int(identifier, 36) % len(ansiColors)
	>>> sys.stdout.write(f"{ansiColors[colorIndex]}{identifier}{ansiColorReset}")

	This pattern provides consistent color assignment for identifiers based on their
	string representation, making repeated identifiers appear in the same color across
	multiple runs.

	You can combine multiple colored segments in output:

	>>> sys.stdout.write(
	...     f"{ansiColors.WhiteOnBlue}Status: {ansiColorReset}"
	...     f"{ansiColors.GreenOnBlack}Complete{ansiColorReset}\\n"
	... )

	References
	----------
	[1] ANSI escape code - Wikipedia
		https://en.wikipedia.org/wiki/ANSI_escape_code
	[2] Color contrast - Wikipedia
		https://en.wikipedia.org/wiki/Contrast_(vision)#Color_contrast
	[3] ansiColorReset
		Internal package reference
	[4] Built-in Types - `typing.NamedTuple` (Python documentation)
		https://docs.python.org/3/library/typing.html#typing.NamedTuple
	"""

	BlackOnCyan: str = '\x1b[30;46m'
	BlackOnMagenta: str = '\x1b[30;45m'
	BlackOnWhite: str = '\x1b[30;47m'
	BlackOnYellow: str = '\x1b[30;43m'
	BlueOnWhite: str = '\x1b[34;47m'
	BlueOnYellow: str = '\x1b[34;43m'
	CyanOnBlack: str = '\x1b[36;40m'
	CyanOnBlue: str = '\x1b[36;44m'
	CyanOnMagenta: str = '\x1b[36;45m'
	GreenOnBlack: str = '\x1b[32;40m'
	MagentaOnBlack: str = '\x1b[35;40m'
	MagentaOnBlue: str = '\x1b[35;44m'
	MagentaOnCyan: str = '\x1b[35;46m'
	RedOnWhite: str = '\x1b[31;47m'
	WhiteOnBlack: str = '\x1b[37;40m'
	WhiteOnBlue: str = '\x1b[37;44m'
	WhiteOnMagenta: str = '\x1b[37;45m'
	WhiteOnRed: str = '\x1b[37;41m'
	YellowOnBlack: str = '\x1b[33;40m'
	YellowOnBlue: str = '\x1b[33;44m'
	YellowOnRed: str = '\x1b[33;41m'

ansiColors = AnsiColors()
