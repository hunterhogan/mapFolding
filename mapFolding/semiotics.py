"""Use a single-base positional-numeral system as a proxy for Cartesian coordinates.

(AI generated docstring)

Each `DimensionIndex` identifies a position in a single-base positional-numeral system [1]. The
index ideographs assign `1` to the named position and `0` to each other `DimensionIndex`.
The base is `_dimensionLength`, which is `2` for maps whose dimension lengths are all `2`.

Functions beginning with `首` enumerate each `DimensionIndex` starting from the most significant
digit, within `totalDimensions` positions.

Contents
--------
Functions
	Dimension indices
		dimensionIndex
			Convert the integer value of a position to its `DimensionIndex`.
	Positions enumerated from the head
		首一
			Assign `1` to index 1 from the head.
		首一三
			Assign `1` to indices 1 and 3 from the head.
		首一二
			Assign `1` to indices 1 and 2 from the head.
		首一二三
			Assign `1` to indices 1, 2, and 3 from the head.
		首三
			Assign `1` to index 3 from the head.
		首二
			Assign `1` to index 2 from the head.
		首二三
			Assign `1` to indices 2 and 3 from the head.
		首零
			Assign `1` to index 0 from the head.
		首零一
			Assign `1` to indices 0 and 1 from the head.
		首零一三
			Assign `1` to indices 0, 1, and 3 from the head.
		首零一二
			Assign `1` to indices 0, 1, and 2 from the head.
		首零一二三
			Assign `1` to indices 0, 1, 2, and 3 from the head.
		首零三
			Assign `1` to indices 0 and 3 from the head.
		首零二
			Assign `1` to indices 0 and 2 from the head.
		首零二三
			Assign `1` to indices 0, 2, and 3 from the head.
Variables
	一
		Assign `1` to `dimensionIndex = 1`.
	七
		Assign `1` to `dimensionIndex = 7`.
	三
		Assign `1` to `dimensionIndex = 3`.
	九
		Assign `1` to `dimensionIndex = 9`.
	二
		Assign `1` to `dimensionIndex = 2`.
	五
		Assign `1` to `dimensionIndex = 5`.
	八
		Assign `1` to `dimensionIndex = 8`.
	六
		Assign `1` to `dimensionIndex = 6`.
	四
		Assign `1` to `dimensionIndex = 4`.
	零
		Assign `1` to `dimensionIndex = 0`.

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation
"""
from __future__ import annotations

from functools import cache
from math import log
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import DimensionIndex

#======== Using a single-base positional-numeral system as a proxy for Cartesian coordinates =======
# https://en.wikipedia.org/wiki/Positional_notation

"""NOTE Do you hate my system of ideographs for powers of 2?

With relatively little effort you could use `astToolkit` (pip install astToolkit) to replace all of the ideographs with
`state.mapShapeProducts[dimensionIndex]`. With `astToolkit`, you create a transformation that you can apply after any update.
"""

_dimensionLength: int = 2  # Hypothetically, change to 3 for 3ⁿ-dimensional maps.

_dimensionIndex: DimensionIndex = 0				# == 0

零: int = _dimensionLength ** _dimensionIndex
"""dimensionIndex = 0. Assign `1` to `dimensionIndex = 0` and `0` to each other `DimensionIndex`."""

# 一
_base: int = _dimensionLength
_dimensionIndex += 1							# == 1
_power: int = _dimensionIndex
一: int = _base ** _power						# == _dimensionLength ** _dimensionIndex
"""dimensionIndex = 1. Assign `1` to `dimensionIndex = 1` and `0` to each other `DimensionIndex`."""

# 二
_radix: int = _dimensionLength
_dimensionIndex += 1							# == 2
_place_ValueIndex: int = _dimensionIndex
二: int = _radix ** _place_ValueIndex			# == _dimensionLength ** _dimensionIndex
"""dimensionIndex = 2. Assign `1` to `dimensionIndex = 2` and `0` to each other `DimensionIndex`."""

# etc.
三: int = _dimensionLength ** 3
"""dimensionIndex = 3. Assign `1` to `dimensionIndex = 3` and `0` to each other `DimensionIndex`."""
四: int = _dimensionLength ** 4
"""dimensionIndex = 4. Assign `1` to `dimensionIndex = 4` and `0` to each other `DimensionIndex`."""
五: int = _dimensionLength ** 5
"""dimensionIndex = 5. Assign `1` to `dimensionIndex = 5` and `0` to each other `DimensionIndex`."""
六: int = _dimensionLength ** 6
"""dimensionIndex = 6. Assign `1` to `dimensionIndex = 6` and `0` to each other `DimensionIndex`."""
七: int = _dimensionLength ** 7
"""dimensionIndex = 7. Assign `1` to `dimensionIndex = 7` and `0` to each other `DimensionIndex`."""
八: int = _dimensionLength ** 8
"""dimensionIndex = 8. Assign `1` to `dimensionIndex = 8` and `0` to each other `DimensionIndex`."""
九: int = _dimensionLength ** 9
"""dimensionIndex = 9. Assign `1` to `dimensionIndex = 9` and `0` to each other `DimensionIndex`."""

@cache
def dimensionIndex(dimensionAsNonnegativeInteger: int, /, *, dimensionLength: int = _dimensionLength) -> DimensionIndex:
	"""In a single-base positional-numeral system, convert the integer value of a position to its `DimensionIndex`.

	Returns
	-------
	index: DimensionIndex
		The `DimensionIndex` corresponding to the provided dimension value.
	"""
	return int(log(dimensionAsNonnegativeInteger, dimensionLength))

#-------- Access the dimension coordinates encoded in a number relative to the number's most significant digit -------

@cache
def 首零(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` (`dimensionIndex = 0`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('1' + '0' * (totalDimensions - 1), _dimensionLength)

@cache
def 首零一(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `一` (`dimensionIndex = 0` and `dimensionIndex = 1`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('11' + '0' * (totalDimensions - 2), _dimensionLength)

@cache
def 首零一二(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `二` (`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('111' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首零二(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `二` (`dimensionIndex = 0` and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('101' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首一(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` (`dimensionIndex = 1`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('01' + '0' * (totalDimensions - 2), _dimensionLength)

@cache
def 首一二(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` and `二` (`dimensionIndex = 1` and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('011' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首二(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `二` (`dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('001' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首三(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `三` (`dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('0001' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零一二三(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, `二`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 1`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('1111' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零一三(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('1101' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零二三(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `二`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('1011' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零三(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `三` (`dimensionIndex = 0` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('1001' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首一二三(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一`, `二`, and `三` (`dimensionIndex = 1`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('0111' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首一三(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` and `三` (`dimensionIndex = 1` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('0101' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首二三(totalDimensions: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `二` and `三` (`dimensionIndex = 2` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`."""
	return int('0011' + '0' * (totalDimensions - 4), _dimensionLength)
