from __future__ import annotations

from functools import cache
from math import log
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.theTypes import DimensionIndex

#======== Using a single-base positional-numeral system as a proxy for Cartesian coordinates =======
# https://en.wikipedia.org/wiki/Positional_notation

# Ideogram pronunciation references:
# https://en.wikipedia.org/wiki/Chinese_numerals
# https://en.wikipedia.org/wiki/Japanese_numerals
# https://en.wikipedia.org/wiki/Korean_numerals
# https://en.wikipedia.org/wiki/Vietnamese_numerals

"""NOTE Do you hate my system of ideographs for powers of 2?

With relatively little effort you could use `astToolkit` (pip install astToolkit) to replace all of the ideographs with
`state.mapShapeProducts[dimensionIndex]`. With `astToolkit`, you create a transformation that you can apply after any update.
"""

_dimensionLength: int = 2  # Hypothetically, change to 3 for 3ⁿ-dimensional maps.

_dimensionIndex: DimensionIndex = 0				# == 0

零: int = _dimensionLength ** _dimensionIndex
"""dimensionIndex = 0. Assign `1` to `dimensionIndex = 0` and `0` to each other `DimensionIndex`. Read as index zero/líng."""

# 一
_base: int = _dimensionLength
_dimensionIndex += 1							# == 1
_power: int = _dimensionIndex
一: int = _base ** _power						# == _dimensionLength ** _dimensionIndex
"""dimensionIndex = 1. Assign `1` to `dimensionIndex = 1` and `0` to each other `DimensionIndex`. Read as index one/yī."""

# 二
_radix: int = _dimensionLength
_dimensionIndex += 1							# == 2
_place_ValueIndex: int = _dimensionIndex
二: int = _radix ** _place_ValueIndex			# == _dimensionLength ** _dimensionIndex
"""dimensionIndex = 2. Assign `1` to `dimensionIndex = 2` and `0` to each other `DimensionIndex`. Read as index two/èr."""

# etc.
三: int = _dimensionLength ** 3
"""dimensionIndex = 3. Assign `1` to `dimensionIndex = 3` and `0` to each other `DimensionIndex`. Read as index three/sān."""
四: int = _dimensionLength ** 4
"""dimensionIndex = 4. Assign `1` to `dimensionIndex = 4` and `0` to each other `DimensionIndex`. Read as index four/sì."""
五: int = _dimensionLength ** 5
"""dimensionIndex = 5. Assign `1` to `dimensionIndex = 5` and `0` to each other `DimensionIndex`. Read as index five/wǔ."""
六: int = _dimensionLength ** 6
"""dimensionIndex = 6. Assign `1` to `dimensionIndex = 6` and `0` to each other `DimensionIndex`. Read as index six/liù."""
七: int = _dimensionLength ** 7
"""dimensionIndex = 7. Assign `1` to `dimensionIndex = 7` and `0` to each other `DimensionIndex`. Read as index seven/qī."""
八: int = _dimensionLength ** 8
"""dimensionIndex = 8. Assign `1` to `dimensionIndex = 8` and `0` to each other `DimensionIndex`. Read as index eight/bā."""
九: int = _dimensionLength ** 9
"""dimensionIndex = 9. Assign `1` to `dimensionIndex = 9` and `0` to each other `DimensionIndex`. Read as index nine/jiǔ."""

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
def 首零(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` (`dimensionIndex = 0`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, index zero
	- shǒu líng
	"""
	return int('1' + '0' * (dimensionsTotal - 1), _dimensionLength)

@cache
def 首零一(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `一` (`dimensionIndex = 0` and `dimensionIndex = 1`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-one
	- shǒu líng yī
	"""
	return int('11' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def 首零一二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `二` (`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-one-two
	- shǒu líng yī èr
	"""
	return int('111' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首零二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `二` (`dimensionIndex = 0` and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-two
	- shǒu líng èr
	"""
	return int('101' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首一(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` (`dimensionIndex = 1`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, index one
	- shǒu yī
	"""
	return int('01' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def 首一二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` and `二` (`dimensionIndex = 1` and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices one-two
	- shǒu yī èr
	"""
	return int('011' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `二` (`dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, index two
	- shǒu èr
	"""
	return int('001' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `三` (`dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, index three
	- shǒu sān
	"""
	return int('0001' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零一二三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, `二`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 1`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-one-two-three
	- shǒu líng yī èr sān
	"""
	return int('1111' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零一三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-one-three
	- shǒu líng yī sān
	"""
	return int('1101' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零二三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `二`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-two-three
	- shǒu líng èr sān
	"""
	return int('1011' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `三` (`dimensionIndex = 0` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-three
	- shǒu líng sān
	"""
	return int('1001' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首一二三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一`, `二`, and `三` (`dimensionIndex = 1`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices one-two-three
	- shǒu yī èr sān
	"""
	return int('0111' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首一三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` and `三` (`dimensionIndex = 1` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices one-three
	- shǒu yī sān
	"""
	return int('0101' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首二三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `二` and `三` (`dimensionIndex = 2` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices two-three
	- shǒu èr sān
	"""
	return int('0011' + '0' * (dimensionsTotal - 4), _dimensionLength)
