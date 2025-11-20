from functools import cache
from typing import Final

# ======= Using a single-base positional-numeral system as a proxy for Cartesian coordinates =======
# https://en.wikipedia.org/wiki/Positional_notation

_dimensionLength: int	= 2
_dimensionIndex:  int	= 0								# == 0

零: Final[int] = _dimensionLength ** _dimensionIndex

# 一
_base: int				= _dimensionLength
_dimensionIndex			+= 1							# == 1
_power: int				= _dimensionIndex
一: Final[int] = _base ** _power						# == _dimensionLength ** _dimensionIndex

# 二
_radix: int				= _dimensionLength
_dimensionIndex			+= 1							# == 2
_place_ValueIndex: int	= _dimensionIndex
二: Final[int] = _radix ** _place_ValueIndex			# == _dimensionLength ** _dimensionIndex

# etc.
三: Final[int] = _dimensionLength ** 3
四: Final[int] = _dimensionLength ** 4
五: Final[int] = _dimensionLength ** 5
六: Final[int] = _dimensionLength ** 6
七: Final[int] = _dimensionLength ** 7
八: Final[int] = _dimensionLength ** 8
九: Final[int] = _dimensionLength ** 9

# ------- Access the dimension coordinates encoded in a number relative to the number's most significant digit -------

@cache
def 首零(dimensionsTotal: int, /) -> int:
	"""Enumerate each `dimensionIndex` starting from the head `首`, assign `1` to `零` (`dimensionIndex = 0`), and assign `0` to each other `dimensionIndex` in `dimensionsTotal`."""
	return int('1' + '0' * (dimensionsTotal - 1), _dimensionLength)

@cache
def 首一(dimensionsTotal: int, /) -> int:
	"""Enumerate each `dimensionIndex` starting from the head `首`, assign `1` to `一` (`dimensionIndex = 1`), and assign `0` to each other `dimensionIndex` in `dimensionsTotal`."""
	return int('01' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def 首零一(dimensionsTotal: int, /) -> int:
	"""Enumerate each `dimensionIndex` starting from the head `首`, assign `1` to `零` and `一` (`dimensionIndex = 0` and `dimensionIndex = 1`), and assign `0` to each other `dimensionIndex` in `dimensionsTotal`."""
	return int('11' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def 首二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `dimensionIndex` starting from the head `首`, assign `1` to `二` (`dimensionIndex = 2`), and assign `0` to each other `dimensionIndex` in `dimensionsTotal`."""
	return int('001' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首零一二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `dimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `二` (`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 2`), and assign `0` to each other `dimensionIndex` in `dimensionsTotal`."""
	return int('111' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `dimensionIndex` starting from the head `首`, assign `1` to `三` (`dimensionIndex = 3`), and assign `0` to each other `dimensionIndex` in `dimensionsTotal`."""
	return int('0001' + '0' * (dimensionsTotal - 4), _dimensionLength)

# ======= Semantic replacements for ambiguous values =======

decreasing: int = -1
indexLeaf0: Final[int] = 0
origin: Final[int] = (0 * 九) + (0 * 八) + (0 * 七) + (0 * 六) + (0 * 五) + (0 * 四) + (0 * 三) + (0 * 二) + (0 * 一) + (0 * 零)
