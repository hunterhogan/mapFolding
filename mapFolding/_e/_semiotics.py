from functools import cache
from hunterMakesPy.parseParameters import intInnit
from mapFolding._e import DimensionIndex
from math import log
from operator import getitem

#======== Using a single-base positional-numeral system as a proxy for Cartesian coordinates =======
# https://en.wikipedia.org/wiki/Positional_notation

# Ideogram pronunciation references:
# https://en.wikipedia.org/wiki/Chinese_numerals
# https://en.wikipedia.org/wiki/Japanese_numerals
# https://en.wikipedia.org/wiki/Korean_numerals
# https://en.wikipedia.org/wiki/Vietnamese_numerals

"""NOTE Do you hate my system of ideographs for powers of 2?

With relatively little effort you could use `astToolkit` (pip install astToolkit) to replace all of the ideographs with
`state.productsOfDimensions[dimensionIndex]`. With `astToolkit`, you create a transformation that you can apply after any update.
"""

_dimensionLength: int	= 2 # Hypothetically, change to 3 for 3^d-dimensional maps.
_dimensionIndex: DimensionIndex = 0						# == 0

零: int = _dimensionLength ** _dimensionIndex
"""dimensionIndex = 0: assign `1` to `dimensionIndex = 0`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index zero
- líng
- ling4
- rei
- yeong
- linh
"""

# 一
_base: int				= _dimensionLength
_dimensionIndex			+= 1					# == 1
_power: int				= _dimensionIndex
一: int = _base ** _power						# == _dimensionLength ** _dimensionIndex
"""dimensionIndex = 1: assign `1` to `dimensionIndex = 1`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index one
- yī
- jat1
- ichi
- il
- nhất
"""

# 二
_radix: int				= _dimensionLength
_dimensionIndex			+= 1					# == 2
_place_ValueIndex: int	= _dimensionIndex
二: int = _radix ** _place_ValueIndex			# == _dimensionLength ** _dimensionIndex
"""dimensionIndex = 2: assign `1` to `dimensionIndex = 2`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index two
- èr
- ji6
- ni
- i
- nhị
"""

# etc.
三: int = _dimensionLength ** 3
"""dimensionIndex = 3: assign `1` to `dimensionIndex = 3`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index three
- sān
- saam1
- san
- sam
- tam
"""
四: int = _dimensionLength ** 4
"""dimensionIndex = 4: assign `1` to `dimensionIndex = 4`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index four
- sì
- sei3
- shi
- sa
- tứ
"""
五: int = _dimensionLength ** 5
"""dimensionIndex = 5: assign `1` to `dimensionIndex = 5`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index five
- wǔ
- ng5
- go
- o
- ngũ
"""
六: int = _dimensionLength ** 6
"""dimensionIndex = 6: assign `1` to `dimensionIndex = 6`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index six
- liù
- luk6
- roku
- yuk
- lục
"""
七: int = _dimensionLength ** 7
"""dimensionIndex = 7: assign `1` to `dimensionIndex = 7`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index seven
- qī
- cat1
- shichi
- chil
- thất
"""
八: int = _dimensionLength ** 8
"""dimensionIndex = 8: assign `1` to `dimensionIndex = 8`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index eight
- bā
- baat3
- hachi
- pal
- bát
"""
九: int = _dimensionLength ** 9
"""dimensionIndex = 9: assign `1` to `dimensionIndex = 9`, and assign `0` to each other `DimensionIndex`. Read as (any of):
- index nine
- jiǔ
- gau2
- kyū
- gu
- cửu
"""

@cache
def dimensionIndex(dimensionAsNonnegativeInteger: int, /, *, dimensionLength: int = _dimensionLength) -> int:
	"""Convert the integer value of a single dimension into its corresponding `DimensionIndex`."""
	dimension: int = getitem(intInnit([dimensionAsNonnegativeInteger], 'dimensionNonnegative', int), 0)
	if dimension < 0:
		message: str = f"I received `{dimensionAsNonnegativeInteger = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	base: int = getitem(intInnit([dimensionLength], 'dimensionLength', int), 0)
	if base < 1:
		message: str = f"I received `{dimensionLength = }`, but I need an integer value greater than 1."
		raise ValueError(message)
	place_ValueIndex: float = log(dimension, base)
	if not place_ValueIndex.is_integer():
		message: str = f"I received `{dimensionAsNonnegativeInteger = }`, but I need a value that is an integer power of `{dimensionLength = }`."
		raise ValueError(message)
	return int(place_ValueIndex)

#-------- Access the dimension coordinates encoded in a number relative to the number's most significant digit -------

@cache
def 首零(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` (`dimensionIndex = 0`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, index zero
	- shǒu líng
	- sau2 ling4
	- shu rei
	- su yeong
	- thủ linh
	"""
	return int('1' + '0' * (dimensionsTotal - 1), _dimensionLength)

@cache
def 首零一(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `一` (`dimensionIndex = 0` and `dimensionIndex = 1`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-one
	- shǒu líng yī
	- sau2 ling4 jat1
	- shu rei ichi
	- su yeong il
	- thủ linh nhất
	"""
	return int('11' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def 首零一二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `二` (`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-one-two
	- shǒu líng yī èr
	- sau2 ling4 jat1 ji6
	- shu rei ichi ni
	- su yeong il i
	- thủ linh nhất nhị
	"""
	return int('111' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首零二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `二` (`dimensionIndex = 0` and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-two
	- shǒu líng èr
	- sau2 ling4 ji6
	- shu rei ni
	- su yeong i
	- thủ linh nhị
	"""
	return int('101' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首一(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` (`dimensionIndex = 1`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, index one
	- shǒu yī
	- sau2 jat1
	- shu ichi
	- su il
	- thủ nhất
	"""
	return int('01' + '0' * (dimensionsTotal - 2), _dimensionLength)

@cache
def 首一二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` and `二` (`dimensionIndex = 1` and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices one-two
	- shǒu yī èr
	- sau2 jat1 ji6
	- shu ichi ni
	- su il i
	- thủ nhất nhị
	"""
	return int('011' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首二(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `二` (`dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, index two
	- shǒu èr
	- sau2 ji6
	- shu ni
	- su i
	- thủ nhị
	"""
	return int('001' + '0' * (dimensionsTotal - 3), _dimensionLength)

@cache
def 首三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `三` (`dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, index three
	- shǒu sān
	- sau2 saam1
	- shu san
	- su sam
	- thủ tam
	"""
	return int('0001' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零一二三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, `二`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 1`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-one-two-three
	- shǒu líng yī èr sān
	- sau2 ling4 jat1 ji6 saam1
	- shu rei ichi ni san
	- su yeong il i sam
	- thủ linh nhất nhị tam
	"""
	return int('1111' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零一三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-one-three
	- shǒu líng yī sān
	- sau2 ling4 jat1 saam1
	- shu rei ichi san
	- su yeong il sam
	- thủ linh nhất tam
	"""
	return int('1101' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零二三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `二`, and `三` (`dimensionIndex = 0`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-two-three
	- shǒu líng èr sān
	- sau2 ling4 ji6 saam1
	- shu rei ni san
	- su yeong i sam
	- thủ linh nhị tam
	"""
	return int('1011' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首零三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `三` (`dimensionIndex = 0` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices zero-three
	- shǒu líng sān
	- sau2 ling4 saam1
	- shu rei san
	- su yeong sam
	- thủ linh tam
	"""
	return int('1001' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首一二三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一`, `二`, and `三` (`dimensionIndex = 1`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices one-two-three
	- shǒu yī èr sān
	- sau2 jat1 ji6 saam1
	- shu ichi ni san
	- su il i sam
	- thủ nhất nhị tam
	"""
	return int('0111' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首一三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` and `三` (`dimensionIndex = 1` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices one-three
	- shǒu yī sān
	- sau2 jat1 saam1
	- shu ichi san
	- su il sam
	- thủ nhất tam
	"""
	return int('0101' + '0' * (dimensionsTotal - 4), _dimensionLength)

@cache
def 首二三(dimensionsTotal: int, /) -> int:
	"""Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `二` and `三` (`dimensionIndex = 2` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `dimensionsTotal`.

	Read as (any of):
	- from the head, indices two-three
	- shǒu èr sān
	- sau2 ji6 saam1
	- shu ni san
	- su i sam
	- thủ nhị tam
	"""
	return int('0011' + '0' * (dimensionsTotal - 4), _dimensionLength)

#======== Semantic replacements for ambiguous values =======

leafOrigin: int = (0 * 九) + (0 * 八) + (0 * 七) + (0 * 六) + (0 * 五) + (0 * 四) + (0 * 三) + (0 * 二) + (0 * 一) + (0 * 零)
"""The `leaf` at the origin of all dimensions, with `0` in every `DimensionIndex`."""
pileOrigin: int = (0 * 九) + (0 * 八) + (0 * 七) + (0 * 六) + (0 * 五) + (0 * 四) + (0 * 三) + (0 * 二) + (0 * 一) + (0 * 零)
"""The `pile` at the origin of all dimensions, with `0` in every `DimensionIndex`."""

sentinelBitAdjustment: int = 1
"""Adjust bit_count() to exclude the sentinel bit in PileRangeOfLeaves."""

