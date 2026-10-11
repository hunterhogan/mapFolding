"""Encode dimension coordinates with index ideographs.

(AI generated docstring)

You can name dimension values and construct coordinate integers for maps whose axis
lengths are `2`. Index ideographs label bit positions rather than the numeric values.
Functions beginning with `首` count positions from the most significant end.

Contents
--------
Functions
	Dimension indices
		dimensionIndex
			Convert an encoded dimension value to a dimension index.
	Head-relative coordinates
		首一
			Encode head-relative dimension positions 1 as an integer.
		首一三
			Encode head-relative dimension positions 1, 3 as an integer.
		首一二
			Encode head-relative dimension positions 1, 2 as an integer.
		首一二三
			Encode head-relative dimension positions 1, 2, 3 as an integer.
		首三
			Encode head-relative dimension positions 3 as an integer.
		首二
			Encode head-relative dimension positions 2 as an integer.
		首二三
			Encode head-relative dimension positions 2, 3 as an integer.
		首零
			Encode head-relative dimension positions 0 as an integer.
		首零一
			Encode head-relative dimension positions 0, 1 as an integer.
		首零一三
			Encode head-relative dimension positions 0, 1, 3 as an integer.
		首零一二
			Encode head-relative dimension positions 0, 1, 2 as an integer.
		首零一二三
			Encode head-relative dimension positions 0, 1, 2, 3 as an integer.
		首零三
			Encode head-relative dimension positions 0, 3 as an integer.
		首零二
			Encode head-relative dimension positions 0, 2 as an integer.
		首零二三
			Encode head-relative dimension positions 0, 2, 3 as an integer.
Variables
	一
		Encode dimension index `1` as an integer.
	七
		Encode dimension index `7` as an integer.
	三
		Encode dimension index `3` as an integer.
	九
		Encode dimension index `9` as an integer.
	二
		Encode dimension index `2` as an integer.
	五
		Encode dimension index `5` as an integer.
	八
		Encode dimension index `8` as an integer.
	六
		Encode dimension index `6` as an integer.
	四
		Encode dimension index `4` as an integer.
	零
		Encode dimension index `0` as an integer.
"""
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

"""Inspect the ideograph system for powers of two.

With relatively little effort you could use `astToolkit` (pip install astToolkit) to replace all of the ideographs with
`state.mapShapeProducts[dimensionIndex]`. With `astToolkit`, you create a transformation that you can apply after any update.
"""

_dimensionLength: int = 2  # Hypothetically, change to 3 for 3ⁿ-dimensional maps.

_dimensionIndex: DimensionIndex = 0				# == 0

零: int = _dimensionLength ** _dimensionIndex
"""Encode dimension index `0` as an integer.

You can use `零` for the value `1`, which sets only bit `0` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `零` as index 0, pronounced líng [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""

# 一
_base: int = _dimensionLength
_dimensionIndex += 1							# == 1
_power: int = _dimensionIndex
一: int = _base ** _power						# == _dimensionLength ** _dimensionIndex
"""Encode dimension index `1` as an integer.

You can use `一` for the value `2`, which sets only bit `1` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `一` as index 1, pronounced yī [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""

# 二
_radix: int = _dimensionLength
_dimensionIndex += 1							# == 2
_place_ValueIndex: int = _dimensionIndex
二: int = _radix ** _place_ValueIndex			# == _dimensionLength ** _dimensionIndex
"""Encode dimension index `2` as an integer.

You can use `二` for the value `4`, which sets only bit `2` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `二` as index 2, pronounced èr [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""

# etc.
三: int = _dimensionLength ** 3
"""Encode dimension index `3` as an integer.

You can use `三` for the value `8`, which sets only bit `3` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `三` as index 3, pronounced sān [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""
四: int = _dimensionLength ** 4
"""Encode dimension index `4` as an integer.

You can use `四` for the value `16`, which sets only bit `4` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `四` as index 4, pronounced sì [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""
五: int = _dimensionLength ** 5
"""Encode dimension index `5` as an integer.

You can use `五` for the value `32`, which sets only bit `5` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `五` as index 5, pronounced wǔ [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""
六: int = _dimensionLength ** 6
"""Encode dimension index `6` as an integer.

You can use `六` for the value `64`, which sets only bit `6` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `六` as index 6, pronounced liù [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""
七: int = _dimensionLength ** 7
"""Encode dimension index `7` as an integer.

You can use `七` for the value `128`, which sets only bit `7` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `七` as index 7, pronounced qī [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""
八: int = _dimensionLength ** 8
"""Encode dimension index `8` as an integer.

You can use `八` for the value `256`, which sets only bit `8` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `八` as index 8, pronounced bā [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""
九: int = _dimensionLength ** 9
"""Encode dimension index `9` as an integer.

You can use `九` for the value `512`, which sets only bit `9` in
the base-2 coordinate encoding [1]. The identifier names a dimension index rather than
the numeric value. Read `九` as index 9, pronounced jiǔ [2].

References
----------
[1] Positional notation.
	https://en.wikipedia.org/wiki/Positional_notation

[2] Chinese numerals.
	https://en.wikipedia.org/wiki/Chinese_numerals
"""

@cache
def dimensionIndex(dimensionAsNonnegativeInteger: int, /, *, dimensionLength: int = _dimensionLength) -> DimensionIndex:
	"""Convert an encoded dimension value to a dimension index.

	You can recover the index represented by `dimensionAsNonnegativeInteger` in a
	positional numeral system [1]. Supply a positive power of `dimensionLength` to represent
	one dimension. The default base is `2`.

	Parameters
	----------
	dimensionAsNonnegativeInteger : int
		Positive encoded dimension value, normally a power of `dimensionLength`.
	dimensionLength : int = 2
		Numeral base. Supply a value greater than `1`.

	Returns
	-------
	index : DimensionIndex
		Dimension index computed by converting the logarithm to `int`.

	Logarithm Conversion
	--------------------
	You receive an index computed with `math.log` [2] and truncated toward zero. Ensure that the input is
	an exact power of `dimensionLength`; no exact-power check is performed. Floating-point rounding can
	affect your result. You can receive `ValueError` for invalid logarithm arguments or
	`ZeroDivisionError` for a base of `1`.

	Examples
	--------
	From `pinByCrease.py`, with the named dimension value already imported.

		```python
		dimensionIndex(二)
		```

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Python `math.log`.
		https://docs.python.org/3/library/math.html#math.log
	"""
	return int(log(dimensionAsNonnegativeInteger, dimensionLength))

#-------- Access the dimension coordinates encoded in a number relative to the number's most significant digit -------

@cache
def 首零(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 0 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 0 to
	`1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首零` as from the head, indices 0,
	pronounced shǒu líng. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `1`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `1` keeps the
	full `1`-digit prefix because multiplying a string by a negative count produces an empty string.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('1' + '0' * (totalDimensions - 1), _dimensionLength)

@cache
def 首零一(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 0, 1 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 0, 1
	to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首零一` as from the head, indices 0, 1,
	pronounced shǒu líng yī. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `2`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `2` keeps the
	full `2`-digit prefix because multiplying a string by a negative count produces an empty string.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('11' + '0' * (totalDimensions - 2), _dimensionLength)

@cache
def 首零一二(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 0, 1, 2 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 0, 1,
	2 to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首零一二` as from the head, indices 0, 1, 2,
	pronounced shǒu líng yī èr. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `3`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `3` keeps the
	full `3`-digit prefix because multiplying a string by a negative count produces an empty string.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('111' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首零二(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 0, 2 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 0, 2
	to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首零二` as from the head, indices 0, 2,
	pronounced shǒu líng èr. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `3`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `3` keeps the
	full `3`-digit prefix because multiplying a string by a negative count produces an empty string.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('101' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首一(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 1 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 1 to
	`1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首一` as from the head, indices 1,
	pronounced shǒu yī. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `2`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `2` keeps the
	full `2`-digit prefix because multiplying a string by a negative count produces an empty string.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('01' + '0' * (totalDimensions - 2), _dimensionLength)

@cache
def 首一二(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 1, 2 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 1, 2
	to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首一二` as from the head, indices 1, 2,
	pronounced shǒu yī èr. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `3`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `3` keeps the
	full `3`-digit prefix because multiplying a string by a negative count produces an empty string.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('011' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首二(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 2 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 2 to
	`1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首二` as from the head, indices 2,
	pronounced shǒu èr. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `3`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `3` keeps the
	full `3`-digit prefix because multiplying a string by a negative count produces an empty string.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('001' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首三(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 3 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 3 to
	`1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首三` as from the head, indices 3,
	pronounced shǒu sān. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `4`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `4` keeps the
	full `4`-digit prefix because multiplying a string by a negative count produces an empty string.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('0001' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零一二三(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 0, 1, 2, 3 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 0, 1,
	2, 3 to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首零一二三` as from the head, indices 0, 1, 2,
	3, pronounced shǒu líng yī èr sān. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `4`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `4` keeps the
	full `4`-digit prefix because multiplying a string by a negative count produces an empty string.

	Examples
	--------
	No usage outside tests was found in the codebase.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('1111' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零一三(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 0, 1, 3 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 0, 1,
	3 to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首零一三` as from the head, indices 0, 1, 3,
	pronounced shǒu líng yī sān. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `4`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `4` keeps the
	full `4`-digit prefix because multiplying a string by a negative count produces an empty string.

	Examples
	--------
	No usage outside tests was found in the codebase.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('1101' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零二三(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 0, 2, 3 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 0, 2,
	3 to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首零二三` as from the head, indices 0, 2, 3,
	pronounced shǒu líng èr sān. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `4`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `4` keeps the
	full `4`-digit prefix because multiplying a string by a negative count produces an empty string.

	Examples
	--------
	No usage outside tests was found in the codebase.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('1011' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零三(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 0, 3 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 0, 3
	to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首零三` as from the head, indices 0, 3,
	pronounced shǒu líng sān. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `4`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `4` keeps the
	full `4`-digit prefix because multiplying a string by a negative count produces an empty string.

	Examples
	--------
	No usage outside tests was found in the codebase.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('1001' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首一二三(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 1, 2, 3 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 1, 2,
	3 to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首一二三` as from the head, indices 1, 2, 3,
	pronounced shǒu yī èr sān. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `4`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `4` keeps the
	full `4`-digit prefix because multiplying a string by a negative count produces an empty string.

	Examples
	--------
	No usage outside tests was found in the codebase.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('0111' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首一三(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 1, 3 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 1, 3
	to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首一三` as from the head, indices 1, 3,
	pronounced shǒu yī sān. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `4`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `4` keeps the
	full `4`-digit prefix because multiplying a string by a negative count produces an empty string.

	Examples
	--------
	No usage outside tests was found in the codebase.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('0101' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首二三(totalDimensions: int, /) -> int:
	"""Encode head-relative dimension positions 2, 3 as an integer.

	You can construct a coordinate value for a map with `totalDimensions` axes by setting positions 2, 3
	to `1` and every other position to `0`. Count these positions from the most significant end. The
	coordinate encoding follows positional notation [1]. Read `首二三` as from the head, indices 2, 3,
	pronounced shǒu èr sān. The index ideographs use Chinese numerals [2].

	Parameters
	----------
	totalDimensions : int
		Number of axes in the map. Supply at least `4`.

	Returns
	-------
	coordinate : int
		Integer encoded by the selected head-relative positions.

	Width Handling
	--------------
	Supply a valid `totalDimensions` without relying on input validation. A value below `4` keeps the
	full `4`-digit prefix because multiplying a string by a negative count produces an empty string.

	Examples
	--------
	No usage outside tests was found in the codebase.

	References
	----------
	[1] Positional notation.
		https://en.wikipedia.org/wiki/Positional_notation

	[2] Chinese numerals.
		https://en.wikipedia.org/wiki/Chinese_numerals
	"""
	return int('0011' + '0' * (totalDimensions - 4), _dimensionLength)
