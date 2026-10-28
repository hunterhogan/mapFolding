"""Provide origins and positional coordinate symbols for maps with dimension lengths of two.

(AI generated docstring)

The positional symbols and head-relative constructors use positional notation in base two.
The accompanying `_semiotics.md` explains the coordinate notation; `_symbols.md` provides a
symbol reference.
Both guides are installed beside this module.

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
	Origins
		leafOrigin
			Identify the leaf at the origin.
		pileOrigin
			Identify the pile at the origin.
	Positions enumerated from the tail
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

"""
from __future__ import annotations

from functools import cache
from math import log
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import DimensionIndex, Leaf, Pile
	from typing import Final

#======== Origins identify the leaf and pile at zero ========

leafOrigin: Final[Leaf] = 0
"""The `leaf` at the origin of all dimensions, with `0` in every `DimensionIndex`."""

pileOrigin: Final[Pile] = 0
"""Identify the `pile` with stack index `0`."""

#======== Tail-relative symbols select one dimension coordinate ========

_dimensionLength: int = 2

_dimensionIndex: DimensionIndex = 0				# == 0

零: int = _dimensionLength ** _dimensionIndex
"""dimensionIndex = 0. Assign `1` to `dimensionIndex = 0` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""

#-------- Tail position one --------
_base: int = _dimensionLength
_dimensionIndex += 1							# == 1
_power: int = _dimensionIndex
一: int = _base ** _power						# == _dimensionLength ** _dimensionIndex
"""dimensionIndex = 1. Assign `1` to `dimensionIndex = 1` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""

#-------- Tail position two --------
_radix: int = _dimensionLength
_dimensionIndex += 1							# == 2
_place_ValueIndex: int = _dimensionIndex
二: int = _radix ** _place_ValueIndex			# == _dimensionLength ** _dimensionIndex
"""dimensionIndex = 2. Assign `1` to `dimensionIndex = 2` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""

#-------- Tail positions three through nine --------
三: int = _dimensionLength ** 3
"""dimensionIndex = 3. Assign `1` to `dimensionIndex = 3` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""
四: int = _dimensionLength ** 4
"""dimensionIndex = 4. Assign `1` to `dimensionIndex = 4` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""
五: int = _dimensionLength ** 5
"""dimensionIndex = 5. Assign `1` to `dimensionIndex = 5` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""
六: int = _dimensionLength ** 6
"""dimensionIndex = 6. Assign `1` to `dimensionIndex = 6` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""
七: int = _dimensionLength ** 7
"""dimensionIndex = 7. Assign `1` to `dimensionIndex = 7` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""
八: int = _dimensionLength ** 8
"""dimensionIndex = 8. Assign `1` to `dimensionIndex = 8` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""
九: int = _dimensionLength ** 9
"""dimensionIndex = 9. Assign `1` to `dimensionIndex = 9` and `0` to each other `DimensionIndex`.

Count dimension positions from the tail.
"""

#======== Recover the tail index of a single-dimension coordinate ========

@cache
def dimensionIndex(dimensionAsNonnegativeInteger: int, /, *, dimensionLength: int = _dimensionLength) -> DimensionIndex:
	"""Recover a tail-relative `DimensionIndex` from a single-dimension coordinate.

	(AI generated docstring)

	Parameters
	----------
	dimensionAsNonnegativeInteger : int
		Positive positional value with coordinate `1` in one dimension and `0` elsewhere.
	dimensionLength : int = 2
		Base of the positional value. The package's coordinate symbols use two.

	Returns
	-------
	index : DimensionIndex
		Tail-relative index computed from the base-`dimensionLength` logarithm.
	"""
	return int(log(dimensionAsNonnegativeInteger, dimensionLength))

#======== Head-relative constructors select coordinates within totalDimensions ========

#-------- Head positions zero through two --------

@cache
def 首零(totalDimensions: int, /) -> int:
	"""Select head position zero within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`
	(`dimensionIndex = 0`), and assign `0` to each other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 1.
	"""
	return int('1' + '0' * (totalDimensions - 1), _dimensionLength)

@cache
def 首零一(totalDimensions: int, /) -> int:
	"""Select head positions zero and one within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `一`
	(`dimensionIndex = 0` and `dimensionIndex = 1`), and assign `0` to each other `DimensionIndex` in
	`totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 2.
	"""
	return int('11' + '0' * (totalDimensions - 2), _dimensionLength)

@cache
def 首零一二(totalDimensions: int, /) -> int:
	"""Select head positions zero, one, and two within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `二`
	(`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 2`), and assign `0` to each
	other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 3.
	"""
	return int('111' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首零二(totalDimensions: int, /) -> int:
	"""Select head positions zero and two within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `二`
	(`dimensionIndex = 0` and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in
	`totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 3.
	"""
	return int('101' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首一(totalDimensions: int, /) -> int:
	"""Select head position one within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一`
	(`dimensionIndex = 1`), and assign `0` to each other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 2.
	"""
	return int('01' + '0' * (totalDimensions - 2), _dimensionLength)

@cache
def 首一二(totalDimensions: int, /) -> int:
	"""Select head positions one and two within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` and `二`
	(`dimensionIndex = 1` and `dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in
	`totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 3.
	"""
	return int('011' + '0' * (totalDimensions - 3), _dimensionLength)

@cache
def 首二(totalDimensions: int, /) -> int:
	"""Select head position two within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `二`
	(`dimensionIndex = 2`), and assign `0` to each other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 3.
	"""
	return int('001' + '0' * (totalDimensions - 3), _dimensionLength)

#-------- Head position three and combinations containing it --------

@cache
def 首三(totalDimensions: int, /) -> int:
	"""Select head position three within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `三`
	(`dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 4.
	"""
	return int('0001' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零一二三(totalDimensions: int, /) -> int:
	"""Select head positions zero, one, two, and three within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, `二`, and `三`
	(`dimensionIndex = 0`, `dimensionIndex = 1`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and
	assign `0` to each other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 4.
	"""
	return int('1111' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零一三(totalDimensions: int, /) -> int:
	"""Select head positions zero, one, and three within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `一`, and `三`
	(`dimensionIndex = 0`, `dimensionIndex = 1`, and `dimensionIndex = 3`), and assign `0` to each
	other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 4.
	"""
	return int('1101' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零二三(totalDimensions: int, /) -> int:
	"""Select head positions zero, two, and three within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零`, `二`, and `三`
	(`dimensionIndex = 0`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each
	other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 4.
	"""
	return int('1011' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首零三(totalDimensions: int, /) -> int:
	"""Select head positions zero and three within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `零` and `三`
	(`dimensionIndex = 0` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in
	`totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 4.
	"""
	return int('1001' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首一二三(totalDimensions: int, /) -> int:
	"""Select head positions one, two, and three within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一`, `二`, and `三`
	(`dimensionIndex = 1`, `dimensionIndex = 2`, and `dimensionIndex = 3`), and assign `0` to each
	other `DimensionIndex` in `totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 4.
	"""
	return int('0111' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首一三(totalDimensions: int, /) -> int:
	"""Select head positions one and three within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `一` and `三`
	(`dimensionIndex = 1` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in
	`totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 4.
	"""
	return int('0101' + '0' * (totalDimensions - 4), _dimensionLength)

@cache
def 首二三(totalDimensions: int, /) -> int:
	"""Select head positions two and three within the map width.

	(AI generated docstring)

	Enumerate each `DimensionIndex` starting from the head `首`, assign `1` to `二` and `三`
	(`dimensionIndex = 2` and `dimensionIndex = 3`), and assign `0` to each other `DimensionIndex` in
	`totalDimensions`.

	Parameters
	----------
	totalDimensions : int
		Number of dimension positions, including zero coordinates; at least 4.
	"""
	return int('0011' + '0' * (totalDimensions - 4), _dimensionLength)
