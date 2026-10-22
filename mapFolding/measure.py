"""Measure positional-numeral features of leaf and pile integers.

(AI generated docstring)

You can locate digits with value `1`, count digits with value `1`, and project or invert leaf
values. The positional-numeral system lives in
`mapFolding._semiotics` [1].

Contents
--------
Functions
	invertLeafIn2上nDimensions
		Invert the lowest `totalDimensions` bits of `integerNonnegative`.
	leafInSubHyperplane
		Remove the most significant set bit from the leaf value `notLeafOrigin`.
	ptount
		Count lower set bits after subtracting three from `integerAbove3`.
	工dimensionTail
		Locate the least significant set bit in `integerNonnegative`.
	工dimension首一
		Locate the second most significant set bit in `integerNonnegative`.
	工dimension首三
		Locate the fourth most significant set bit in `integerNonnegative`.
	工dimension首二
		Locate the third most significant set bit in `integerNonnegative`.
	工dimension首零
		Locate the most significant set bit in `integerNonnegative`.
	工totalDimensionsOdd
		Count set bits below the head bit of `integerNonnegative`.
	工totalDimensionsTail
		Measure trailing set bits in `integerNonnegative` within the map width.

References
----------
[1] `mapFolding._semiotics`
"""
from __future__ import annotations

from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_scan1, f_mod_2exp
from hunterMakesPy import raiseIfNone
from mapFolding import 一, 零
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateElimination

def 工totalDimensionsTail(state: StateElimination, integerNonnegative: int) -> int:
	"""Measure trailing set bits in `integerNonnegative` within the map width.

	(AI generated docstring)

	You can inspect consecutive set bits starting at the least significant end of
	`integerNonnegative`. The map width comes from `state.totalDimensions`. The result is
	`0` when the inverted value has no set bit.

	Parameters
	----------
	state : StateElimination
		Elimination state that provides the map shape and dimension count.
	integerNonnegative : int
		Nonnegative integer whose binary digits encode dimension coordinates.

	Returns
	-------
	digitsTrailingOnes : int
		Number of trailing set bits for an input within the map width, except that an input
		with every bit set returns `0`. An even input also returns `0`.

	Bit Scan
	--------
	You invert the map-width bits with `invertLeafIn2上nDimensions` [1], scan the result with
	`gmpy2.bit_scan1` [2], and replace `None` with `0`. You preserve bits above the map width during
	inversion.

	References
	----------
	[1] `invertLeafIn2上nDimensions`

	[2] gmpy2 integer operations.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return bit_scan1(invertLeafIn2上nDimensions(state.totalDimensions, integerNonnegative)) or 0

@cache
def 工dimension首零(integerNonnegative: int, /) -> int:
	"""Locate the most significant set bit in `integerNonnegative`.

	(AI generated docstring)

	You can inspect the dimension index of the most significant set bit in
	`integerNonnegative`. Dimension indices count from zero at the least significant end.
	The head convention is defined in the positional-numeral system [1].

	Parameters
	----------
	integerNonnegative : int
		Nonnegative integer whose binary digits encode dimension coordinates.

	Returns
	-------
	dimensionIndexNearest首 : int
		Zero-based position of the most significant set bit. The input `0` returns `0`.

	References
	----------
	[1] `mapFolding._semiotics`
	"""
	return max(0, integerNonnegative.bit_length() - 1)

@cache
def 工dimension首一(integerNonnegative: int, /) -> int | None:
	"""Locate the second most significant set bit in `integerNonnegative`.

	(AI generated docstring)

	You can inspect the dimension index of the second most significant set bit in
	`integerNonnegative`. Dimension indices count from zero at the least significant end.
	The head convention is defined in the positional-numeral system [1].

	Parameters
	----------
	integerNonnegative : int
		Nonnegative integer whose binary digits encode dimension coordinates.

	Returns
	-------
	dimensionIndexSecondNearest首 : int | None
		Zero-based position of the second most significant set bit. A positive input
		with fewer than two set bits returns `None`. The input `0` returns `0`.

	References
	----------
	[1] `mapFolding._semiotics`
	"""
	anotherInteger: int = int(bit_flip(integerNonnegative, 工dimension首零(integerNonnegative)))
	if anotherInteger == 0:
		dimensionSecondNearest: int | None = None
	else:
		dimensionSecondNearest = 工dimension首零(anotherInteger)
	return dimensionSecondNearest

@cache
def 工dimension首二(integerNonnegative: int, /) -> int | None:
	"""Locate the third most significant set bit in `integerNonnegative`.

	(AI generated docstring)

	You can inspect the dimension index of the third most significant set bit in
	`integerNonnegative`. Dimension indices count from zero at the least significant end.
	The head convention is defined in the positional-numeral system [1].

	Parameters
	----------
	integerNonnegative : int
		Nonnegative integer whose binary digits encode dimension coordinates.

	Returns
	-------
	dimensionIndexThirdNearest首 : int | None
		Zero-based position of the third most significant set bit. An input with
		fewer than three set bits returns `None`.

	References
	----------
	[1] `mapFolding._semiotics`
	"""
	dimensionNearest: int = 工dimension首零(integerNonnegative)
	dimensionSecondNearest: int | None = 工dimension首一(integerNonnegative)

	if dimensionSecondNearest in {0, None}:
		dimensionThirdNearest: int | None = None
	else:
		anotherInteger: int = int(bit_flip(integerNonnegative, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)))
		if anotherInteger == 0:
			dimensionThirdNearest = None
		else:
			dimensionThirdNearest = 工dimension首零(anotherInteger)
	return dimensionThirdNearest

@cache
def 工dimension首三(integerNonnegative: int, /) -> int | None:
	"""Locate the fourth most significant set bit in `integerNonnegative`.

	(AI generated docstring)

	You can inspect the dimension index of the fourth most significant set bit in
	`integerNonnegative`. Dimension indices count from zero at the least significant end.
	The head convention is defined in the positional-numeral system [1].

	Parameters
	----------
	integerNonnegative : int
		Nonnegative integer whose binary digits encode dimension coordinates.

	Returns
	-------
	dimensionIndexFourthNearest首 : int | None
		Zero-based position of the fourth most significant set bit. An input with
		fewer than four set bits returns `None`.

	References
	----------
	[1] `mapFolding._semiotics`
	"""
	dimensionNearest: int = 工dimension首零(integerNonnegative)
	dimensionSecondNearest: int | None = 工dimension首一(integerNonnegative)
	dimensionThirdNearest: int | None = 工dimension首二(integerNonnegative)

	if dimensionThirdNearest in {0, None}:
		dimensionFourthNearest: int | None = None
	else:
		anotherInteger: int = int(bit_flip(integerNonnegative, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)).bit_flip(raiseIfNone(dimensionThirdNearest)))
		if anotherInteger == 0:
			dimensionFourthNearest = None
		else:
			dimensionFourthNearest = 工dimension首零(anotherInteger)
	return dimensionFourthNearest

@cache
def leafInSubHyperplane(notLeafOrigin: int, /) -> int:
	"""Remove the most significant set bit from the leaf value `notLeafOrigin`.

	(AI generated docstring, which may or may not have been accurate; edited by me, Hunter Hogan,
	which may or may not have improved it.)

	You can project the leaf value `notLeafOrigin` into the lower-dimensional
	part of the map by clearing the most significant set bit. Every lower bit keeps the same
	position and value.

	Parameters
	----------
	notLeafOrigin : int
		Positive leaf value. Supply a valid input without relying on validation.

	Returns
	-------
	leafSubHyperplane : int
		Leaf value with the most significant set bit removed. A power of two returns
		`0`; the input `0` also returns `0`.

	Remainder Calculation
	---------------------
	You use the index returned by `工dimension首零` [1] as the exponent in `gmpy2.f_mod_2exp` [2].

	Examples
	--------
	From `pinByDomain.py`, with `leafAt二Ante首` read from the workbench state.

		```python
		boxOfRemoveLeaves.extend([leafInSubHyperplane(leafAt二Ante首)])
		```

	References
	----------
	[1] `工dimension首零`

	[2] gmpy2 integer operations.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return int(f_mod_2exp(notLeafOrigin, 工dimension首零(notLeafOrigin)))

@cache
def 工dimensionTail(integerNonnegative: int, /) -> int:
	"""Locate the least significant set bit in `integerNonnegative`.

	(AI generated docstring)

	You can inspect the dimension index of the first set bit when reading `integerNonnegative` from the
	least significant end. You receive index `0` for the input `0`, which has no set bit.

	Parameters
	----------
	integerNonnegative : int
		Nonnegative integer whose binary digits encode dimension coordinates.

	Returns
	-------
	dimensionIndexNearestTail : int
		Zero-based position of the least significant set bit, or `0` for the input `0`.

	Bit Scan
	--------
	You scan the input with `gmpy2.bit_scan1` [1] and receive `0` when the scan returns `None`.

	References
	----------
	[1] gmpy2 integer operations.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return bit_scan1(integerNonnegative) or 0

@cache
def 工totalDimensionsOdd(integerNonnegative: int, /) -> int:
	"""Count set bits below the head bit of `integerNonnegative`.

	(AI generated docstring)

	You can count the active coordinate positions of `integerNonnegative` after
	excluding the most significant set bit. The input `0` returns `0`.

	Parameters
	----------
	integerNonnegative : int
		Nonnegative integer whose binary digits encode dimension coordinates.

	Returns
	-------
	countOnesExcludingHead : int
		Number of set bits minus one, with a minimum of `0`.
	"""
	return max(0, integerNonnegative.bit_count() - 1)

@cache
def invertLeafIn2上nDimensions(totalDimensions: int, integerNonnegative: int) -> int:
	"""Invert the lowest `totalDimensions` bits of `integerNonnegative`.

	(AI generated docstring)

	You can exchange set and unset coordinate bits of `integerNonnegative` within
	the map width `totalDimensions`. Bits above the map width keep their original values.

	Parameters
	----------
	totalDimensions : int
		Nonnegative number of bit positions to invert.
	integerNonnegative : int
		Nonnegative integer to invert.

	Returns
	-------
	integerInverted : int
		Input value with every bit in the specified width inverted.

	Inversion Mask
	--------------
	You apply XOR with the all-set mask returned by `gmpy2.bit_mask` [1], preserving input bits above the
	mask width.

	References
	----------
	[1] gmpy2 integer operations.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return int(integerNonnegative ^ bit_mask(totalDimensions))

@cache
def ptount(integerAbove3: int, /) -> int:
	"""Count lower set bits after subtracting three from `integerAbove3`.

	(AI generated docstring)

	You can obtain a coordinate measurement used to exclude crease neighbors by subtracting `3` from
	`integerAbove3`, removing the most significant set bit, and counting the remaining set bits.

	Parameters
	----------
	integerAbove3 : int
		Integer greater than `3`, as required by the crease analysis. Supply a valid input without relying
		on validation.

	Returns
	-------
	measurement : int
		Number of set bits after subtraction and head-bit removal.

	Coordinate Projection
	---------------------
	The projection uses `leafInSubHyperplane` [1]
	on the value after subtraction.

	Examples
	--------
	From `pinByCrease.py`, with `leafAt一零` read from the workbench state.

		```python
		boxOfCreaseIndicesExcluded.extend([*range(工dimension首零(leafAt一零), 5), ptount(leafAt一零)])
		```

	References
	----------
	[1] `leafInSubHyperplane`
	"""
	return leafInSubHyperplane(integerAbove3 - (一 + 零)).bit_count()
