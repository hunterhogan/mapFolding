"""Measure bit-level coordinate features of `Leaf` and `Pile` integers.

This module treats an `int` value as a single-base positional-numeral system as a proxy for Cartesian
coordinates [1]. For 2ⁿ-dimensional maps, the proxy uses base-2 digits, where each digit position
corresponds to a dimension index. The conventions for dimension-index ideographs and related constants
live in `mapFolding._e._2上nDimensionalSemiotics` [2].

Contents
--------
工dimension首三
	Locate the fourth most-significant non-zero digit index.
工dimensionTail
	Locate the least-significant non-zero digit index.
工dimension首零
	Locate the most-significant non-zero digit index.
工dimension首一
	Locate the second most-significant non-zero digit index.
工dimension首二
	Locate the third most-significant non-zero digit index.
工totalDimensionsTail
	Count consecutive tail digits with value `1` in a masked width.
工totalDimensionsOdd
	Count non-head digits with value `1`.
invertLeafIn2上nDimensions
	Invert base-2 digits within a fixed dimension count.
leafInSubHyperplane
	Project a non-origin `leaf` to a sub-hyperplane by dropping the head digit.
ptount
	Compute a bit-count-derived measurement after subtracting `一+零`.

References
----------
[1] Positional notation - Wikipedia
	https://en.wikipedia.org/wiki/Positional_notation
[2] mapFolding._e._2上nDimensionalSemiotics
"""
from __future__ import annotations

from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_scan1, f_mod_2exp
from hunterMakesPy import raiseIfNone
from mapFolding._e._2上nDimensional import 一, 零
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import EliminationState

def 工totalDimensionsTail(state: EliminationState, integerNonnegative: int) -> int:
	"""Count consecutive tail radix-2 digits with value `1` in `integerNonnegative`.

	Parameters
	----------
	state : EliminationState
		State container that provides `state.totalDimensions`.
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	digitsTrailingOnes : int
		The count of consecutive least-significant base-2 digits equal to `1`, bounded by
		`state.totalDimensions`.

	Examples
	--------
	```python
		if ((isEven吗(leafAt二Ante首) or (isOdd吗(leafAt二Ante首) and (dimensionIndex(dimension) <
		工totalDimensionsTail(state, leafAt二Ante首))))):
			boxOfRemoveLeaves.extend([dimension])
	```
	"""
	return bit_scan1(invertLeafIn2上nDimensions(state.totalDimensions, integerNonnegative)) or 0

@cache
def 工dimension首零(integerNonnegative: int, /) -> int:
	"""Locate the most-significant non-zero radix-2 digit index in `integerNonnegative`.

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexNearest首 : int
		The 0-indexed position of the most-significant base-2 digit with value `1`.

	Examples
	--------
	```python
		dimensionHead: int = 工dimension首零(leafAt二)
	```
	"""
	return max(0, integerNonnegative.bit_length() - 1)

@cache
def 工dimension首一(integerNonnegative: int, /) -> int | None:
	"""Locate the second most-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy
	for Cartesian coordinates [1]. The digit order is interpreted relative to the head ideograph `首`,
	following `mapFolding._e._2上nDimensionalSemiotics` [2].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexSecondNearest首 : int | None
		The 0-indexed position of the second most-significant base-2 digit with value `1`. The return
		value is `None` when `integerNonnegative` has fewer than two non-zero digits.

	Examples
	--------
	```python
		if (工totalDimensionsOdd(pileOfLeaf二一) == 1
			and (工dimension首零(pileOfLeaf二一)
				- raiseIfNone(工dimension首一(pileOfLeaf 二一)) < 2)
		):
			addend: int = mapShapeProducts[totalDimensions-2] + 4
	```

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._2上nDimensionalSemiotics
	"""
	anotherInteger: int = int(bit_flip(integerNonnegative, 工dimension首零(integerNonnegative)))
	if anotherInteger == 0:
		dimensionSecondNearest: int | None = None
	else:
		dimensionSecondNearest = 工dimension首零(anotherInteger)
	return dimensionSecondNearest

@cache
def 工dimension首二(integerNonnegative: int, /) -> int | None:
	"""Locate the third most-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy
	for Cartesian coordinates [1]. The digit order is interpreted relative to the head ideograph `首`,
	following `mapFolding._e._2上nDimensionalSemiotics` [2].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexThirdNearest首 : int | None
		The 0-indexed position of the third most-significant base-2 digit with value `1`. The return
		value is `None` when `integerNonnegative` has fewer than three non-zero digits.

	Examples
	--------
	```python
		if (工dimension首二(pileOfLeaf零) == 一)
		and (二+零 <= 工dimension首零(pileOfLeaf 零)):
			次Domain0: int = (pilesTotal // 2) + 1
			boxOfIndicesPilesExcluded.extend([次Domain0])
	```

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._2上nDimensionalSemiotics
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
	"""Locate the fourth most-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy
	for Cartesian coordinates [1]. The digit order is interpreted relative to the head ideograph `首`,
	following `mapFolding._e._2上nDimensionalSemiotics` [2].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexFourthNearest首 : int | None
		The 0-indexed position of the fourth most-significant base-2 digit with value `1`. The return
		value is `None` when `integerNonnegative` has fewer than four non-zero digits.

	Examples
	--------
	```python
		if 工dimension首二(pileOfLeaf零) == 一+零:
			次Domain0 = pilesTotal // 4 if 工dimension首三(pileOfLeaf零) == 一:
				indicesDomain0ToExclude.extend([次Domain0])
	```

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._2上nDimensionalSemiotics
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
	"""Project `notLeafOrigin` to a sub-hyperplane by dropping the head radix-2 digit.

	(AI generated docstring, which may or may not have been accurate; edited by me, Hunter Hogan,
	which may or may not have improved it.)

	This function treats `notLeafOrigin` as a single-base positional-numeral system as a proxy for
	Cartesian coordinates [1]. For 2ⁿ-dimensional maps, the base-2 most-significant digit marks the
	dimension index nearest the head ideograph `首`, following `mapFolding._e._2上
	nDimensionalSemiotics` [2].

	This function returns the value formed by removing the most-significant non-zero digit and
	preserving the remaining digits. This function is implemented as a modulus operation on a power of
	two using `gmpy2.f_mod_2exp` [3]. This function validates `notLeafOrigin` using
	`hunterMakesPy.parseParameters.intInnit` [4].

	Parameters
	----------
	notLeafOrigin : int
		A `leaf` value that is not equal to `leafOrigin`.

	Returns
	-------
	leafSubHyperplane : int
		The `leaf` value in the sub-hyperplane implied by `notLeafOrigin`.

	Examples
	--------
	```python
		if 工totalDimensionsOdd(leafAt二Ante首) == 1:
			boxOfRemoveLeaves.extend([leafInSubHyperplane(leafAt二Ante首)])
	```

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._2上nDimensionalSemiotics
	"""
	return int(f_mod_2exp(notLeafOrigin, 工dimension首零(notLeafOrigin)))

@cache
def 工dimensionTail(integerNonnegative: int, /) -> int:
	"""Locate the least-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy
	for Cartesian coordinates [1]. The least-significant non-zero digit index is the count of trailing
	zeros in the base-2 representation. This function follows the dimension-index conventions in
	`mapFolding._e._2上nDimensionalSemiotics` [2].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexNearestTail : int
		The 0-indexed position of the least-significant base-2 digit with value `1`.

	Examples
	--------
	```python
		dimensionTail: int = 工dimensionTail(pileOfLeaf二一)
	```

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._2上nDimensionalSemiotics
	"""
	return bit_scan1(integerNonnegative) or 0

@cache
def 工totalDimensionsOdd(integerNonnegative: int, /) -> int:
	"""Count non-head radix-2 digits with value `1` in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy
	for Cartesian coordinates [1]. For a non-zero value, the most-significant digit is treated as the
	head digit nearest the ideograph `首`, following `mapFolding._e._2上nDimensionalSemiotics` [2].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	countOnesExcludingHead : int
		The count of digits equal to `1`, excluding the most-significant digit. The return value is
		`0` when `integerNonnegative` is `0`.

	Examples
	--------
	```python
		slicingIndices: int = isOdd吗(工totalDimensionsOdd(leaf))
	```

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._2上nDimensionalSemiotics
	"""
	return max(0, integerNonnegative.bit_count() - 1)

@cache
def invertLeafIn2上nDimensions(totalDimensions: int, integerNonnegative: int) -> int:
	"""Invert base-2 digits in `integerNonnegative` within a dimension count `totalDimensions`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy
	for Cartesian coordinates [1]. The fixed-width inversion uses `totalDimensions` as the digit
	width, which aligns with the dimension indexing conventions in `mapFolding._e._2上
	nDimensionalSemiotics` [2].

	Parameters
	----------
	totalDimensions : int
		The number of base-2 digit positions that define the inversion mask.
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	integerInverted : int
		The value produced by XOR with `bit_mask(totalDimensions)`.

	Examples
	--------
	```python
		anInteger: int = invertLeafIn2上nDimensions(state.totalDimensions, integerNonnegative) return
		bit_scan1(anInteger) or 0
	```

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._2上nDimensionalSemiotics
	"""
	return int(integerNonnegative ^ bit_mask(totalDimensions))

@cache
def ptount(integerAbove3: int, /) -> int:
	"""Compute a bit-count-derived measurement after subtracting `一+零` from `integerAbove3`.

	This function treats `integerAbove3` as a single-base positional-numeral system as a proxy for
	Cartesian coordinates [1]. This function follows the dimension-index conventions in
	`mapFolding._e._2上nDimensionalSemiotics` [2].

	Parameters
	----------
	integerAbove3 : int
		Input value interpreted as a base-2 positional coordinate encoding. The value must be at least
		`3`.

	Returns
	-------
	measurement : int
		The `int.bit_count` value of the projected sub-hyperplane value.

	Examples
	--------
	```python
		if isOdd吗(leafAt一零):
			boxOfCreaseIndicesExcluded.extend([*range(工dimension首零(leafAt一零), 5), ptount(leafAt 一零)])
	```

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._2上nDimensionalSemiotics
	"""
	return leafInSubHyperplane(integerAbove3 - (一 + 零)).bit_count()
