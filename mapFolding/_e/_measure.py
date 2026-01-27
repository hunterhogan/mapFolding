"""You can use this module to measure bit-level coordinate features of `leaf` and `pile` integers.

(AI generated docstring)

This module treats an `int` value as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
For $2^n$-dimensional maps, the proxy uses base $2$ digits, where each digit position corresponds to a dimension index.
The conventions for dimension-index ideographs and related constants live in `mapFolding._e._semiotics` [2].

Contents
--------
dimensionFourthNearest首
	You can locate the fourth most-significant non-zero digit index.
dimensionNearestTail
	You can locate the least-significant non-zero digit index.
dimensionNearest首
	You can locate the most-significant non-zero digit index.
dimensionSecondNearest首
	You can locate the second most-significant non-zero digit index.
dimensionThirdNearest首
	You can locate the third most-significant non-zero digit index.
dimensionsConsecutiveAtTail
	You can count consecutive tail digits with value `1` in a masked width.
howManyDimensionsHaveOddParity
	You can count non-head digits with value `1`.
invertLeafIn2上nDimensions
	You can invert base-2 digits within a fixed dimension count.
leafInSubHyperplane
	You can project a non-origin `leaf` to a sub-hyperplane by dropping the head digit.
ptount
	You can compute a bit-count-derived measurement after subtracting `一+零`.

References
----------
[1] Positional notation - Wikipedia
	https://en.wikipedia.org/wiki/Positional_notation
[2] mapFolding._e._semiotics
	Internal package reference

"""

from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_scan1, f_mod_2exp
from hunterMakesPy import raiseIfNone
from hunterMakesPy.parseParameters import intInnit
from mapFolding._e import 一, 零
from mapFolding._e.dataBaskets import EliminationState
from operator import getitem

def dimensionsConsecutiveAtTail(state: EliminationState, integerNonnegative: int) -> int:
	"""You can count consecutive tail radix-2 digits with value `1` in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	This function uses the dimension-count `state.dimensionsTotal` to invert digits within a fixed width, using the conventions
	in `mapFolding._e._semiotics` [2].
	This function uses `gmpy2.bit_scan1` [3] on the inverted value.

	Parameters
	----------
	state : EliminationState
		State container that provides `state.dimensionsTotal`.
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	digitsTrailingOnes : int
		The count of consecutive least-significant base-2 digits equal to `1`, bounded by `state.dimensionsTotal`.

	Raises
	------
	ValueError
		Raised when `integerNonnegative` is less than `0`.

	Examples
	--------
	This function is used when selecting leaf exclusions.

	>>> if (is_even(leafAt二Ante首)
	... 	or (is_odd(leafAt二Ante首) and (dimensionIndex(dimension) < dimensionsConsecutiveAtTail(state, leafAt二Ante首)))):
	... 	listRemoveLeaves.extend([dimension])

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/
	"""
# NOTE HEY! `invertLeafIn2上nDimensions` is pulling double duty: it sanitizes `integerNonnegative` and inverts it. So if you figure out how to
# achieve this functionality without calling `invertLeafIn2上nDimensions`, you need to add defensive code here.
	anInteger: int = invertLeafIn2上nDimensions(state.dimensionsTotal, integerNonnegative)
	return bit_scan1(anInteger) or 0

@cache
def dimensionNearest首(integerNonnegative: int, /) -> int:
	"""You can locate the most-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	The most-significant non-zero digit index corresponds to the dimension index nearest the head ideograph `首`, following
	`mapFolding._e._semiotics` [2].
	This function validates `integerNonnegative` using `hunterMakesPy.parseParameters.intInnit` [3].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexNearest首 : int
		The 0-indexed position of the most-significant base-2 digit with value `1`.

	Raises
	------
	ValueError
		Raised when `integerNonnegative` is less than `0`.

	Examples
	--------
	This function is used to compute a head-dimension index.

	>>> dimensionHead: int = dimensionNearest首(leafAt二)

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return max(0, anInteger.bit_length() - 1)

@cache
def dimensionSecondNearest首(integerNonnegative: int, /) -> int | None:
	"""You can locate the second most-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	The digit order is interpreted relative to the head ideograph `首`, following `mapFolding._e._semiotics` [2].
	This function uses `gmpy2.bit_flip` [3] to clear the most-significant digit.
	This function validates `integerNonnegative` using `hunterMakesPy.parseParameters.intInnit` [4].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexSecondNearest首 : int | None
		The 0-indexed position of the second most-significant base-2 digit with value `1`.
		The return value is `None` when `integerNonnegative` has fewer than two non-zero digits.

	Raises
	------
	ValueError
		Raised when `integerNonnegative` is less than `0`.

	Examples
	--------
	This function is used as part of exclusion rules.

	>>> if (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1
	... 	and (dimensionNearest首(pileOfLeaf二一) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf二一)) < 2)
	... ):
	... 	addend: int = productsOfDimensions[dimensionsTotal-2] + 4

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/
	[4] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	dimensionSecondNearest: int | None = None
	anotherInteger = int(bit_flip(anInteger, dimensionNearest首(anInteger)))
	if anotherInteger == 0:
		dimensionSecondNearest = None
	else:
		dimensionSecondNearest = dimensionNearest首(anotherInteger)
	return dimensionSecondNearest

@cache
def dimensionThirdNearest首(integerNonnegative: int, /) -> int | None:
	"""You can locate the third most-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	The digit order is interpreted relative to the head ideograph `首`, following `mapFolding._e._semiotics` [2].
	This function uses `gmpy2.bit_flip` [3] to clear digits.
	This function validates `integerNonnegative` using `hunterMakesPy.parseParameters.intInnit` [4].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexThirdNearest首 : int | None
		The 0-indexed position of the third most-significant base-2 digit with value `1`.
		The return value is `None` when `integerNonnegative` has fewer than three non-zero digits.

	Raises
	------
	ValueError
		Raised when `integerNonnegative` is less than `0`.

	Examples
	--------
	This function is used in domain index elimination.

	>>> if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
	... 	indexDomain0: int = (pilesTotal // 2) + 1
	... 	listIndicesPilesExcluded.extend([indexDomain0])

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/
	[4] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)

	dimensionNearest: int = dimensionNearest首(anInteger)
	dimensionSecondNearest: int | None = dimensionSecondNearest首(anInteger)
	dimensionThirdNearest: int | None = None

	if dimensionSecondNearest in [0, None]:
		dimensionThirdNearest = None
	else:
		anotherInteger = int(bit_flip(anInteger, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)))
		if anotherInteger == 0:
			dimensionThirdNearest = None
		else:
			dimensionThirdNearest = dimensionNearest首(anotherInteger)
	return dimensionThirdNearest

@cache
def dimensionFourthNearest首(integerNonnegative: int, /) -> int | None:
	"""You can locate the fourth most-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	The digit order is interpreted relative to the head ideograph `首`, following `mapFolding._e._semiotics` [2].
	This function uses `gmpy2.bit_flip` [3] to clear digits.
	This function validates `integerNonnegative` using `hunterMakesPy.parseParameters.intInnit` [4].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexFourthNearest首 : int | None
		The 0-indexed position of the fourth most-significant base-2 digit with value `1`.
		The return value is `None` when `integerNonnegative` has fewer than four non-zero digits.

	Raises
	------
	ValueError
		Raised when `integerNonnegative` is less than `0`.

	Examples
	--------
	This function is used as a refinement when a third-nearest digit matches a specific pattern.

	>>> if dimensionThirdNearest首(pileOfLeaf零) == 一+零:
	... 	indexDomain0 = pilesTotal // 4
	... 	if dimensionFourthNearest首(pileOfLeaf零) == 一:
	... 		indicesDomain0ToExclude.extend([indexDomain0])

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/
	[4] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)

	dimensionNearest: int = dimensionNearest首(anInteger)
	dimensionSecondNearest: int | None = dimensionSecondNearest首(anInteger)
	dimensionThirdNearest: int | None = dimensionThirdNearest首(anInteger)
	dimensionFourthNearest: int | None = None

	if dimensionThirdNearest in [0, None]:
		dimensionFourthNearest = None
	else:
		anotherInteger = int(bit_flip(anInteger, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)).bit_flip(raiseIfNone(dimensionThirdNearest)))
		if anotherInteger == 0:
			dimensionFourthNearest = None
		else:
			dimensionFourthNearest = dimensionNearest首(anotherInteger)
	return dimensionFourthNearest

@cache
def leafInSubHyperplane(notLeafOrigin: int, /) -> int:
	"""You can project `notLeafOrigin` to a sub-hyperplane by dropping the head radix-2 digit.

	(AI generated docstring, which may or may not have been accurate; edited by me, Hunter Hogan, which may or may not have improved it.)

	This function treats `notLeafOrigin` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	For $2^n$-dimensional maps, the base $2$ most-significant digit marks the dimension index nearest the head ideograph `首`, following
	`mapFolding._e._semiotics` [2].

	This function returns the value formed by removing the most-significant non-zero digit and preserving the remaining digits.
	This function is implemented as a modulus operation on a power of two using `gmpy2.f_mod_2exp` [3].
	This function validates `notLeafOrigin` using `hunterMakesPy.parseParameters.intInnit` [4].

	Parameters
	----------
	notLeafOrigin : int
		A `leaf` value that is not equal to `leafOrigin`.

	Returns
	-------
	leafSubHyperplane : int
		The `leaf` value in the sub-hyperplane implied by `notLeafOrigin`.

	Raises
	------
	ValueError
		Raised when `notLeafOrigin` is less than `1`.

	Examples
	--------
	This function is used when a `leaf` has a specific parity pattern.

	>>> if howManyDimensionsHaveOddParity(leafAt二Ante首) == 1:
	... 	listRemoveLeaves.extend([leafInSubHyperplane(leafAt二Ante首)])

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/
	[4] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([notLeafOrigin], 'notLeafOrigin', int), 0)
	if anInteger < 1:
		message: str = f"I received `{notLeafOrigin = }`, but I need a value greater than 0."
		raise ValueError(message)
	return int(f_mod_2exp(anInteger, dimensionNearest首(anInteger)))

@cache
def dimensionNearestTail(integerNonnegative: int, /) -> int:
	"""You can locate the least-significant non-zero radix-2 digit index in `integerNonnegative`.

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	The least-significant non-zero digit index is the count of trailing zeros in the base $2$ representation.
	This function follows the dimension-index conventions in `mapFolding._e._semiotics` [2].
	This function uses `gmpy2.bit_scan1` [3].
	This function validates `integerNonnegative` using `hunterMakesPy.parseParameters.intInnit` [4].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	dimensionIndexNearestTail : int
		The 0-indexed position of the least-significant base-2 digit with value `1`.

	Raises
	------
	ValueError
		Raised when `integerNonnegative` is less than `0`.

	Examples
	--------
	This function is used when building domain tuples.

	>>> dimensionTail: int = dimensionNearestTail(pileOfLeaf二一)

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/
	[4] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return bit_scan1(anInteger) or 0

@cache
def howManyDimensionsHaveOddParity(integerNonnegative: int, /) -> int:
	"""You can count non-head radix-2 digits with value `1` in `integerNonnegative`.

	(AI generated docstring)

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	For a non-zero value, the most-significant digit is treated as the head digit nearest the ideograph `首`, following
	`mapFolding._e._semiotics` [2].
	This function validates `integerNonnegative` using `hunterMakesPy.parseParameters.intInnit` [3].

	Parameters
	----------
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	countOnesExcludingHead : int
		The count of digits equal to `1`, excluding the most-significant digit.
		The return value is `0` when `integerNonnegative` is `0`.

	Raises
	------
	ValueError
		Raised when `integerNonnegative` is less than `0`.

	Examples
	--------
	This function is used to select the crease slicing index.

	>>> slicingIndices: int = is_odd(howManyDimensionsHaveOddParity(leaf))

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return max(0, anInteger.bit_count() - 1)

@cache
def invertLeafIn2上nDimensions(dimensionsTotal: int, integerNonnegative: int) -> int:
	"""You can invert base-2 digits in `integerNonnegative` within a dimension count `dimensionsTotal`.

	(AI generated docstring)

	This function treats `integerNonnegative` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	The fixed-width inversion uses `dimensionsTotal` as the digit width, which aligns with the dimension indexing conventions in
	`mapFolding._e._semiotics` [2].
	This function uses `gmpy2.bit_mask` [3] to build an inversion mask.
	This function validates `integerNonnegative` using `hunterMakesPy.parseParameters.intInnit` [4].

	Parameters
	----------
	dimensionsTotal : int
		The number of base-2 digit positions that define the inversion mask.
	integerNonnegative : int
		Input value interpreted as a base-2 positional coordinate encoding.

	Returns
	-------
	integerInverted : int
		The value produced by XOR with `bit_mask(dimensionsTotal)`.

	Raises
	------
	ValueError
		Raised when `integerNonnegative` is less than `0`.

	Examples
	--------
	This function is used to compute a tail measurement by inverting within `state.dimensionsTotal`.

	>>> anInteger: int = invertLeafIn2上nDimensions(state.dimensionsTotal, integerNonnegative)
	>>> return bit_scan1(anInteger) or 0

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/
	[4] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return int(anInteger ^ bit_mask(dimensionsTotal))

@cache
def ptount(integerAbove3: int, /) -> int:
	"""You can compute a bit-count-derived measurement after subtracting `一+零` from `integerAbove3`.

	This function treats `integerAbove3` as a single-base positional-numeral system as a proxy for Cartesian coordinates [1].
	This function follows the dimension-index conventions in `mapFolding._e._semiotics` [2].

	This function delegates to `leafInSubHyperplane` [3].
	This function validates `integerAbove3` using `hunterMakesPy.parseParameters.intInnit` [4].
	This function computes `leafInSubHyperplane(integerAbove3 - (一+零)).bit_count()`.

	Parameters
	----------
	integerAbove3 : int
		Input value interpreted as a base-2 positional coordinate encoding.
		The value must be at least `3`.

	Returns
	-------
	measurement : int
		The `int.bit_count` value of the projected sub-hyperplane value.

	Raises
	------
	ValueError
		Raised when `integerAbove3` is less than `3`.

	Examples
	--------
	This function is used to choose crease indices.

	>>> if isOdd吗(leafAt一零):
	... 	listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一零), 5), ptount(leafAt一零)])

	References
	----------
	[1] Positional notation - Wikipedia
		https://en.wikipedia.org/wiki/Positional_notation
	[2] mapFolding._e._semiotics
		Internal package reference
	[3] mapFolding._e._measure.leafInSubHyperplane
		Internal package reference
	[4] hunterMakesPy
		https://context7.com/hunterhogan/huntermakespy
	"""
	anInteger: int = getitem(intInnit([integerAbove3], 'integerAbove3', int), 0)
	if anInteger < 3:
		message: str = f"I received `{integerAbove3 = }`, but I need a value greater than 3."
		raise ValueError(message)

	return leafInSubHyperplane(anInteger - (一+零)).bit_count()

