from functools import cache
from hunterMakesPy import intInnit, raiseIfNone
import gmpy2

@cache
def dimensionNearest首(integerNonnegative: int, /) -> int:
	"""Find the 0-indexed position of the most significant non-zero radix-2 digit in `integerNonnegative`."""
	anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return max(0, anInteger.bit_length() - 1)

@cache
def dimensionSecondNearest首(integerNonnegative: int, /) -> int | None:
	"""Find the 0-indexed position of the second most significant non-zero radix-2 digit, if any, in `integerNonnegative`."""
	anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	dimensionSecondNearest: int | None = None
	anotherInteger = int(gmpy2.bit_flip(anInteger, dimensionNearest首(anInteger)))
	if anotherInteger == 0:
		dimensionSecondNearest = None
	else:
		dimensionSecondNearest = dimensionNearest首(anotherInteger)
	return dimensionSecondNearest

@cache
def dimensionThirdNearest首(integerNonnegative: int, /) -> int | None:
	"""Find the 0-indexed position of the third most significant non-zero radix-2 digit, if any, in `integerNonnegative`."""
	anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)

	dimensionNearest: int = dimensionNearest首(anInteger)
	dimensionSecondNearest: int | None = dimensionSecondNearest首(anInteger)
	dimensionThirdNearest: int | None = None

	if dimensionSecondNearest in [0, None]:
		dimensionThirdNearest = None
	else:
		anotherInteger = int(gmpy2.bit_flip(anInteger, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)))
		if anotherInteger == 0:
			dimensionThirdNearest = None
		else:
			dimensionThirdNearest = dimensionNearest首(anotherInteger)
	return dimensionThirdNearest

@cache
def dimensionFourthNearest首(integerNonnegative: int, /) -> int | None:
	"""Find the 0-indexed position of the fourth most significant non-zero radix-2 digit, if any, in `integerNonnegative`."""
	anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
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
		anotherInteger = int(gmpy2.bit_flip(anInteger, dimensionNearest).bit_flip(raiseIfNone(dimensionSecondNearest)).bit_flip(raiseIfNone(dimensionThirdNearest)))
		if anotherInteger == 0:
			dimensionFourthNearest = None
		else:
			dimensionFourthNearest = dimensionNearest首(anotherInteger)
	return dimensionFourthNearest

@cache
def leafInSubHyperplane(leafAbove1: int, /) -> int:
	"""Compute the projection of a hyperplane leaf onto its lower-dimensional sub-hyperplane.

	(AI generated docstring.)

	For 2^d hyperplane maps, each leaf's `leaf` encodes its d-dimensional coordinates
	in binary (base-2 positional notation). The most significant bit (MSB) indicates the
	highest dimension where the coordinate equals 1.

	This function extracts the lower (MSB - 1) bits, which represent the leaf's position
	when projected onto the sub-hyperplane formed by all dimensions below the MSB dimension.
	That is equivalent to the `leaf` of the leaf within the lower-dimensional sub-hyperplane.

	Technical implementation: `leafAbove1 mod 2^(bit_length - 1)`

	Parameters
	----------
	leafAbove1 : int
		The `leaf` > 1 representing a leaf in a 2^d hyperplane.

	Returns
	-------
	leafSubHyperplane : int
		The position of the leaf within the sub-hyperplane defined by dimensions [0, ..., MSB-1].
		This is the value formed by all bits except the MSB.
	"""
	anInteger: int = intInnit([leafAbove1], 'leafAbove1', type[int])[0]
	if anInteger <= 1:
		message: str = f"I received `{leafAbove1 = }`, but I need a value greater than 1."
		raise ValueError(message)
	return int(gmpy2.f_mod_2exp(anInteger, anInteger.bit_length() - 1))

@cache
def howMany0coordinatesAtTail(integerNonnegative: int, /) -> int:
	"""Compute the number of times `integerNonnegative` is divisible by 2; aka 'CTZ', Count Trailing Zeros in the binary form."""
	anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return gmpy2.bit_scan1(anInteger) or 0

@cache
def howManyDimensionsHaveOddParity(integerNonnegative: int, /) -> int:
	anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return max(0, anInteger.bit_count() - 1)

@cache
def ptount(integerAbove3: int, /) -> int:
	"""After subtracting 0b000011 from `integerAbove3`, measure the distance from a ***p***ower of ***t***wo's bit c***ount***.

	Notes
	-----
	- Pronounced "tount" because the "p" is silent.
	- Just like the "p", the reason why this is useful is silent.
	- I suspect there is a more direct route to measure this but I am unaware of it.
	"""
	anInteger: int = intInnit([integerAbove3], 'integerAbove3', type[int])[0]
	if anInteger <= 3:
		message: str = f"I received `{integerAbove3 = }`, but I need a value greater than 3."
		raise ValueError(message)

	return leafInSubHyperplane(anInteger - 0b000011).bit_count()

