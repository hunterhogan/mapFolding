from functools import cache
from gmpy2 import bit_flip, bit_scan1, f_mod_2exp
from hunterMakesPy import raiseIfNone
from hunterMakesPy.parseParameters import intInnit
from mapFolding._e import Z0Z_invert, 一, 零
from mapFolding._e.dataBaskets import EliminationState
from operator import getitem

@cache
def dimensionNearest首(integerNonnegative: int, /) -> int:
	"""Find the 0-indexed position of the most significant non-zero radix-2 digit in `integerNonnegative`."""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return max(0, anInteger.bit_length() - 1)

@cache
def dimensionSecondNearest首(integerNonnegative: int, /) -> int | None:
	"""Find the 0-indexed position of the second most significant non-zero radix-2 digit, if any, in `integerNonnegative`."""
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
	"""Find the 0-indexed position of the third most significant non-zero radix-2 digit, if any, in `integerNonnegative`."""
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
	"""Find the 0-indexed position of the fourth most significant non-zero radix-2 digit, if any, in `integerNonnegative`."""
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
	"""For `notLeafOrigin` in a map with d-many dimensions, compute the projection of `notLeafOrigin` onto the sub-hyperplane that has one fewer dimension.

	(AI generated docstring, which may or may not have been accurate; edited by me, Hunter Hogan, which may or may not have improved it.)

	For 2^n-dimensional hyperplane maps, each leaf's `leaf` encodes its d-dimensional coordinates in binary (base-2 positional notation). The
	most significant digit (MSD) indicates the dimension nearest the "head", 首, where the coordinate equals 1: it follows that
	other dimensions closer to the "head" have coordinates equal to 0.

	The other coordinates might not be 0, and they represent the leaf's position when projected onto the sub-hyperplane: the
	"equivalent" `leaf` of `notLeafOrigin` in space with one fewer dimension.

	Parameters
	----------
	notLeafOrigin : int
		A `leaf` in a 2^n-dimensional map.

	Returns
	-------
	leafSubHyperplane : int
		The position of the leaf within the sub-hyperplane defined by dimensions [0, ..., MSD-1]. This is the value formed by all
		digits except the MSD.
	"""
	anInteger: int = getitem(intInnit([notLeafOrigin], 'notLeafOrigin', int), 0)
	if anInteger < 1:
		message: str = f"I received `{notLeafOrigin = }`, but I need a value greater than 0."
		raise ValueError(message)
	return int(f_mod_2exp(anInteger, dimensionNearest首(anInteger)))

@cache
def dimensionNearestTail(integerNonnegative: int, /) -> int:
	"""Find the 0-indexed position of the least significant non-zero radix-2 digit in `integerNonnegative`.

	Because I am using a radix-2 positional-numeral system as a proxy for Cartesian coordinates, this is functionally equivalent
	to computing the number of times `integerNonnegative` is divisible by 2; aka 'CTZ', Count Trailing Zeros in the binary form.
	"""
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return bit_scan1(anInteger) or 0

def Z0Z_0NearestTail(state: EliminationState, integerNonnegative: int) -> int:
	"""Find the 0-indexed position of the least significant ZERO radix-2 digit in `integerNonnegative`."""
# NOTE HEY! `Z0Z_invert` is pulling double duty: it sanitizes `integerNonnegative` and inverts it. So if you figure out how to
# achieve this functionality without calling `Z0Z_invert`, you need to add defensive code here.
	anInteger: int = Z0Z_invert(state.dimensionsTotal, integerNonnegative)
	return bit_scan1(anInteger) or 0

@cache
def howManyDimensionsHaveOddParity(integerNonnegative: int, /) -> int:
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', int), 0)
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return max(0, anInteger.bit_count() - 1)

@cache
def ptount(integerAbove3: int, /) -> int:
	"""After subtracting 一+零 from `integerAbove3`, measure the distance from a ***p***ower of ***t***wo's "bit c***ount***".

	Notes
	-----
	- Pronounced "tount" because the "p" is silent.
	- Just like the "p", the reason why this is useful is silent.
	- I suspect there is a more direct route to measure this but I am unaware of it.
	- I have noticed that 16+3 and 32+3 are often special cases. 16 and 32 have `howMany0coordinatesAtTail` of 4 and 5 respectively.
	- In one case, I was using `ptount(leafAt一零) + 1`, but I directly substituted
		`raiseIfNone(dimensionSecondNearest首(leafAt一零))` for the same results. I created `ptount` well before I created `dimensionSecondNearest首`.
	"""
	anInteger: int = getitem(intInnit([integerAbove3], 'integerAbove3', int), 0)
	if anInteger < 3:
		message: str = f"I received `{integerAbove3 = }`, but I need a value greater than 3."
		raise ValueError(message)

	return leafInSubHyperplane(anInteger - (一+零)).bit_count()



