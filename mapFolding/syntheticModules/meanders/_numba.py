from __future__ import annotations

from mapFolding.theTypes import 形ArcCode
import numba

#=SIN= Pyright suppression: `numba.vectorize` is partially unknown.
@numba.vectorize((f"{形ArcCode.__name__}({形ArcCode.__name__})",), cache=True, nopython=True)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
def flipTheExtra_0b1(intWithExtra_0b1: 形ArcCode) -> 形ArcCode:
	"""Flip a bit based on Dyck path with a Numba-generated universal function [1].

	You can call `flipTheExtra_0b1` with a `numpy.uint64`, a `numpy.ndarray` [2], or a
	`pandas.Series` [3] that contains the fixed-width arc-code representation.

	Warning
	-------
	The function will loop infinitely if _any_ element does not have a bit that needs flipping.

	Parameters
	----------
	intWithExtra_0b1 : numpy.uint64 | numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] | pandas.Series
		One arc code or a container of arc codes with unbalanced closures.

	Returns
	-------
	flipped : numpy.uint64 | numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] | pandas.Series
		The same scalar or container representation with one bit flipped in each arc code.

	References
	----------
	[1] Numba - Creating NumPy universal functions
		https://numba.readthedocs.io/en/stable/user/vectorize.html
	[2] NumPy - Universal functions
		https://numpy.org/doc/stable/reference/ufuncs.html
	[3] pandas.Series
		https://pandas.pydata.org/docs/reference/api/pandas.Series.html
	"""
	return intWithExtra_0b1 ^ walkDyckPath(intWithExtra_0b1)

@numba.jit(cache=True, nopython=True)
def walkDyckPath(intWithExtra_0b1: 形ArcCode) -> 形ArcCode:
	findTheExtra_0b1 = 0
	flipExtra_0b1_Here = 1
	while 0 <= findTheExtra_0b1:
		flipExtra_0b1_Here <<= 2
		if intWithExtra_0b1 & flipExtra_0b1_Here == 0:
			findTheExtra_0b1 += 1
		else:
			findTheExtra_0b1 -= 1
	return 形ArcCode(flipExtra_0b1_Here)
