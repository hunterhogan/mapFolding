"""Be DRY."""
from functools import cache
from hunterMakesPy import raiseIfNone
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.reference.A000682facts import A000682_n_k_buckets
from mapFolding.reference.A005316facts import (
	A005316_n_k_buckets, bucketsIf_k_EVEN_by_nLess_k, bucketsIf_k_ODD_by_nLess_k)
from math import exp, log
from typing import Any, NamedTuple
import math
import numpy
import pandas

"""Goals:
- Extreme abstraction.
- Find operations with latent intermediate arrays and make the intermediate array explicit.
- Reduce or eliminate intermediate arrays and selector arrays.
- Write formulas in prefix notation.
- For each formula, find an equivalent prefix notation formula that never uses the same variable as input more than once: that
	would allow the evaluation of the expression with only a single stack, which saves memory.
- Standardize code as much as possible to create duplicate code.
- Convert duplicate code to procedures.
"""

class ImaKey(NamedTuple):
	"""keys for dictionaries."""

	oeisID: str
	kIsOdd: bool
	nLess_kIsOdd: bool

def areIntegersWide(state: MatrixMeandersNumPyState, *, dataframe: pandas.DataFrame | None = None, array: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]] | None = None, fixedSizeMAXIMUMarcCode: bool = False) -> bool:
	"""Check if the largest values are wider than the maximum limits.

	Parameters
	----------
	state : MatrixMeandersState
		The current state of the computation, including `dictionaryMeanders`.
	dataframe : pandas.DataFrame | None = None
		Optional DataFrame containing 'analyzed' and 'crossings' columns. If provided, use this instead of `state.dictionaryMeanders`.
	fixedSizeMAXIMUMarcCode : bool = False
		Set this to `True` if you cast `state.MAXIMUMarcCode` to the same fixed size integer type as `state.datatypeArcCode`.

	Returns
	-------
	wider : bool
		True if at least one integer is too wide.

	Notes
	-----
	Casting `state.MAXIMUMarcCode` to a fixed-size 64-bit unsigned integer might cause the flow to be a little more
	complicated because `MAXIMUMarcCode` is usually 1-bit larger than the `max(arcCode)` value.

	If you start the algorithm with very large `arcCode` in your `dictionaryMeanders` (*i.e.,* A000682), then the
	flow will go to a function that does not use fixed size integers. When the integers are below the limits (*e.g.,*
	`bitWidthArcCodeMaximum`), the flow will go to a function with fixed size integers. In that case, casting
	`MAXIMUMarcCode` to a fixed size merely delays the transition from one function to the other by one iteration.

	If you start with small values in `dictionaryMeanders`, however, then the flow goes to the function with fixed size
	integers and usually stays there until `crossings` is huge, which is near the end of the computation. If you cast
	`MAXIMUMarcCode` into a 64-bit unsigned integer, however, then around `state.kOfMatrix == 28`, the bit width of
	`MAXIMUMarcCode` might exceed the limit. That will cause the flow to go to the function that does not have fixed size
	integers for a few iterations before returning to the function with fixed size integers.
	"""
	if dataframe is not None:
		arcCodeWidest = int(dataframe['analyzed'].max()).bit_length()
		crossingsWidest = int(dataframe['crossings'].max()).bit_length()
	elif array is not None:
		arcCodeWidest = int(array[state.slicerArcCode].max()).bit_length()
		crossingsWidest = int(array[state.slicerCrossings].max()).bit_length()
	elif not state.dictionaryMeanders:
		arcCodeWidest = int(state.arrayMeanders[state.slicerArcCode].max()).bit_length()
		crossingsWidest = int(state.arrayMeanders[state.slicerCrossings].max()).bit_length()
	else:
		arcCodeWidest: int = max(state.dictionaryMeanders.keys()).bit_length()
		crossingsWidest: int = max(state.dictionaryMeanders.values()).bit_length()

	MAXIMUMarcCode: int = 0
	if fixedSizeMAXIMUMarcCode:
		MAXIMUMarcCode = state.MAXIMUMarcCode

	return (arcCodeWidest > raiseIfNone(state.bitWidthLimitArcCode)
		or crossingsWidest > raiseIfNone(state.bitWidthLimitCrossings)
		or MAXIMUMarcCode > raiseIfNone(state.bitWidthLimitArcCode)
		)

@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)
"""Flip a bit based on Dyck path: element-wise ufunc (*u*niversal *func*tion) for a NumPy `ndarray` (*Num*erical *Py*thon *n-d*imensional array).

Warning
-------
The function will loop infinitely if an element does not have a bit that needs flipping.

Parameters
----------
arrayTarget : numpy.ndarray[tuple[int], numpy.dtype[numpy.unsignedinteger[Any]]]
	An array with one axis of unsigned integers and unbalanced closures.

Returns
-------
arrayFlipped : numpy.ndarray[tuple[int], numpy.dtype[numpy.unsignedinteger[Any]]]
	An array with the same shape as `arrayTarget` but with one bit flipped in each element.
"""

def getBucketsTotal(state: MatrixMeandersNumPyState, safetyMultiplicand: float = 1.2) -> int:
	"""Estimate the total number of non-unique arcCode that will be computed from the existing arcCode.

	Notes
	-----
	TODO remake this function from scratch.

	Factors:
		- The starting quantity of `arcCode`.
		- The value(s) of the starting `arcCode`.
		- n
		- k
		- Whether this bucketsTotal is increasing, as compared to all of the prior bucketsTotal.
		- If increasing, is it exponential or logarithmic?
		- The maximum value.
		- If decreasing, I don't really know the factors.
		- If I know the actual value or if I must estimate it.

	Figure out an intelligent flow for so many factors.
	"""
	theDictionary: dict[str, dict[int, dict[int, int]]] = {'A005316': A005316_n_k_buckets, 'A000682': A000682_n_k_buckets}
	bucketsTotal: int = theDictionary.get(state.oeisID, {}).get(state.n, {}).get(state.kOfMatrix, -8)
	if bucketsTotal > 0:
		return bucketsTotal

	dictionaryExponentialCoefficients: dict[ImaKey, float] = {
		(ImaKey(oeisID='', kIsOdd=False, nLess_kIsOdd=True)): 0.834,
		(ImaKey(oeisID='', kIsOdd=False, nLess_kIsOdd=False)): 1.5803,
		(ImaKey(oeisID='', kIsOdd=True, nLess_kIsOdd=True)): 1.556,
		(ImaKey(oeisID='', kIsOdd=True, nLess_kIsOdd=False)): 1.8047,
	}

	logarithmicOffsets: dict[ImaKey, float] ={
		(ImaKey('A000682', kIsOdd=False, nLess_kIsOdd=False)): 0.0,

		(ImaKey('A000682', kIsOdd=False, nLess_kIsOdd=True)): -0.07302547148212568,
		(ImaKey('A000682', kIsOdd=True, nLess_kIsOdd=False)): -0.00595307513938792,
		(ImaKey('A000682', kIsOdd=True, nLess_kIsOdd=True)): -0.012201222865243722,
		(ImaKey('A005316', kIsOdd=False, nLess_kIsOdd=False)): -0.6392728422078733,
		(ImaKey('A005316', kIsOdd=False, nLess_kIsOdd=True)): -0.6904925299923548,
		(ImaKey('A005316', kIsOdd=True, nLess_kIsOdd=False)): 0.0,
		(ImaKey('A005316', kIsOdd=True, nLess_kIsOdd=True)): 0.0,
	}

	logarithmicParameters: dict[str, float] = {
		'intercept': -166.1750299793178,
		'log(n)': 1259.0051001675547,
		'log(k)': -396.4306071056408,
		'log(nLess_k)': -854.3309503739766,
		'k/n': 716.530410654819,
		'(k/n)^2': -2527.035113444166,
		'normalized k': -882.7054406339189,
	}

	bucketsTotalMaximumBy_kOfMatrix: dict[int, int] = {1:3, 2:12, 3:40, 4:125, 5:392, 6:1254, 7:4087, 8:13623, 9:46181, 10:159137, 11:555469, 12:1961369, 13:6991893, 14:25134208}

	xCommon = 1.57

	nLess_k: int = state.n - state.kOfMatrix
	kIsOdd: bool = bool(state.kOfMatrix & 1)
	nLess_kIsOdd: bool = bool(nLess_k & 1)
	kIsEven: bool = not kIsOdd

	bucketsTotalAtMaximum: bool = state.kOfMatrix <= ((state.n - 1 - (state.kOfMatrix % 2)) // 3)
	bucketsTotalGrowsExponentially: bool = state.kOfMatrix > nLess_k
	bucketsTotalGrowsLogarithmically: bool = state.kOfMatrix > ((state.n - (state.n % 3)) // 3)

	if bucketsTotalAtMaximum:
		if state.kOfMatrix in bucketsTotalMaximumBy_kOfMatrix:
			bucketsTotal = bucketsTotalMaximumBy_kOfMatrix[state.kOfMatrix]
		else:
			c = 0.95037
			r = 3.3591258254
			if kIsOdd:
				c = 0.92444
				r = 3.35776
			bucketsTotal = int(c * r**state.kOfMatrix * safetyMultiplicand)

	elif state.kOfMatrix <= max(bucketsTotalMaximumBy_kOfMatrix.keys()):
		# If `kOfMatrix` is low, use maximum bucketsTotal. 1. Can't underestimate. 2. Skip computation that can underestimate.
		# 3. The potential difference in memory use is relatively small.
		bucketsTotal = bucketsTotalMaximumBy_kOfMatrix[state.kOfMatrix]

	elif bucketsTotalGrowsExponentially:
		if (state.oeisID == 'A005316') and kIsOdd and (nLess_k in bucketsIf_k_ODD_by_nLess_k):
			# If I already know bucketsTotal.
			bucketsTotal = bucketsIf_k_ODD_by_nLess_k[nLess_k]
		elif (state.oeisID == 'A005316') and kIsEven and (nLess_k in bucketsIf_k_EVEN_by_nLess_k):
			# If I already know bucketsTotal.
			bucketsTotal = bucketsIf_k_EVEN_by_nLess_k[nLess_k]
		else: # I estimate bucketsTotal during exponential growth.
			xInstant: int = math.ceil(nLess_k / 2)
			A000682adjustStartingArcCode: float = 0.25
			startingConditionsCoefficient: float = dictionaryExponentialCoefficients[ImaKey('', kIsOdd, nLess_kIsOdd)]
			if kIsOdd and nLess_kIsOdd:
				A000682adjustStartingArcCode = 0.0
			if state.oeisID == 'A000682': # NOTE Net effect is between `*= n` and `*= n * 2.2` if n=46.
				startingConditionsCoefficient *= state.n * (((state.n // 2) + 2) ** A000682adjustStartingArcCode)
			bucketsTotal = int(startingConditionsCoefficient * math.exp(xCommon * xInstant))

	elif bucketsTotalGrowsLogarithmically:
		xPower: float = (0
			+ logarithmicParameters['intercept']
			+ logarithmicParameters['log(n)'] * log(state.n)
			+ logarithmicParameters['log(k)'] * log(state.kOfMatrix)
			+ logarithmicParameters['log(nLess_k)'] * log(nLess_k)
			+ logarithmicParameters['k/n'] * (state.kOfMatrix / state.n)
			+ logarithmicParameters['(k/n)^2'] * (state.kOfMatrix / state.n)**2
			+ logarithmicParameters['normalized k'] * nLess_k / state.n
			+ logarithmicOffsets[ImaKey(state.oeisID, kIsOdd, nLess_kIsOdd)]
		)

		bucketsTotal = int(exp(xPower * safetyMultiplicand))

	else:
		message = "I shouldn't be here."
		raise SystemError(message)
	return bucketsTotal

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
	"""Find the bit position for flipping paired curve endpoints in meander transfer matrices.

	Parameters
	----------
	intWithExtra_0b1 : int
		Binary representation of curve locations with an extra bit encoding parity information.

	Returns
	-------
	flipExtra_0b1_Here : int
		Bit mask indicating the position where the balance condition fails, formatted as 2^(2k).

	3L33T H@X0R
	------------
	Binary search for first negative balance in shifted bit pairs. Returns 2^(2k) mask for
	bit position k where cumulative balance counter transitions from non-negative to negative.

	Mathematics
	-----------
	Implements the Dyck path balance verification algorithm from Jensen's transfer matrix
	enumeration. Computes the position where ∑(i=0 to k) (-1)^b_i < 0 for the first time,
	where b_i are the bits of the input at positions 2i.

	"""
	findTheExtra_0b1: int = 0
	flipExtra_0b1_Here: int = 1
	while True:
		flipExtra_0b1_Here <<= 2
		if intWithExtra_0b1 & flipExtra_0b1_Here == 0:
			findTheExtra_0b1 += 1
		else:
			findTheExtra_0b1 -= 1
		if findTheExtra_0b1 < 0:
			break
	return flipExtra_0b1_Here
