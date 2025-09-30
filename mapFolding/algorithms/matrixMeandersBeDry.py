"""Be DRY."""
from functools import cache
from hunterMakesPy import raiseIfNone
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.reference.A000682facts import A000682_n_boundary_buckets
from mapFolding.reference.A005316facts import A005316_n_boundary_buckets
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

def areIntegersWide(state: MatrixMeandersNumPyState, *, dataframe: pandas.DataFrame | None = None, fixedSizeMAXIMUMarcCode: bool = False) -> bool:
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
	`MAXIMUMarcCode` into a 64-bit unsigned integer, however, then around `state.boundary == 28`, the bit width of
	`MAXIMUMarcCode` might exceed the limit. That will cause the flow to go to the function that does not have fixed size
	integers for a few iterations before returning to the function with fixed size integers.
	"""
	if dataframe is not None:
		arcCodeWidest = int(dataframe['analyzed'].max()).bit_length()
		crossingsWidest = int(dataframe['crossings'].max()).bit_length()
	elif not state.dictionaryMeanders:
		arcCodeWidest = int(state.arrayArcCodes.max()).bit_length()
		crossingsWidest = int(state.arrayCrossings.max()).bit_length()
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

def getBucketsTotal(state: MatrixMeandersNumPyState, safetyMultiplicand: float = 1.2) -> int:  # noqa: ARG001
	"""Under renovation: Estimate the total number of non-unique arcCode that will be computed from the existing arcCode.

	Warning
	-------
	Because `matrixMeandersPandas` does not store anything in `state.arrayArcCodes`, if `matrixMeandersPandas` requests
	bucketsTotal for a value not in the dictionary, the returned value will be 0. But `matrixMeandersPandas` should have a safety
	check that will allocate more space.

	Notes
	-----
	TODO remake this function from scratch.

	Factors:
		- The starting quantity of `arcCode`.
		- The value(s) of the starting `arcCode`.
		- n
		- boundary
		- Whether this bucketsTotal is increasing, as compared to all of the prior bucketsTotal.
		- If increasing, is it exponential or logarithmic?
		- The maximum value.
		- If decreasing, I don't really know the factors.
		- If I know the actual value or if I must estimate it.

	Figure out an intelligent flow for so many factors.
	"""
	theDictionary: dict[str, dict[int, dict[int, int]]] = {'A005316': A005316_n_boundary_buckets, 'A000682': A000682_n_boundary_buckets}
	bucketsTotal: int = theDictionary.get(state.oeisID, {}).get(state.n, {}).get(state.boundary, 0)
	if bucketsTotal <= 0:
		bucketsTotal = int(3.55 * len(state.arrayArcCodes))

	return bucketsTotal

def getSignaturesTotal(state: MatrixMeandersNumPyState) -> int:
	"""Get the total number of signatures for the current `n` and `boundary`.

	Parameters
	----------
	state : MatrixMeandersState
		The current state of the computation.

	Returns
	-------
	signaturesTotal : int
		The total number of signatures for the current `n` and `boundary`.

	"""
	from mapFolding.reference.matrixMeandersAnalysis.signatures import signatures  # noqa: PLC0415
	return signatures[state.oeisID].get(state.n, {}).get(state.boundary, int(3.55 * len(state.arrayArcCodes)))

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

