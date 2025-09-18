"""Count meanders with matrix transfer algorithm."""
from functools import cache
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding import MatrixMeandersNumPyState
import pandas

# NOTE TODO
# TODO Write the code that synthesizes this module. TODO
# NOTE TODO FIXME

def areIntegersWide(state: MatrixMeandersNumPyState, dataframe: pandas.DataFrame | None = None, *, fixedSizeMAXIMUMcurveLocations: bool = False) -> bool:
	"""Check if the largest values are wider than the maximum limits.

	Parameters
	----------
	state : MatrixMeandersState
		The current state of the computation, including `dictionaryCurveLocations`.
	dataframe : pandas.DataFrame | None = None
		Optional DataFrame containing 'analyzed' and 'distinctCrossings' columns. If provided, use this instead of `state.dictionaryCurveLocations`.
	fixedSizeMAXIMUMcurveLocations : bool = False
		Set this to `True` if you cast `state.MAXIMUMcurveLocations` to the same fixed size integer type as `state.datatypeCurveLocations`.

	Returns
	-------
	wider : bool
		True if at least one integer is too wide.

	Notes
	-----
	Casting `state.MAXIMUMcurveLocations` to a fixed-size 64-bit unsigned integer might cause the flow to be a little more
	complicated because `MAXIMUMcurveLocations` is usually 1-bit larger than the `max(curveLocations)` value.

	If you start the algorithm with very large `curveLocations` in your `dictionaryCurveLocations` (*i.e.,* A000682), then the
	flow will go to a function that does not use fixed size integers. When the integers are below the limits (*e.g.,*
	`bitWidthCurveLocationsMaximum`), the flow will go to a function with fixed size integers. In that case, casting
	`MAXIMUMcurveLocations` to a fixed size merely delays the transition from one function to the other by one iteration.

	If you start with small values in `dictionaryCurveLocations`, however, then the flow goes to the function with fixed size
	integers and usually stays there until `distinctCrossings` is huge, which is near the end of the computation. If you cast
	`MAXIMUMcurveLocations` into a 64-bit unsigned integer, however, then around `state.kOfMatrix == 28`, the bit width of
	`MAXIMUMcurveLocations` might exceed the limit. That will cause the flow to go to the function that does not have fixed size
	integers for a few iterations before returning to the function with fixed size integers.
	"""
	if dataframe is None:
		curveLocationsWidest: int = max(state.dictionaryCurveLocations.keys()).bit_length()
		distinctCrossingsWidest: int = max(state.dictionaryCurveLocations.values()).bit_length()
	else:
		curveLocationsWidest = int(dataframe['analyzed'].max()).bit_length()
		distinctCrossingsWidest = int(dataframe['distinctCrossings'].max()).bit_length()

	MAXIMUMcurveLocations: int = 0
	if fixedSizeMAXIMUMcurveLocations:
		MAXIMUMcurveLocations = state.MAXIMUMcurveLocations

	return (curveLocationsWidest > raiseIfNone(state.bitWidthLimitCurveLocations)
		or distinctCrossingsWidest > raiseIfNone(state.bitWidthLimitDistinctCrossings)
		or MAXIMUMcurveLocations > raiseIfNone(state.bitWidthLimitCurveLocations)
		)

def outfitDictionaryBitGroups(state: MatrixMeandersNumPyState) -> dict[tuple[int, int], int]:
	"""Outfit `dictionaryBitGroups` so it may manage the computations for one iteration of the transfer matrix.

	Parameters
	----------
	state : MatrixMeandersState
		The current state of the computation, including `dictionaryCurveLocations`.

	Returns
	-------
	dictionaryBitGroups : dict[tuple[int, int], int]
		A dictionary of `(bitsAlpha, bitsZulu)` to `distinctCrossings`.
	"""
	state.bitWidth = max(state.dictionaryCurveLocations.keys()).bit_length()
	return {(curveLocations & state.locatorBitsAlpha, (curveLocations & state.locatorBitsZulu) >> 1): distinctCrossings
		for curveLocations, distinctCrossings in state.dictionaryCurveLocations.items()}

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
		if (intWithExtra_0b1 & flipExtra_0b1_Here) == 0:
			findTheExtra_0b1 += 1
		else:
			findTheExtra_0b1 -= 1
		if findTheExtra_0b1 < 0:
			break
	return flipExtra_0b1_Here

def count(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Count meanders with matrix transfer algorithm using Python primitive `int` contained in a Python primitive `dict`.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing current `kOfMatrix`, `dictionaryCurveLocations`, and thresholds.

	Notes
	-----
	The algorithm is sophisticated, but this implementation is straightforward. Compute each index one at a time, compute each
	`curveLocations` one at a time, and compute each type of analysis one at a time.
	"""
	dictionaryBitGroups: dict[tuple[int, int], int] = {}

	while (state.kOfMatrix > 0 and areIntegersWide(state)):
		state.kOfMatrix -= 1

		dictionaryBitGroups = outfitDictionaryBitGroups(state)
		state.dictionaryCurveLocations.clear()
		goByeBye()

		for (bitsAlpha, bitsZulu), distinctCrossings in dictionaryBitGroups.items():
			bitsAlphaCurves: bool = bitsAlpha > 1
			bitsZuluHasCurves: bool = bitsZulu > 1
			bitsAlphaIsEven = bitsZuluIsEven = 0

			curveLocationAnalysis = ((bitsAlpha | (bitsZulu << 1)) << 2) | 3
			# simple
			if curveLocationAnalysis < state.MAXIMUMcurveLocations:
				state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if bitsAlphaCurves:
				curveLocationAnalysis = (bitsAlpha >> 2) | (bitsZulu << 3) | ((bitsAlphaIsEven := 1 - (bitsAlpha & 1)) << 1)
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if bitsZuluHasCurves:
				curveLocationAnalysis = (bitsZulu >> 1) | (bitsAlpha << 2) | (bitsZuluIsEven := 1 - (bitsZulu & 1))
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if bitsAlphaCurves and bitsZuluHasCurves and (bitsAlphaIsEven or bitsZuluIsEven):
				# aligned
				if bitsAlphaIsEven and not bitsZuluIsEven:
					bitsAlpha ^= walkDyckPath(bitsAlpha)  # noqa: PLW2901
				elif bitsZuluIsEven and not bitsAlphaIsEven:
					bitsZulu ^= walkDyckPath(bitsZulu)  # noqa: PLW2901

				curveLocationAnalysis: int = ((bitsZulu >> 2) << 1) | (bitsAlpha >> 2)
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

	return state
