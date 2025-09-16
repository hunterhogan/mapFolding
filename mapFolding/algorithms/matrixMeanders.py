"""Count meanders with matrix transfer algorithm.

Notes
-----
- Odd/even of `groupAlpha` == the odd/even of `curveLocations`. Proof: `groupAlphaIsEven = curveLocations & 1 & 1 ^ 1`.
- Odd/even of `groupZulu` == `curveLocations` second-least significant bit. So `groupZuluIsEven = bool(curveLocations & 2 ^ 2)`.
"""
from functools import cache
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding import MatrixMeandersState
from mapFolding.algorithms.getBucketsTotal import getBucketsTotal
from pathlib import Path
from warnings import warn
import numpy
import pandas

pathRoot: Path = Path.cwd() / 'curves'
pathRoot.mkdir(exist_ok=True, parents=True)

@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def outfitDictionaryCurveGroups(state: MatrixMeandersState) -> dict[tuple[int, int], int]:
	"""Outfit `dictionaryCurveGroups` so it may manage the computations for one iteration of the transfer matrix.

	Parameters
	----------
	state : MatrixMeandersState
		The current state of the computation, including `dictionaryCurveLocations`.

	Returns
	-------
	dictionaryCurveGroups : dict[tuple[int, int], int]
		A dictionary of `(groupAlpha, groupZulu)` to `distinctCrossings`.
	"""
	state.bitWidth = max(state.dictionaryCurveLocations.keys()).bit_length()
	return {(curveLocations & state.locatorGroupAlpha, (curveLocations & state.locatorGroupZulu) >> 1): distinctCrossings
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

def areIntegersWide(state: MatrixMeandersState, dataframe: pandas.DataFrame | None = None, *, fixedSizeMAXIMUMcurveLocations: bool = False) -> bool:
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

	return (curveLocationsWidest > raiseIfNone(state.bitWidthCurveLocationsMaximum)
		or distinctCrossingsWidest > raiseIfNone(state.bitWidthDistinctCrossingsMaximum)
		or MAXIMUMcurveLocations > raiseIfNone(state.bitWidthCurveLocationsMaximum)
		)

def countBigInt(state: MatrixMeandersState) -> MatrixMeandersState:
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
	dictionaryCurveGroups: dict[tuple[int, int], int] = {}

	while (state.kOfMatrix > 0 and areIntegersWide(state)):
		state.kOfMatrix -= 1

		dictionaryCurveGroups = outfitDictionaryCurveGroups(state)
		state.dictionaryCurveLocations.clear()
		goByeBye()

		for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
			groupAlphaCurves: bool = groupAlpha > 1
			groupZuluHasCurves: bool = groupZulu > 1
			groupAlphaIsEven = groupZuluIsEven = 0

			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			# simple
			if curveLocationAnalysis < state.MAXIMUMcurveLocations:
				state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 1)) << 1)
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupZuluHasCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupAlphaCurves and groupZuluHasCurves and (groupAlphaIsEven or groupZuluIsEven):
				# aligned
				if groupAlphaIsEven and not groupZuluIsEven:
					groupAlpha ^= walkDyckPath(groupAlpha)  # noqa: PLW2901
				elif groupZuluIsEven and not groupAlphaIsEven:
					groupZulu ^= walkDyckPath(groupZulu)  # noqa: PLW2901

				curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

	return state

# ruff: noqa: B023

def countPandas(state: MatrixMeandersState) -> MatrixMeandersState:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing current `kOfMatrix`, `dictionaryCurveLocations`, and thresholds.

	Returns
	-------
	state : MatrixMeandersState
		Updated state with new `kOfMatrix` and `dictionaryCurveLocations`.
	"""
	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=state.dictionaryCurveLocations.keys(), copy=False, dtype=state.datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=state.dictionaryCurveLocations.values(), copy=False, dtype=state.datatypeDistinctCrossings)
		}, dtype=state.datatypeCurveLocations
	)
	state.dictionaryCurveLocations.clear()

	while (state.kOfMatrix > 0 and not areIntegersWide(state, dataframeAnalyzed)):

		def aggregateCurveLocations()  -> None:
			nonlocal dataframeAnalyzed
			dataframeAnalyzed = dataframeAnalyzed.iloc[0:state.indexStartAnalyzed].groupby('analyzed', sort=False)['distinctCrossings'].aggregate('sum').reset_index()

		def analyzeCurveLocationsAligned() -> None:
			"""Compute `curveLocations` from `groupAlpha` and `groupZulu` if at least one is an even number.

			Before computing `curveLocations`, some values of `groupAlpha` and `groupZulu` are modified.

			Warning
			-------
			This function deletes rows from `dataframeCurveLocations`. Always run this analysis last.

			Formula
			-------
			```python
			if groupAlpha > 1 and groupZulu > 1 and (groupAlphaIsEven or groupZuluIsEven):
				curveLocations = (groupAlpha >> 2) | ((groupZulu >> 2) << 1)
			```
			"""
			nonlocal dataframeCurveLocations

			# NOTE Step 1 drop unqualified rows

			ImaGroupZulpha: pandas.Series = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
			ImaGroupZulpha &= state.locatorGroupAlpha # Ima `groupAlpha`.

			dataframeCurveLocations = dataframeCurveLocations.loc[(ImaGroupZulpha > 1)] # if groupAlphaHasCurves

			del ImaGroupZulpha

			ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
			ImaGroupZulpha &= state.locatorGroupZulu # Ima `groupZulu`.
			ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

			dataframeCurveLocations = dataframeCurveLocations.loc[(ImaGroupZulpha > 1)] # if groupZuluHasCurves

			del ImaGroupZulpha

			ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
			ImaGroupZulpha &= 0b10 # Ima `groupZulu`.
			ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)
			ImaGroupZulpha &= 1 # (groupZulu & 1)
			ImaGroupZulpha ^= 1 # (1 - (groupZulu ...))
			dataframeCurveLocations.loc[:, 'analyzed'] = ImaGroupZulpha # selectorGroupZuluAtEven

			del ImaGroupZulpha

			ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
			ImaGroupZulpha &= 1 # (groupAlpha & 1)
			ImaGroupZulpha ^= 1 # (1 - (groupAlpha ...))
			ImaGroupZulpha = ImaGroupZulpha.astype(bool) # selectorGroupAlphaAtODD

			dataframeCurveLocations = dataframeCurveLocations.loc[(ImaGroupZulpha) | (dataframeCurveLocations.loc[:, 'analyzed'])] # if (groupAlphaIsEven or groupZuluIsEven)

			del ImaGroupZulpha

			# NOTE Step 2 modify rows

			# Make a selector for groupZuluAtEven, so you can modify groupAlpha
			ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
			ImaGroupZulpha &= 0b10 # Ima `groupZulu`.
			ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)
			ImaGroupZulpha &= 1 # (groupZulu & 1)
			ImaGroupZulpha ^= 1 # (1 - (groupZulu ...))
			ImaGroupZulpha = ImaGroupZulpha.astype(bool) # selectorGroupZuluAtEven

			dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['curveLocations'] # Ima `groupAlpha`.
			dataframeCurveLocations.loc[:, 'analyzed'] &= state.locatorGroupAlpha # (groupAlpha)

			# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
			dataframeCurveLocations.loc[(~ImaGroupZulpha), 'analyzed'] = state.datatypeCurveLocations( # pyright: ignore[reportCallIssue, reportArgumentType]
				flipTheExtra_0b1AsUfunc(dataframeCurveLocations.loc[(~ImaGroupZulpha), 'analyzed']))

			del ImaGroupZulpha

			# if groupZuluIsEven and not groupAlphaIsEven, modifyGroupZuluPairedToOdd
			ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
			ImaGroupZulpha &= state.locatorGroupZulu # Ima `groupZulu`.
			ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

			ImaGroupZulpha.loc[(dataframeCurveLocations.loc[:, 'curveLocations'] & 1).astype(bool)] = state.datatypeCurveLocations( # pyright: ignore[reportArgumentType, reportCallIssue]
				flipTheExtra_0b1AsUfunc(ImaGroupZulpha.loc[(dataframeCurveLocations.loc[:, 'curveLocations'] & 1).astype(bool)])) # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]

			# NOTE Step 3 compute curveLocations

			dataframeCurveLocations.loc[:, 'analyzed'] //= 2**2 # (groupAlpha >> 2)

			ImaGroupZulpha //= 2**2 # (groupZulu >> 2)
			ImaGroupZulpha *= 2**1 # ((groupZulu ...) << 1)

			dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupZulu ...)

			del ImaGroupZulpha

			dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		def analyzeCurveLocationsAlpha() -> None:
			"""Compute `curveLocations` from `groupAlpha`.

			Formula
			-------
			```python
			if groupAlpha > 1:
				curveLocations = ((1 - (groupAlpha & 1)) << 1) | (groupZulu << 3) | (groupAlpha >> 2)
			# `(1 - (groupAlpha & 1)` is an evenness test.
			```
			"""
			nonlocal dataframeCurveLocations
			dataframeCurveLocations['analyzed'] = dataframeCurveLocations['curveLocations']
			dataframeCurveLocations.loc[:, 'analyzed'] &= 1 # (groupAlpha & 1)
			dataframeCurveLocations.loc[:, 'analyzed'] ^= 1 # (1 - (groupAlpha ...))

			dataframeCurveLocations.loc[:, 'analyzed'] *= 2**1 # ((groupAlpha ...) << 1)

			ImaGroupZulpha: pandas.Series = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
			ImaGroupZulpha &= state.locatorGroupZulu # Ima `groupZulu`.
			ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

			ImaGroupZulpha *= 2**3 # (groupZulu << 3)
			dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupZulu ...)

			del ImaGroupZulpha

			dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # ... | (groupAlpha >> 2)

			ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
			ImaGroupZulpha &= state.locatorGroupAlpha # Ima `groupAlpha`.

			dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupAlpha)
			dataframeCurveLocations.loc[:, 'analyzed'] //= 2**2 # (... >> 2)

			dataframeCurveLocations.loc[(ImaGroupZulpha <= 1), 'analyzed'] = 0 # if groupAlpha > 1

			del ImaGroupZulpha

			dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		def analyzeCurveLocationsSimple() -> None:
			"""Compute curveLocations with the 'simple' bridges formula.

			Formula
			-------
			```python
			curveLocations = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			```

			Notes
			-----
			Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the last bits are '0b00'. '0 + 3'
			is '0b11', and '0b00 | 0b11' is also '0b11'.

			"""
			nonlocal dataframeCurveLocations
			dataframeCurveLocations['analyzed'] = dataframeCurveLocations['curveLocations']
			dataframeCurveLocations.loc[:, 'analyzed'] &= state.locatorGroupAlpha

			groupZulu: pandas.Series = dataframeCurveLocations['curveLocations'].copy()
			groupZulu &= state.locatorGroupZulu
			groupZulu //= 2**1 # (groupZulu >> 1)
			groupZulu *= 2**1 # (groupZulu << 1)

			dataframeCurveLocations.loc[:, 'analyzed'] |= groupZulu # ((groupAlpha | (groupZulu ...))

			del groupZulu

			dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # (... << 2)
			dataframeCurveLocations.loc[:, 'analyzed'] += 3 # (...) | 3
			dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		def analyzeCurveLocationsZulu() -> None:
			"""Compute `curveLocations` from `groupZulu`.

			Formula
			-------
			```python
			if groupZulu > 1:
				curveLocations = (1 - (groupZulu & 1)) | (groupAlpha << 2) | (groupZulu >> 1)
			```
			"""
			nonlocal dataframeCurveLocations
			dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['curveLocations'] # Ima `groupZulu`.
			dataframeCurveLocations.loc[:, 'analyzed'] &= 0b10 # Ima `groupZulu`.
			dataframeCurveLocations.loc[:, 'analyzed'] //= 2**1 # Ima `groupZulu` (groupZulu >> 1)
			dataframeCurveLocations.loc[:, 'analyzed'] &= 1 # (groupZulu & 1)
			dataframeCurveLocations.loc[:, 'analyzed'] ^= 1 # (1 - (groupZulu ...))

			ImaGroupZulpha: pandas.Series = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
			ImaGroupZulpha &= state.locatorGroupAlpha # Ima `groupAlpha`.

			ImaGroupZulpha *= 2**2 # (groupAlpha << 2)
			dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupAlpha ...)

			del ImaGroupZulpha

			ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
			ImaGroupZulpha &= state.locatorGroupZulu # Ima `groupZulu`.
			ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

			ImaGroupZulpha //= 2**1 # (groupZulu >> 1)

			dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupZulu ...)

			del ImaGroupZulpha

			ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
			ImaGroupZulpha &= state.locatorGroupZulu # Ima `groupZulu`.
			ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

			dataframeCurveLocations.loc[ImaGroupZulpha <= 1, 'analyzed'] = 0 # if groupZulu > 1

			del ImaGroupZulpha

			dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		def recordCurveLocations() -> None:
			nonlocal dataframeAnalyzed

			indexStopAnalyzed: int = state.indexStartAnalyzed + int((dataframeCurveLocations['analyzed'] > 0).sum()) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

			if indexStopAnalyzed > state.indexStartAnalyzed:
				if len(dataframeAnalyzed.index) < indexStopAnalyzed:
					warn(f"Lengthened `dataframeAnalyzed` from {len(dataframeAnalyzed.index)} to {indexStopAnalyzed=}; n={state.n}, {state.kOfMatrix=}.", stacklevel=2)
					dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)

				dataframeAnalyzed.loc[state.indexStartAnalyzed:indexStopAnalyzed - 1, ['analyzed', 'distinctCrossings']] = (
					dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0), ['analyzed', 'distinctCrossings']
								].to_numpy(dtype=state.datatypeCurveLocations, copy=False)
				)

				state.indexStartAnalyzed = indexStopAnalyzed

			del indexStopAnalyzed

		dataframeCurveLocations = pandas.DataFrame({
			'curveLocations': pandas.Series(name='curveLocations', data=dataframeAnalyzed['analyzed'], copy=False, dtype=state.datatypeCurveLocations)
			, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=state.datatypeCurveLocations)
			, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=dataframeAnalyzed['distinctCrossings'], copy=False, dtype=state.datatypeDistinctCrossings)
			} # pyright: ignore[reportUnknownArgumentType]
		)

		del dataframeAnalyzed
		goByeBye()

		state.bitWidth = int(dataframeCurveLocations['curveLocations'].max()).bit_length()
		length: int = getBucketsTotal(state)
		dataframeAnalyzed = pandas.DataFrame({
			'analyzed': pandas.Series(0, pandas.RangeIndex(length), dtype=state.datatypeCurveLocations, name='analyzed')
			, 'distinctCrossings': pandas.Series(0, pandas.RangeIndex(length), dtype=state.datatypeDistinctCrossings, name='distinctCrossings')
			}, index=pandas.RangeIndex(length), columns=['analyzed', 'distinctCrossings'], dtype=state.datatypeCurveLocations # pyright: ignore[reportUnknownArgumentType]
		)

		state.kOfMatrix -= 1

		state.indexStartAnalyzed = 0

		analyzeCurveLocationsSimple()
		recordCurveLocations()

		analyzeCurveLocationsAlpha()
		recordCurveLocations()

		analyzeCurveLocationsZulu()
		recordCurveLocations()

		analyzeCurveLocationsAligned()
		recordCurveLocations()
		del dataframeCurveLocations
		goByeBye()

		aggregateCurveLocations()

		if state.n >= 45:  # for data collection
			print(state.n, state.kOfMatrix+1, state.indexStartAnalyzed, sep=',')  # noqa: T201

	state.dictionaryCurveLocations = dataframeAnalyzed.set_index('analyzed')['distinctCrossings'].to_dict()
	return state

def doTheNeedful(state: MatrixMeandersState) -> int:
	"""Compute a(n) meanders with the transfer matrix algorithm.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing current `kOfMatrix`, `dictionaryCurveLocations`, and thresholds.

	Returns
	-------
	a(n) : int
		The computed value of a(n).

	Notes
	-----
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex

	See Also
	--------
	https://oeis.org/A000682
	https://oeis.org/A005316
	"""
	while state.kOfMatrix > 0:
		goByeBye()

		if areIntegersWide(state):
			state = countBigInt(state)
		else:
			state = countPandas(state)

	return sum(state.dictionaryCurveLocations.values())

