from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from warnings import warn
import pandas

# TODO investigate adding another condition to `areIntegersWide`: while dict is faster than pandas, stay in bigInt.

# ruff: noqa: B023

def countPandas(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
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

	while (state.kOfMatrix > 0 and not areIntegersWide(state, dataframe=dataframeAnalyzed)):

		def aggregateCurveLocations()  -> None:
			nonlocal dataframeAnalyzed
			dataframeAnalyzed = dataframeAnalyzed.iloc[0:state.indexTarget].groupby('analyzed', sort=False)['distinctCrossings'].aggregate('sum').reset_index()

		def analyzeCurveLocationsAligned() -> None:
			"""Compute `curveLocations` from `bitsAlpha` and `bitsZulu` if at least one is an even number.

			Before computing `curveLocations`, some values of `bitsAlpha` and `bitsZulu` are modified.

			Warning
			-------
			This function deletes rows from `dataframeCurveLocations`. Always run this analysis last.

			Formula
			-------
			```python
			if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven):
				curveLocations = (bitsAlpha >> 2) | ((bitsZulu >> 2) << 1)
			```
			"""
			nonlocal dataframeCurveLocations

			# NOTE Step 1 drop unqualified rows

			bitsTarget: pandas.Series = dataframeCurveLocations['curveLocations'].copy() # `bitsAlpha`
			bitsTarget &= state.locatorBitsAlpha # `bitsAlpha`

			dataframeCurveLocations = dataframeCurveLocations.loc[(bitsTarget > 1)] # if bitsAlphaHasCurves

			del bitsTarget

			bitsTarget = dataframeCurveLocations['curveLocations'].copy() # `bitsZulu`
			bitsTarget &= state.locatorBitsZulu # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)

			dataframeCurveLocations = dataframeCurveLocations.loc[(bitsTarget > 1)] # if bitsZuluHasCurves

			del bitsTarget

			bitsTarget = dataframeCurveLocations['curveLocations'].copy() # `bitsZulu`
			bitsTarget &= 0b10 # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= 1 # (bitsZulu & 1)
			bitsTarget ^= 1 # (1 - (bitsZulu ...))
			dataframeCurveLocations.loc[:, 'analyzed'] = bitsTarget # selectorBitsZuluAtEven

			del bitsTarget

			bitsTarget = dataframeCurveLocations['curveLocations'].copy() # `bitsAlpha`
			bitsTarget &= 1 # (bitsAlpha & 1)
			bitsTarget ^= 1 # (1 - (bitsAlpha ...))
			bitsTarget = bitsTarget.astype(bool) # selectorBitsAlphaAtODD

			dataframeCurveLocations = dataframeCurveLocations.loc[(bitsTarget) | (dataframeCurveLocations.loc[:, 'analyzed'])] # if (bitsAlphaIsEven or bitsZuluIsEven)

			del bitsTarget

			# NOTE Step 2 modify rows

			# Make a selector for bitsZuluAtEven, so you can modify bitsAlpha
			bitsTarget = dataframeCurveLocations['curveLocations'].copy() # `bitsZulu`
			bitsTarget &= 0b10 # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= 1 # (bitsZulu & 1)
			bitsTarget ^= 1 # (1 - (bitsZulu ...))
			bitsTarget = bitsTarget.astype(bool) # selectorBitsZuluAtEven

			dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['curveLocations'] # `bitsAlpha`
			dataframeCurveLocations.loc[:, 'analyzed'] &= state.locatorBitsAlpha # `bitsAlpha`

			# if bitsAlphaIsEven and not bitsZuluIsEven, modify bitsAlphaPairedToOdd
			dataframeCurveLocations.loc[(~bitsTarget), 'analyzed'] = state.datatypeCurveLocations( # pyright: ignore[reportCallIssue, reportArgumentType]
				flipTheExtra_0b1AsUfunc(dataframeCurveLocations.loc[(~bitsTarget), 'analyzed']))

			del bitsTarget

			# if bitsZuluIsEven and not bitsAlphaIsEven, modify bitsZuluPairedToOdd
			bitsTarget = dataframeCurveLocations['curveLocations'].copy() # `bitsZulu`
			bitsTarget &= state.locatorBitsZulu # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)

			bitsTarget.loc[(dataframeCurveLocations.loc[:, 'curveLocations'] & 1).astype(bool)] = state.datatypeCurveLocations( # pyright: ignore[reportArgumentType, reportCallIssue]
				flipTheExtra_0b1AsUfunc(bitsTarget.loc[(dataframeCurveLocations.loc[:, 'curveLocations'] & 1).astype(bool)])) # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]

			# NOTE Step 3 compute curveLocations

			dataframeCurveLocations.loc[:, 'analyzed'] //= 2**2 # (bitsAlpha >> 2)

			bitsTarget //= 2**2 # (bitsZulu >> 2)
			bitsTarget *= 2**1 # ((bitsZulu ...) << 1)

			dataframeCurveLocations.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsZulu ...)

			del bitsTarget

			dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		def analyzeCurveLocationsAlpha() -> None:
			"""Compute `curveLocations` from `bitsAlpha`.

			Formula
			-------
			```python
			if bitsAlpha > 1:
				curveLocations = ((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) | (bitsAlpha >> 2)
			# `(1 - (bitsAlpha & 1)` is an evenness test.
			```
			"""
			nonlocal dataframeCurveLocations
			dataframeCurveLocations['analyzed'] = dataframeCurveLocations['curveLocations']
			dataframeCurveLocations.loc[:, 'analyzed'] &= 1 # (bitsAlpha & 1)
			dataframeCurveLocations.loc[:, 'analyzed'] ^= 1 # (1 - (bitsAlpha ...))

			dataframeCurveLocations.loc[:, 'analyzed'] *= 2**1 # ((bitsAlpha ...) << 1)

			bitsTarget: pandas.Series = dataframeCurveLocations['curveLocations'].copy() # `bitsZulu`
			bitsTarget &= state.locatorBitsZulu # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)

			bitsTarget *= 2**3 # (bitsZulu << 3)
			dataframeCurveLocations.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsZulu ...)

			del bitsTarget

			"""NOTE In this code block, I rearranged the "formula" to use `bitsTarget` for two goals. 1. `(bitsAlpha >> 2)`.
			2. `if bitsAlpha > 1`. The trick is in the equivalence of v1 and v2.
				v1: BITScow | (BITSwalk >> 2)
				v2: ((BITScow << 2) | BITSwalk) >> 2

			The "formula" calls for v1, but by using v2, `bitsTarget` is not changed. Therefore, because `bitsTarget` is
			`bitsAlpha`, I can use `bitsTarget` for goal 2, `if bitsAlpha > 1`.
			"""
			dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # ... | (bitsAlpha >> 2)

			bitsTarget = dataframeCurveLocations['curveLocations'].copy() # `bitsAlpha`
			bitsTarget &= state.locatorBitsAlpha # `bitsAlpha`

			dataframeCurveLocations.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsAlpha)
			dataframeCurveLocations.loc[:, 'analyzed'] //= 2**2 # (... >> 2)

			dataframeCurveLocations.loc[(bitsTarget <= 1), 'analyzed'] = 0 # if bitsAlpha > 1

			del bitsTarget

			dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		def analyzeCurveLocationsSimple() -> None:
			"""Compute curveLocations with the 'simple' formula.

			Formula
			-------
			```python
			curveLocations = ((bitsAlpha | (bitsZulu << 1)) << 2) | 3
			```

			Notes
			-----
			Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the last bits are '0b00'. '0 + 3'
			is '0b11', and '0b00 | 0b11' is also '0b11'.

			"""
			nonlocal dataframeCurveLocations
			dataframeCurveLocations['analyzed'] = dataframeCurveLocations['curveLocations']
			dataframeCurveLocations.loc[:, 'analyzed'] &= state.locatorBitsAlpha

			bitsZulu: pandas.Series = dataframeCurveLocations['curveLocations'].copy()
			bitsZulu &= state.locatorBitsZulu
			bitsZulu //= 2**1 # (bitsZulu >> 1)
			bitsZulu *= 2**1 # (bitsZulu << 1)

			dataframeCurveLocations.loc[:, 'analyzed'] |= bitsZulu # ((bitsAlpha | (bitsZulu ...))

			del bitsZulu

			dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # (... << 2)
			dataframeCurveLocations.loc[:, 'analyzed'] += 3 # (...) | 3
			dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		def analyzeCurveLocationsZulu() -> None:
			"""Compute `curveLocations` from `bitsZulu`.

			Formula
			-------
			```python
			if bitsZulu > 1:
				curveLocations = (1 - (bitsZulu & 1)) | (bitsAlpha << 2) | (bitsZulu >> 1)
			```
			"""
			nonlocal dataframeCurveLocations
			dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['curveLocations'] # `bitsZulu`
			dataframeCurveLocations.loc[:, 'analyzed'] &= 0b10 # `bitsZulu`
			dataframeCurveLocations.loc[:, 'analyzed'] //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			dataframeCurveLocations.loc[:, 'analyzed'] &= 1 # (bitsZulu & 1)
			dataframeCurveLocations.loc[:, 'analyzed'] ^= 1 # (1 - (bitsZulu ...))

			bitsTarget: pandas.Series = dataframeCurveLocations['curveLocations'].copy() # `bitsAlpha`
			bitsTarget &= state.locatorBitsAlpha # `bitsAlpha`

			bitsTarget *= 2**2 # (bitsAlpha << 2)
			dataframeCurveLocations.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsAlpha ...)

			del bitsTarget

			# NOTE No, IDK why I didn't use the same trick as in `analyzeCurveLocationsAlpha`. I _think_ I wrote this code before I figured out that trick.
			bitsTarget = dataframeCurveLocations['curveLocations'].copy() # `bitsZulu`
			bitsTarget &= state.locatorBitsZulu # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)

			bitsTarget //= 2**1 # (bitsZulu >> 1)

			dataframeCurveLocations.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsZulu ...)

			del bitsTarget

			bitsTarget = dataframeCurveLocations['curveLocations'].copy() # `bitsZulu`
			bitsTarget &= state.locatorBitsZulu # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)

			dataframeCurveLocations.loc[bitsTarget <= 1, 'analyzed'] = 0 # if bitsZulu > 1

			del bitsTarget

			dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		def recordCurveLocations() -> None:
			nonlocal dataframeAnalyzed

			indexStopAnalyzed: int = state.indexTarget + int((dataframeCurveLocations['analyzed'] > 0).sum()) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

			if indexStopAnalyzed > state.indexTarget:
				if len(dataframeAnalyzed.index) < indexStopAnalyzed:
					warn(f"Lengthened `dataframeAnalyzed` from {len(dataframeAnalyzed.index)} to {indexStopAnalyzed=}; n={state.n}, {state.kOfMatrix=}.", stacklevel=2)
					dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)

				dataframeAnalyzed.loc[state.indexTarget:indexStopAnalyzed - 1, ['analyzed', 'distinctCrossings']] = (
					dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0), ['analyzed', 'distinctCrossings']
								].to_numpy(dtype=state.datatypeCurveLocations, copy=False)
				)

				state.indexTarget = indexStopAnalyzed

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

		state.indexTarget = 0

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
			print(state.n, state.kOfMatrix+1, state.indexTarget, sep=',')  # noqa: T201

	state.dictionaryCurveLocations = dataframeAnalyzed.set_index('analyzed')['distinctCrossings'].to_dict()
	return state

def doTheNeedful(state: MatrixMeandersNumPyState) -> int:
	"""Compute `distinctCrossings` with a transfer matrix algorithm implemented in pandas.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	distinctCrossings : int
		The computed value of `distinctCrossings`.

	Notes
	-----
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex

	See Also
	--------
	https://oeis.org/A000682
	https://oeis.org/A005316
	"""
	while state.kOfMatrix > 0:
		if areIntegersWide(state):
			state = countBigInt(state)
		else:
			state = countPandas(state)

		goByeBye()

	return sum(state.dictionaryCurveLocations.values())
