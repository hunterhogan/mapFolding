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
		The algorithm state containing current `kOfMatrix`, `dictionaryMeanders`, and thresholds.

	Returns
	-------
	state : MatrixMeandersState
		Updated state with new `kOfMatrix` and `dictionaryMeanders`.
	"""
	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=state.dictionaryMeanders.keys(), copy=False, dtype=state.datatypeArcCode)
		, 'crossings': pandas.Series(name='crossings', data=state.dictionaryMeanders.values(), copy=False, dtype=state.datatypeCrossings)
		}, dtype=state.datatypeArcCode
	)
	state.dictionaryMeanders.clear()

	while (state.kOfMatrix > 0 and not areIntegersWide(state, dataframe=dataframeAnalyzed)):

		def aggregateArcCodes()  -> None:
			nonlocal dataframeAnalyzed
			dataframeAnalyzed = dataframeAnalyzed.iloc[0:state.indexTarget].groupby('analyzed', sort=False)['crossings'].aggregate('sum').reset_index()

		def analyzeArcCodesAligned() -> None:
			"""Compute `arcCode` from `bitsAlpha` and `bitsZulu` if at least one is an even number.

			Before computing `arcCode`, some values of `bitsAlpha` and `bitsZulu` are modified.

			Warning
			-------
			This function deletes rows from `dataframeMeanders`. Always run this analysis last.

			Formula
			-------
			```python
			if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven):
				arcCode = (bitsAlpha >> 2) | ((bitsZulu >> 2) << 1)
			```
			"""
			nonlocal dataframeMeanders

			# NOTE Step 1 drop unqualified rows

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy() # `bitsAlpha`
			bitsTarget &= state.locatorBits # `bitsAlpha`

			dataframeMeanders = dataframeMeanders.loc[(bitsTarget > 1)] # if bitsAlphaHasCurves

			del bitsTarget

			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= state.locatorBits # `bitsZulu`

			dataframeMeanders = dataframeMeanders.loc[(bitsTarget > 1)] # if bitsZuluHasCurves

			del bitsTarget

			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsZulu`
			bitsTarget &= 0b10 # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= 1 # (bitsZulu & 1)
			bitsTarget ^= 1 # (1 - (bitsZulu ...))
			dataframeMeanders.loc[:, 'analyzed'] = bitsTarget # selectorBitsZuluAtEven

			del bitsTarget

			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsAlpha`
			bitsTarget &= 1 # (bitsAlpha & 1)
			bitsTarget ^= 1 # (1 - (bitsAlpha ...))
			bitsTarget = bitsTarget.astype(bool) # selectorBitsAlphaAtODD

			dataframeMeanders = dataframeMeanders.loc[(bitsTarget) | (dataframeMeanders.loc[:, 'analyzed'])] # if (bitsAlphaIsEven or bitsZuluIsEven)

			del bitsTarget

			# NOTE Step 2 modify rows

			# Make a selector for bitsZuluAtEven, so you can modify bitsAlpha
			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsZulu`
			bitsTarget &= 0b10 # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= 1 # (bitsZulu & 1)
			bitsTarget ^= 1 # (1 - (bitsZulu ...))
			bitsTarget = bitsTarget.astype(bool) # selectorBitsZuluAtEven

			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'] # `bitsAlpha`
			dataframeMeanders.loc[:, 'analyzed'] &= state.locatorBits # `bitsAlpha`

			# if bitsAlphaIsEven and not bitsZuluIsEven, modify bitsAlphaPairedToOdd
			dataframeMeanders.loc[(~bitsTarget), 'analyzed'] = state.datatypeArcCode( # pyright: ignore[reportCallIssue, reportArgumentType]
				flipTheExtra_0b1AsUfunc(dataframeMeanders.loc[(~bitsTarget), 'analyzed']))

			del bitsTarget

			# if bitsZuluIsEven and not bitsAlphaIsEven, modify bitsZuluPairedToOdd
			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= state.locatorBits # `bitsZulu`

			bitsTarget.loc[(dataframeMeanders.loc[:, 'arcCode'] & 1).astype(bool)] = state.datatypeArcCode( # pyright: ignore[reportArgumentType, reportCallIssue]
				flipTheExtra_0b1AsUfunc(bitsTarget.loc[(dataframeMeanders.loc[:, 'arcCode'] & 1).astype(bool)])) # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]

			# NOTE Step 3 compute arcCode

			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 # (bitsAlpha >> 2)

			bitsTarget //= 2**2 # (bitsZulu >> 2)
			bitsTarget *= 2**1 # ((bitsZulu ...) << 1)

			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsZulu ...)

			del bitsTarget

			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

		def analyzeBitsAlpha() -> None:
			"""Compute `arcCode` from `bitsAlpha`.

			Formula
			-------
			```python
			if bitsAlpha > 1:
				arcCode = ((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) | (bitsAlpha >> 2)
			# `(1 - (bitsAlpha & 1)` is an evenness test.
			```
			"""
			nonlocal dataframeMeanders
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode']
			dataframeMeanders.loc[:, 'analyzed'] &= 1 # (bitsAlpha & 1)
			dataframeMeanders.loc[:, 'analyzed'] ^= 1 # (1 - (bitsAlpha ...))

			dataframeMeanders.loc[:, 'analyzed'] *= 2**1 # ((bitsAlpha ...) << 1)

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy() # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= state.locatorBits # `bitsZulu`

			bitsTarget *= 2**3 # (bitsZulu << 3)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsZulu ...)

			del bitsTarget

			"""NOTE In this code block, I rearranged the "formula" to use `bitsTarget` for two goals. 1. `(bitsAlpha >> 2)`.
			2. `if bitsAlpha > 1`. The trick is in the equivalence of v1 and v2.
				v1: BITScow | (BITSwalk >> 2)
				v2: ((BITScow << 2) | BITSwalk) >> 2

			The "formula" calls for v1, but by using v2, `bitsTarget` is not changed. Therefore, because `bitsTarget` is
			`bitsAlpha`, I can use `bitsTarget` for goal 2, `if bitsAlpha > 1`.
			"""
			dataframeMeanders.loc[:, 'analyzed'] *= 2**2 # ... | (bitsAlpha >> 2)

			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsAlpha`
			bitsTarget &= state.locatorBits # `bitsAlpha`

			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsAlpha)
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 # (... >> 2)

			dataframeMeanders.loc[(bitsTarget <= 1), 'analyzed'] = 0 # if bitsAlpha > 1

			del bitsTarget

			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

		def analyzeArcCodesSimple() -> None:
			"""Compute arcCode with the 'simple' formula.

			Formula
			-------
			```python
			arcCode = ((bitsAlpha | (bitsZulu << 1)) << 2) | 3
			```

			Notes
			-----
			Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the last bits are '0b00'. '0 + 3'
			is '0b11', and '0b00 | 0b11' is also '0b11'.

			"""
			nonlocal dataframeMeanders
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode']
			dataframeMeanders.loc[:, 'analyzed'] &= state.locatorBits

			bitsZulu: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsZulu //= 2**1 # (bitsZulu >> 1)
			bitsZulu &= state.locatorBits # `bitsZulu`
			bitsZulu *= 2**1 # (bitsZulu << 1)

			dataframeMeanders.loc[:, 'analyzed'] |= bitsZulu # ((bitsAlpha | (bitsZulu ...))

			del bitsZulu

			dataframeMeanders.loc[:, 'analyzed'] *= 2**2 # (... << 2)
			dataframeMeanders.loc[:, 'analyzed'] += 3 # (...) | 3
			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

		def analyzeBitsZulu() -> None:
			"""Compute `arcCode` from `bitsZulu`.

			Formula
			-------
			```python
			if bitsZulu > 1:
				arcCode = (1 - (bitsZulu & 1)) | (bitsAlpha << 2) | (bitsZulu >> 1)
			```
			"""
			nonlocal dataframeMeanders
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'] # `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] &= 0b10 # `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			dataframeMeanders.loc[:, 'analyzed'] &= 1 # (bitsZulu & 1)
			dataframeMeanders.loc[:, 'analyzed'] ^= 1 # (1 - (bitsZulu ...))

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy() # `bitsAlpha`
			bitsTarget &= state.locatorBits # `bitsAlpha`

			bitsTarget *= 2**2 # (bitsAlpha << 2)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsAlpha ...)

			del bitsTarget

			# NOTE No, IDK why I didn't use the same trick as in `analyzeBitsAlpha`. I _think_ I wrote this code before I figured out that trick.
			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= state.locatorBits # `bitsZulu`

			bitsTarget //= 2**1 # (bitsZulu >> 1)

			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsZulu ...)

			del bitsTarget

			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= state.locatorBits # `bitsZulu`

			dataframeMeanders.loc[bitsTarget <= 1, 'analyzed'] = 0 # if bitsZulu > 1

			del bitsTarget

			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

		def recordArcCodes() -> None:
			nonlocal dataframeAnalyzed

			indexStopAnalyzed: int = state.indexTarget + int((dataframeMeanders['analyzed'] > 0).sum()) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

			if indexStopAnalyzed > state.indexTarget:
				if len(dataframeAnalyzed.index) < indexStopAnalyzed:
					warn(f"Lengthened `dataframeAnalyzed` from {len(dataframeAnalyzed.index)} to {indexStopAnalyzed=}; n={state.n}, {state.kOfMatrix=}.", stacklevel=2)
					dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)

				dataframeAnalyzed.loc[state.indexTarget:indexStopAnalyzed - 1, ['analyzed', 'crossings']] = (
					dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0), ['analyzed', 'crossings']
								].to_numpy(dtype=state.datatypeArcCode, copy=False)
				)

				state.indexTarget = indexStopAnalyzed

			del indexStopAnalyzed

		dataframeMeanders = pandas.DataFrame({
			'arcCode': pandas.Series(name='arcCode', data=dataframeAnalyzed['analyzed'], copy=False, dtype=state.datatypeArcCode)
			, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=state.datatypeArcCode)
			, 'crossings': pandas.Series(name='crossings', data=dataframeAnalyzed['crossings'], copy=False, dtype=state.datatypeCrossings)
			} # pyright: ignore[reportUnknownArgumentType]
		)

		del dataframeAnalyzed
		goByeBye()

		state.bitWidth = int(dataframeMeanders['arcCode'].max()).bit_length()
		length: int = getBucketsTotal(state)
		dataframeAnalyzed = pandas.DataFrame({
			'analyzed': pandas.Series(0, pandas.RangeIndex(length), dtype=state.datatypeArcCode, name='analyzed')
			, 'crossings': pandas.Series(0, pandas.RangeIndex(length), dtype=state.datatypeCrossings, name='crossings')
			}, index=pandas.RangeIndex(length), columns=['analyzed', 'crossings'], dtype=state.datatypeArcCode # pyright: ignore[reportUnknownArgumentType]
		)

		state.kOfMatrix -= 1

		state.indexTarget = 0

		analyzeArcCodesSimple()
		recordArcCodes()

		analyzeBitsAlpha()
		recordArcCodes()

		analyzeBitsZulu()
		recordArcCodes()

		analyzeArcCodesAligned()
		recordArcCodes()
		del dataframeMeanders
		goByeBye()

		aggregateArcCodes()

		if state.n >= 45:  # for data collection
			print(state.n, state.kOfMatrix+1, state.indexTarget, sep=',')  # noqa: T201

	state.dictionaryMeanders = dataframeAnalyzed.set_index('analyzed')['crossings'].to_dict()
	return state

def doTheNeedful(state: MatrixMeandersNumPyState) -> int:
	"""Compute `crossings` with a transfer matrix algorithm implemented in pandas.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	crossings : int
		The computed value of `crossings`.

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

	return sum(state.dictionaryMeanders.values())
