from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from warnings import warn
import pandas

def countPandas(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing current `boundary`, `dictionaryMeanders`, and thresholds.

	Returns
	-------
	state : MatrixMeandersState
		Updated state with new `boundary` and `dictionaryMeanders`.
	"""
	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=state.dictionaryMeanders.keys(), copy=False, dtype=state.datatypeArcCode)
		, 'crossings': pandas.Series(name='crossings', data=state.dictionaryMeanders.values(), copy=False, dtype=state.datatypeCrossings)
		}
	)
	state.dictionaryMeanders.clear()

	while (state.boundary > 0 and not areIntegersWide(state, dataframe=dataframeAnalyzed)):

		def aggregateArcCodes()  -> None:
			nonlocal dataframeAnalyzed
			dataframeAnalyzed = dataframeAnalyzed.iloc[0:state.indexTarget].groupby('analyzed', sort=False)['crossings'].aggregate('sum').reset_index()

		def analyzeArcCodesAligned(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
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
			# NOTE Step 1 drop unqualified rows
# ======= > * > bitsAlpha 1 bitsZulu 1 ====================
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode'].copy()			# `bitsAlpha`
			dataframeMeanders['analyzed'] &= state.locatorBits							# `bitsAlpha`

			dataframeMeanders['analyzed'] = dataframeMeanders['analyzed'].gt(1)			# if bitsAlphaHasArcs

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()				# `bitsZulu`
			bitsTarget //= 2**1 														# `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= state.locatorBits 											# `bitsZulu`

			dataframeMeanders['analyzed'] *= bitsTarget
			del bitsTarget
			dataframeMeanders = dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 1)] 	# if (bitsAlphaHasArcs and bitsZuluHasArcs)

# ======= ^ & & bitsAlpha 1 bitsZulu 1 ====================
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()	# `bitsAlpha`
			dataframeMeanders.loc[:, 'analyzed'] &= state.locatorBits					# `bitsAlpha`

			dataframeMeanders.loc[:, 'analyzed'] &= 1									# `bitsAlpha`

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()				# `bitsZulu`
			bitsTarget //= 2**1															# `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= state.locatorBits 											# `bitsZulu`

			dataframeMeanders.loc[:, 'analyzed'] &= bitsTarget
			del bitsTarget
			dataframeMeanders.loc[:, 'analyzed'] ^= 1

			dataframeMeanders = dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0)] 	# if (bitsAlphaIsEven or bitsZuluIsEven)

			# NOTE Step 2 modify rows
			# Make a selector for bitsZuluAtOdd, so you can modify bitsAlpha
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()	# `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1								# `bitsZulu` (bitsZulu >> 1)
			dataframeMeanders.loc[:, 'analyzed'] &= 1									# selectorBitsZuluAtOdd

			bitsTarget = dataframeMeanders['arcCode'].copy()							# `bitsAlpha`
			bitsTarget &= state.locatorBits												# `bitsAlpha`

			# if bitsAlphaAtEven and not bitsZuluAtEven, modify bitsAlphaPairedToOdd
			bitsTarget.loc[(dataframeMeanders['analyzed'] > 0)] = state.datatypeArcCode(
				flipTheExtra_0b1AsUfunc(bitsTarget.loc[(dataframeMeanders['analyzed'] > 0)]))

			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()	# `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1								# `bitsZulu` (bitsZulu >> 1)
			dataframeMeanders.loc[:, 'analyzed'] &= state.locatorBits					# `bitsZulu`

			# if bitsZuluAtEven and not bitsAlphaAtEven, modify bitsZuluPairedToOdd
			dataframeMeanders.loc[((dataframeMeanders.loc[:, 'arcCode'] & 1) > 0), 'analyzed'] = state.datatypeArcCode(
				flipTheExtra_0b1AsUfunc(dataframeMeanders.loc[((dataframeMeanders.loc[:, 'arcCode'] & 1) > 0), 'analyzed']))

			# NOTE Step 3 compute arcCode
# ======= >> | << >> bitsZulu 2 3 bitsAlpha 2 =============
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 # (bitsZulu >> 2)
			dataframeMeanders.loc[:, 'analyzed'] *= 2**3 # (bitsZulu << 3)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget
			del bitsTarget
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 # (... >> 2)

			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

			return dataframeMeanders

		def analyzeArcCodesSimple(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
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

			return dataframeMeanders

		def analyzeBitsAlpha(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute `arcCode` from `bitsAlpha`.

			Formula
			-------
			```python
			if bitsAlpha > 1:
				arcCode = ((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) | (bitsAlpha >> 2)
			# `(1 - (bitsAlpha & 1)` is an evenness test.
			```
			"""
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

			return dataframeMeanders

		def analyzeBitsZulu(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute `arcCode` from `bitsZulu`.

			Formula
			-------
			```python
			if bitsZulu > 1:
				arcCode = (1 - (bitsZulu & 1)) | (bitsAlpha << 2) | (bitsZulu >> 1)
			```
			"""
# NOTE `(1 - (bitsZulu & 1))` is an evenness test: we want a single bit as the answer.
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'] # `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			dataframeMeanders.loc[:, 'analyzed'] &= 1 # `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] &= 1 # (bitsZulu & 1)
			dataframeMeanders.loc[:, 'analyzed'] ^= 1 # (1 - (bitsZulu ...))

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy() # `bitsAlpha`
			bitsTarget &= state.locatorBits # `bitsAlpha`

			bitsTarget *= 2**2 # (bitsAlpha << 2)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsAlpha ...)
			del bitsTarget

			# NOTE Same trick as in `analyzeBitsAlpha`.
			dataframeMeanders.loc[:, 'analyzed'] *= 2**1 # (... << 1)

			bitsTarget = dataframeMeanders['arcCode'].copy() # `bitsZulu`
			bitsTarget //= 2**1 # `bitsZulu` (bitsZulu >> 1)
			bitsTarget &= state.locatorBits # `bitsZulu`

			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget # ... | (bitsZulu)
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1 # (... >> 1)

			dataframeMeanders.loc[bitsTarget <= 1, 'analyzed'] = 0 # if bitsZulu > 1
			del bitsTarget

			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

			return dataframeMeanders

		def recordArcCodes(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			nonlocal dataframeAnalyzed

			indexStopAnalyzed: int = state.indexTarget + int((dataframeMeanders['analyzed'] > 0).sum())

			if indexStopAnalyzed > state.indexTarget:
				if len(dataframeAnalyzed.index) < indexStopAnalyzed:
					warn(f"Lengthened `dataframeAnalyzed` from {len(dataframeAnalyzed.index)} to {indexStopAnalyzed=}; n={state.n}, {state.boundary=}.", stacklevel=2)
					dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)

				dataframeAnalyzed.loc[state.indexTarget:indexStopAnalyzed - 1, ['analyzed']] = (
					dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0), ['analyzed']
								].to_numpy(dtype=state.datatypeArcCode, copy=False)
				)

				dataframeAnalyzed.loc[state.indexTarget:indexStopAnalyzed - 1, ['crossings']] = (
					dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0), ['crossings']
								].to_numpy(dtype=state.datatypeCrossings, copy=False)
				)

				state.indexTarget = indexStopAnalyzed

			del indexStopAnalyzed

			return dataframeMeanders

		dataframeMeanders = pandas.DataFrame({
			'arcCode': pandas.Series(name='arcCode', data=dataframeAnalyzed['analyzed'], copy=False, dtype=state.datatypeArcCode)
			, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=state.datatypeArcCode)
			, 'crossings': pandas.Series(name='crossings', data=dataframeAnalyzed['crossings'], copy=False, dtype=state.datatypeCrossings)
			}
		)

		del dataframeAnalyzed
		goByeBye()

		state.bitWidth = int(dataframeMeanders['arcCode'].max()).bit_length()
		length: int = getBucketsTotal(state)
		dataframeAnalyzed = pandas.DataFrame({
			'analyzed': pandas.Series(name='analyzed', data=0, index=pandas.RangeIndex(length), dtype=state.datatypeArcCode)
			, 'crossings': pandas.Series(name='crossings', data=0, index=pandas.RangeIndex(length), dtype=state.datatypeCrossings)
			}, index=pandas.RangeIndex(length)
		)

		state.boundary -= 1

		state.indexTarget = 0

		dataframeMeanders: pandas.DataFrame = analyzeArcCodesSimple(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)

		dataframeMeanders = analyzeBitsAlpha(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)

		dataframeMeanders = analyzeBitsZulu(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)

		dataframeMeanders = analyzeArcCodesAligned(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)
		del dataframeMeanders
		goByeBye()

		aggregateArcCodes()

	state.dictionaryMeanders = dataframeAnalyzed.set_index('analyzed')['crossings'].to_dict()
	del dataframeAnalyzed
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
	while state.boundary > 0:
		if areIntegersWide(state):
			state = countBigInt(state)
		else:
			state = countPandas(state)
	return sum(state.dictionaryMeanders.values())
