"""Transfer matrix algorithm implementations in NumPy (*Num*erical *Py*thon) and pandas.

Citations
---------
- https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bib
- https://github.com/hunterhogan/mapFolding/blob/main/citations/Howroyd.bib

See Also
--------
`matrixMeanders`: transfer matrix algorithm implementation in pure Python with `int` (*int*eger) contained in a `dict` (*dict*ionary).
https://oeis.org/A000682
https://oeis.org/A005316
https://github.com/archmageirvine/joeis/blob/5dc2148344bff42182e2128a6c99df78044558c5/src/irvine/oeis/a005/A005316.java
"""
from __future__ import annotations

from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersShare import flipTheExtra_0b1AsUfunc, getBucketsTotal, integersWide吗
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from mapFolding.theTypes import 形ArcCode, 形Crossings
from typing import TYPE_CHECKING
from warnings import warn
import pandas

if TYPE_CHECKING:
	from mapFolding.dataBaskets import MatrixMeandersState

def count(state: MatrixMeandersState) -> MatrixMeandersState:
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
		'analyzed': pandas.Series(name='analyzed', data=state.dictionaryMeanders.keys(), copy=False, dtype=形ArcCode)
		, 'crossings': pandas.Series(name='crossings', data=state.dictionaryMeanders.values(), copy=False, dtype=形Crossings)
		}
	)
	state.dictionaryMeanders.clear()

	while 0 < state.boundary and not integersWide吗(state, dataframe=dataframeAnalyzed):

		def aggregateArcCodes()  -> None:
			nonlocal dataframeAnalyzed
			dataframeAnalyzed = dataframeAnalyzed.iloc[0:state.次Target].groupby('analyzed', sort=False)['crossings'].aggregate('sum').reset_index()

		def analyzeArcCodesAligned(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute `arcCode` from `bitsAlfa` and `bitsZulu` if at least one is an even number.

			Before computing `arcCode`, some values of `bitsAlfa` and `bitsZulu` are modified.

			Warning
			-------
			This function deletes rows from `dataframeMeanders`. Always run this analysis last.

			Formula
			-------
			```python
				if 1 < bitsAlfa and 1 < bitsZulu and (bitsAlfaIsEven or bitsZuluIsEven):
					arcCode = (bitsAlfa >> 2) | ((bitsZulu >> 2) << 1)
			```
			"""
			#--------- Step 1 drop unqualified rows ---------------------------
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders['analyzed'] &= state.bitsLocator       				# `bitsAlfa`

			dataframeMeanders['analyzed'] = dataframeMeanders['analyzed'].gt(1)		# `if bitsAlfaHasArcs`

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator											# `bitsZulu`

			dataframeMeanders['analyzed'] *= bitsTarget
			del bitsTarget
			dataframeMeanders = dataframeMeanders.loc[(1 < dataframeMeanders['analyzed'])]  # `if (bitsAlfaHasArcs and bitsZuluHasArcs)`  # ty: ignore[invalid-assignment]

			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator				# `bitsAlfa`

			dataframeMeanders.loc[:, 'analyzed'] &= 1								# One step of `bitsAlfaAtEven`.

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator											# `bitsZulu`

			dataframeMeanders.loc[:, 'analyzed'] &= bitsTarget						# One step of `bitsZuluAtEven`.
			del bitsTarget
			dataframeMeanders.loc[:, 'analyzed'] ^= 1								# Combined second step for `bitsAlfaAtEven` and `bitsZuluAtEven`.

			dataframeMeanders = dataframeMeanders.loc[(0 < dataframeMeanders['analyzed'])]  # `if (bitsAlfaIsEven or bitsZuluIsEven)`

			#-------- Step 2 modify rows --------------------------------------
			# Make a selector for bitsZuluAtOdd, so you can modify bitsAlfa
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1        					# Truncated conversion to `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] &= 1         						# `selectorBitsZuluAtOdd`

			bitsTarget = dataframeMeanders['arcCode'].copy()
			bitsTarget &= state.bitsLocator            								# `bitsAlfa`

			# `if bitsAlfaAtEven and not bitsZuluAtEven`, modify `bitsAlfaPairedToOdd`
			bitsTarget.loc[(0 < dataframeMeanders['analyzed'])] = 形ArcCode(
				flipTheExtra_0b1AsUfunc(bitsTarget.loc[(0 < dataframeMeanders['analyzed'])]))  # ty: ignore[invalid-assignment]

			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1
			dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator     			# `bitsZulu`

			# `if bitsZuluAtEven and not bitsAlfaAtEven`, modify `bitsZuluPairedToOdd`
			dataframeMeanders.loc[(0 < (dataframeMeanders.loc[:, 'arcCode'] & 1)), 'analyzed'] = 形ArcCode(
				flipTheExtra_0b1AsUfunc(dataframeMeanders.loc[(0 < (dataframeMeanders.loc[:, 'arcCode'] & 1)), 'analyzed']))

			#--------- Step 3 compute `arcCode` -------------------------------
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 							# (bitsZulu >> 2)
			dataframeMeanders.loc[:, 'analyzed'] *= 2**3 							# (... << 3)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget						# (... | bitsAlfa)
			del bitsTarget
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 							# ... >> 2

			dataframeMeanders.loc[state.MAXIMUMarcCode <= dataframeMeanders['analyzed'], 'analyzed'] = 0

			return dataframeMeanders

		def analyzeArcCodesSimple(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute arcCode with the 'simple' formula.

			Formula
			-------
			```python
				arcCode = ((bitsAlfa | (bitsZulu << 1)) << 2) | 3
			```

			Notes
			-----
			Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the
			last bits are '0b00'. '0 + 3' is '0b11', and '0b00 | 0b11' is also '0b11'.
			"""
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode']
			dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator

			bitsZulu: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsZulu //= 2**1
			bitsZulu &= state.bitsLocator 									# `bitsZulu`

			bitsZulu *= 2**1 												# (bitsZulu << 1)

			dataframeMeanders.loc[:, 'analyzed'] |= bitsZulu 				# ((bitsAlfa | (bitsZulu ...))

			del bitsZulu

			dataframeMeanders.loc[:, 'analyzed'] *= 2**2 					# (... << 2)
			dataframeMeanders.loc[:, 'analyzed'] += 3 						# (...) | 3
			dataframeMeanders.loc[state.MAXIMUMarcCode <= dataframeMeanders['analyzed'], 'analyzed'] = 0

			return dataframeMeanders

		def analyzeBitsAlfa(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute `arcCode` from `bitsAlfa`.

			Formula
			-------
			```python
				if 1 < bitsAlfa:
					arcCode = ((1 - (bitsAlfa & 1)) << 1) | (bitsZulu << 3) | (bitsAlfa >> 2)
				# `(1 - (bitsAlfa & 1)` is an evenness test.
			```
			"""
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode']					# Truncated creation of `bitsAlfa`
			dataframeMeanders.loc[:, 'analyzed'] &= 1 										# (bitsAlfa & 1)
			dataframeMeanders.loc[:, 'analyzed'] = 1 - dataframeMeanders.loc[:, 'analyzed']  # (1 - (bitsAlfa ...))

			dataframeMeanders.loc[:, 'analyzed'] *= 2**1 									# ((bitsAlfa ...) << 1)

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator 												# `bitsZulu`

			bitsTarget *= 2**3																# (bitsZulu << 3)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsZulu ...)

			del bitsTarget
			"""NOTE In this code block, I rearranged the "formula" to use `bitsTarget` for two goals.
			1. `(bitsAlfa >> 2)`.
			2. `if 1 < bitsAlfa`. The trick is in the equivalence of v1 and v2.

			v1: BITScow | (BITSwalk >> 2)
			v2: ((BITScow << 2) | BITSwalk) >> 2

			The "formula" calls for v1, but by using v2, `bitsTarget` is not changed. Therefore, because `bitsTarget` is
			`bitsAlfa`, I can use `bitsTarget` for goal 2, `if 1 < bitsAlfa`.
			"""
			dataframeMeanders.loc[:, 'analyzed'] *= 2**2									# ... | (bitsAlfa >> 2)

			bitsTarget = dataframeMeanders['arcCode'].copy()
			bitsTarget &= state.bitsLocator 												# `bitsAlfa`

			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsAlfa)
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 									# (... >> 2)

			dataframeMeanders.loc[(bitsTarget <= 1), 'analyzed'] = 0 						# if 1 < bitsAlfa

			del bitsTarget

			dataframeMeanders.loc[state.MAXIMUMarcCode <= dataframeMeanders['analyzed'], 'analyzed'] = 0

			return dataframeMeanders

		def analyzeBitsZulu(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute `arcCode` from `bitsZulu`.

			Formula
			-------
			```python
				if 1 < bitsZulu:
					arcCode = (1 - (bitsZulu & 1)) | (bitsAlfa << 2) | (bitsZulu >> 1)
			```
			"""
			# `(1 - (bitsZulu & 1))` is an evenness test: we want a single bit as the answer.
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode']
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1
			dataframeMeanders.loc[:, 'analyzed'] &= 1 										# Truncated creation of `bitsZulu`.
			dataframeMeanders.loc[:, 'analyzed'] &= 1 										# (bitsZulu & 1)
			dataframeMeanders.loc[:, 'analyzed'] = 1 - dataframeMeanders.loc[:, 'analyzed']  # (1 - (bitsZulu ...))

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget &= state.bitsLocator 												# `bitsAlfa`

			bitsTarget *= 2**2 																# (bitsAlfa << 2)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsAlfa ...)
			del bitsTarget

			# Same trick as in `analyzeBitsAlfa`.
			dataframeMeanders.loc[:, 'analyzed'] *= 2**1 									# (... << 1)

			bitsTarget = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator 												# `bitsZulu`

			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsZulu)
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1 									# (... >> 1)

			dataframeMeanders.loc[bitsTarget <= 1, 'analyzed'] = 0 							# if 1 < bitsZulu
			del bitsTarget

			dataframeMeanders.loc[state.MAXIMUMarcCode <= dataframeMeanders['analyzed'], 'analyzed'] = 0

			return dataframeMeanders

		def recordArcCodes(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Abstraction makes it easier to do things such as write to disk."""
			nonlocal dataframeAnalyzed

			次StopAnalyzed: int = state.次Target + int((0 < dataframeMeanders['analyzed']).sum())

			if state.次Target < 次StopAnalyzed:
				if len(dataframeAnalyzed.index) < 次StopAnalyzed:
					warn(f"Lengthened `dataframeAnalyzed` from {len(dataframeAnalyzed.index)} to {次StopAnalyzed=}; n={state.n}, {state.boundary=}.", stacklevel=2)
					dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(次StopAnalyzed), fill_value=0)

				dataframeAnalyzed.loc[state.次Target:次StopAnalyzed - 1, ['analyzed']] = (
					dataframeMeanders.loc[(0 < dataframeMeanders['analyzed']), ['analyzed']
								].to_numpy(dtype=形ArcCode, copy=False)
				)

				dataframeAnalyzed.loc[state.次Target:次StopAnalyzed - 1, ['crossings']] = (
					dataframeMeanders.loc[(0 < dataframeMeanders['analyzed']), ['crossings']
								].to_numpy(dtype=形Crossings, copy=False)
				)

				state.次Target = 次StopAnalyzed

			del 次StopAnalyzed

			return dataframeMeanders

		dataframeMeanders = pandas.DataFrame({
			'arcCode': pandas.Series(name='arcCode', data=dataframeAnalyzed['analyzed'], copy=False, dtype=形ArcCode)
			, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=形ArcCode)
			, 'crossings': pandas.Series(name='crossings', data=dataframeAnalyzed['crossings'], copy=False, dtype=形Crossings)
			}
		)

		del dataframeAnalyzed
		goByeBye()

		state.bitWidth = int(dataframeMeanders['arcCode'].max()).bit_length()
		state.setBitsLocator()
		length: int = getBucketsTotal(state)
		dataframeAnalyzed = pandas.DataFrame({
			'analyzed': pandas.Series(name='analyzed', data=0, index=pandas.RangeIndex(length), dtype=形ArcCode)
			, 'crossings': pandas.Series(name='crossings', data=0, index=pandas.RangeIndex(length), dtype=形Crossings)
			}, index=pandas.RangeIndex(length)
		)

		state.boundary -= 1
		state.setMAXIMUMarcCode()

		state.次Target = 0

		dataframeMeanders: pandas.DataFrame = analyzeArcCodesSimple(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)

		dataframeMeanders = analyzeBitsAlfa(dataframeMeanders)
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

def doTheNeedful(state: MatrixMeandersState) -> int:
	"""Compute `crossings` with a transfer matrix algorithm implemented in pandas.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	crossings : int
		The computed value of `crossings`.
	"""
	while 0 < state.boundary:
		if integersWide吗(state):
			state = countBigInt(state)
		else:
			state = count(state)
	return sum(state.dictionaryMeanders.values())
