"""Count meanders with matrix transfer algorithm using pandas DataFrame."""
from functools import cache
from tqdm import tqdm
import numpy
import pandas

datatypeCurveLocationsHARDCODED = numpy.uint64
datatypeDistinctCrossingsHARDCODED = numpy.uint64
datatypeCurveLocations = datatypeCurveLocationsHARDCODED
datatypeDistinctCrossings = datatypeDistinctCrossingsHARDCODED
# idk why this works for values too large for uint64
CategoriesAlignAt = pandas.CategoricalDtype(categories=['evenAlpha', 'evenZulu', 'evenBoth', 'oddBoth'], ordered=False)

@cache
def _walkDyckPath(intWithExtra_0b1: int) -> int:
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

@cache
def _flipTheExtra_0b1(intWithExtra_0b1: int) -> int:
	return intWithExtra_0b1 ^ _walkDyckPath(intWithExtra_0b1)

flipTheExtra_0b1 = numpy.vectorize(_flipTheExtra_0b1, otypes=[datatypeCurveLocations])

def count(bridges: int, dictionaryCurveLocations: dict[int, int]) -> int:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame."""
	def aggregateCurveLocations(MAXIMUMcurveLocations: int) -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = 0
		dataframeAnalyzed = pandas.concat([dataframeAnalyzed
									, dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0)].groupby('analyzed')['distinctCrossings'].aggregate('sum').reset_index()
						], ignore_index=True)
		dataframeAnalyzed = dataframeAnalyzed.groupby('analyzed')['distinctCrossings'].aggregate('sum').reset_index()

		dataframeCurveLocations.loc[:, 'analyzed'] = 0

	def analyzeCurveLocationsAligned(MAXIMUMcurveLocations: int) -> None:
		"""Compute `curveLocations` from `groupAlpha` and `groupZulu` if at least one is an even number.

		Before computing `curveLocations`, some values of `groupAlpha` and `groupZulu` are modified.

		Formula
		-------
		```python
		if groupAlpha > 1 and groupZulu > 1 and (groupAlphaIsEven or groupZuluIsEven):
			curveLocations = (groupAlpha >> 2) | ((groupZulu >> 2) << 1)
		```
		"""
		nonlocal dataframeCurveLocations

		# if groupAlphaIsEven or groupZuluIsEven
		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations[dataframeCurveLocations['alignAt'] == 'oddBoth'].index)

		# if groupAlphaCurves and groupZuluCurves
		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations[(dataframeCurveLocations['groupAlpha'] <= 1) | (dataframeCurveLocations['groupZulu'] <= 1)].index)

		# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
		dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenAlpha', 'groupAlpha'] = flipTheExtra_0b1(dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenAlpha', 'groupAlpha'])

		# if groupZuluIsEven and not groupAlphaIsEven, modifyGroupZuluPairedToOdd
		dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenZulu', 'groupZulu'] = flipTheExtra_0b1(dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenZulu', 'groupZulu'])

		dataframeCurveLocations.loc[:, 'groupAlpha'] //= 2**2 # (groupAlpha >> 2)
		dataframeCurveLocations.loc[:, 'groupZulu'] //= 2**2 # (groupZulu >> 2)
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**1 # ((groupZulu ...) << 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupAlpha'] | dataframeCurveLocations['groupZulu'] # (groupZulu ...) | (groupAlpha ...)

		aggregateCurveLocations(MAXIMUMcurveLocations)

	def analyzeCurveLocationsAlpha(MAXIMUMcurveLocations: int) -> None:
		"""Compute `curveLocations` from `groupAlpha`.

		Formula
		-------
		```python
		if groupAlpha > 1:
			curveLocations = ((1 - (groupAlpha & 1)) << 1) | (groupZulu << 3) | (groupAlpha >> 2)
		```
		"""
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupAlpha'] & 1 # (groupAlpha & 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupAlpha ...))

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] == 1, 'alignAt'] = 'evenAlpha' # groupAlphaIsEven

		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**1 # ((groupAlpha ...) << 1)
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**3 # (groupZulu << 3)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # ... | (groupZulu ...)
		dataframeCurveLocations.loc[:, 'analyzed'] |= (dataframeCurveLocations['groupAlpha'] // 2**2) # ... | (groupAlpha >> 2)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupAlpha'] <= 1, 'analyzed'] = 0 # if groupAlpha > 1

		aggregateCurveLocations(MAXIMUMcurveLocations)
		computeCurveGroups(alpha=False)

	def analyzeCurveLocationsSimple(MAXIMUMcurveLocations: int) -> None:
		"""Compute curveLocations with the 'simple' bridges formula.

		((groupAlpha | (groupZulu << 1)) << 2) | 3
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**1 # (groupZulu << 1)
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['groupAlpha'] | dataframeCurveLocations['groupZulu'] # ((groupAlpha | (groupZulu ...))
		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # (... << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] += 3 # (...) | 3
		"""NOTE about substituting `+= 3` for `|= 3`: (I hope my substitution is valid!)
		Given "n" is a Python `int` >= 0, "0bk" = `bin(n)`, I claim:
		n * 2**2 == n << 2
		bin(n * 2**2) == 0bk00
		0b11 = 0b00 | 0b11
		0bk11 = 0bk00 | 0b11
		0b11 = 0bk11 - 0bk00
		0b11 == int(3)

		therefore, for any non-zero integer, 0bk00, the operation 0bk00 | 0b11 is equivalent to 0bk00 + 0b11.

		Why substitute? I've been having problems implementing bitwise operations in pandas, so I am avoiding them until I learn
		how to implement them in pandas.
		"""

		aggregateCurveLocations(MAXIMUMcurveLocations)

		computeCurveGroups(alpha=False)

	def analyzeCurveLocationsZulu(MAXIMUMcurveLocations: int) -> None:
		"""Compute `curveLocations` from `groupZulu`.

		Formula
		-------
		```python
		if groupZulu > 1:
			curveLocations = (1 - (groupZulu & 1)) | (groupAlpha << 2) | (groupZulu >> 1)
		```
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupZulu'] & 1 # (groupZulu & 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupZulu ...))

		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1) & (dataframeCurveLocations['alignAt'] == 'evenAlpha'), 'alignAt'] = 'evenBoth' # groupZuluIsEven
		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1) & (dataframeCurveLocations['alignAt'] == 'oddBoth'), 'alignAt'] = 'evenZulu' # groupZuluIsEven

		dataframeCurveLocations.loc[:, 'groupAlpha'] *= 2**2 # (groupAlpha << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupAlpha'] # ... | (groupAlpha ...)
		dataframeCurveLocations.loc[:, 'analyzed'] |= (dataframeCurveLocations['groupZulu'] // 2**1) # ... | (groupZulu >> 1)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupZulu'] <= 1, 'analyzed'] = 0 # if groupZulu > 1

		aggregateCurveLocations(MAXIMUMcurveLocations)
		computeCurveGroups(zulu=False)

	def computeCurveGroups(*, alpha: bool = True, zulu: bool = True) -> None:
		"""Compute `groupAlpha` and `groupZulu` with 'bit-masks' on `curveLocations`.

		3L33T H@X0R
		-----------
		- `locatorGroupAlpha`: odd-parity bit-mask
		- `groupAlpha`: odd-parity bit-masked `curveLocations`
		- `locatorGroupZulu`: even-parity bit-mask
		- `groupZulu`: even-parity bit-masked `curveLocations`

		Notes
		-----
		In binary, `locatorGroupAlpha` has alternating 0s and 1s and ends with a 1, such as '101', '0101', and '10101'. The last
		digit is in the 1's column, but programmers usually call it the "least significant bit" (LSB). If we count the columns
		from the right, the 1's column is column 1, the 2's column is column 2, the 4's column is column 3, and so on. When
		counting this way, `locatorGroupAlpha` has 1s in the columns with odd index numbers. Mathematicians and programmers,
		therefore, tend to call `locatorGroupAlpha` something like the "odd bit-mask", the "odd-parity numbers", or simply "odd
		mask" or "odd numbers". In addition to "odd" being inherently ambiguous in this context, this algorithm also segregates
		odd numbers from even numbers, so I avoid using "odd" and "even" in the names of these bit-masks.
		"""
		nonlocal dataframeCurveLocations
		bitWidth: int = int(dataframeCurveLocations['curveLocations'].max()).bit_length()
# TODO scratch this itch: I _feel_ it must be possible to bifurcate `curvesLocations` with one formula. Even if implementation is infeasible, I want to know.
		if alpha:
			locatorGroupAlpha = sum(1 << one for one in range(0, bitWidth, 2))
			dataframeCurveLocations['groupAlpha'] = dataframeCurveLocations['curveLocations'] & locatorGroupAlpha
		if zulu:
			locatorGroupZulu = sum(1 << one for one in range(1, bitWidth, 2))
			dataframeCurveLocations['groupZulu'] = dataframeCurveLocations['curveLocations'] & locatorGroupZulu
			dataframeCurveLocations.loc[:, 'groupZulu'] //= 2**1 # (groupZulu >> 1)

	def outfitDataframeCurveLocations() -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations = dataframeCurveLocations.iloc[0:0]
		dataframeCurveLocations['curveLocations'] = dataframeAnalyzed['analyzed']
		dataframeCurveLocations['distinctCrossings'] = dataframeAnalyzed['distinctCrossings']
		dataframeCurveLocations['alignAt'] = 'oddBoth'
		dataframeCurveLocations['analyzed'] = 0
		dataframeAnalyzed = dataframeAnalyzed.iloc[0:0]
		computeCurveGroups()

	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=list(dictionaryCurveLocations.keys()), dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=list(dictionaryCurveLocations.values()), dtype=datatypeDistinctCrossings)
		}
	)

	dataframeCurveLocations = pandas.DataFrame({
		'curveLocations': pandas.Series(name='curveLocations', data=0, dtype=datatypeCurveLocations)
		, 'groupAlpha': pandas.Series(name='groupAlpha', data=0, dtype=datatypeCurveLocations)
		, 'groupZulu': pandas.Series(name='groupZulu', data=0, dtype=datatypeCurveLocations)
		, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=0, dtype=datatypeDistinctCrossings)
		, 'alignAt': pandas.Series(name='alignAt', data='oddBoth', dtype=CategoriesAlignAt)
		}
	)

	for countdown in tqdm(range((bridges - 1), (0 - 1), -1)):
		MAXIMUMcurveLocations: int = 1 << (2 * countdown + 4)
		outfitDataframeCurveLocations()

		analyzeCurveLocationsSimple(MAXIMUMcurveLocations)
		analyzeCurveLocationsAlpha(MAXIMUMcurveLocations)
		analyzeCurveLocationsZulu(MAXIMUMcurveLocations)
		analyzeCurveLocationsAligned(MAXIMUMcurveLocations)

	return int(dataframeAnalyzed['distinctCrossings'].sum())

def doTheNeedful(bridges: int, dictionaryCurveLocations: dict[int, int]) -> int:
	"""Do the needful."""
	return count(bridges, dictionaryCurveLocations)

def A000682getCurveLocations(n: int) -> dict[int, int]:
	"""A000682."""
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)
	curveStart: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveStart << 1) | curveStart]
	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveStart = (curveStart << 4) | 0b101
		listCurveLocations.append((curveStart << 1) | curveStart)
	return dict.fromkeys(listCurveLocations, 1)

@cache
def A000682(n: int) -> int:
	"""A000682."""
	return doTheNeedful(n - 1, A000682getCurveLocations(n - 1))

def A005316getCurveLocations(n: int) -> dict[int, int]:
	"""A005316."""
	if n & 0b1:
		return {22: 1}
	else:
		return {15: 1}

@cache
def A005316(n: int) -> int:
	"""A005316."""
	return doTheNeedful(n - 1, A005316getCurveLocations(n - 1))
