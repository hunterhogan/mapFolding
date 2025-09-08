"""Count meanders with matrix transfer algorithm."""
from functools import cache
from gc import collect as goByeBye
from math import exp
from warnings import warn
import numpy
import pandas
import sys

# ----------------- environment configuration -------------------------------------------------------------------------
_bitWidthOfFixedSizeInteger: int = 64

_bitWidthOffsetCurveLocationsNecessary: int = 3 # `curveLocations` analysis may need 3 extra bits. For example, `groupZulu << 3`.
_bitWidthOffsetCurveLocationsSafety: int = 1 # I don't have mathematical proof of how many extra bits I need.
_bitWidthOffsetCurveLocations: int = _bitWidthOffsetCurveLocationsNecessary + _bitWidthOffsetCurveLocationsSafety

bitWidthCurveLocationsMaximum: int = _bitWidthOfFixedSizeInteger - _bitWidthOffsetCurveLocations

del _bitWidthOffsetCurveLocationsNecessary, _bitWidthOffsetCurveLocationsSafety, _bitWidthOffsetCurveLocations

_bitWidthOffsetDistinctCrossingsNecessary: int = 0 # I don't know of any.
_bitWidthOffsetDistinctCrossingsEstimation: int = 3 # See reference directory.
_bitWidthOffsetDistinctCrossingsSafety: int = 1
_bitWidthOffsetDistinctCrossings: int = _bitWidthOffsetDistinctCrossingsNecessary + _bitWidthOffsetDistinctCrossingsEstimation + _bitWidthOffsetDistinctCrossingsSafety

bitWidthDistinctCrossingsMaximum: int = _bitWidthOfFixedSizeInteger - _bitWidthOffsetDistinctCrossings

del _bitWidthOffsetDistinctCrossingsNecessary, _bitWidthOffsetDistinctCrossingsEstimation, _bitWidthOffsetDistinctCrossingsSafety, _bitWidthOffsetDistinctCrossings
del _bitWidthOfFixedSizeInteger

datatypeCurveLocations = datatypeDistinctCrossings = numpy.uint64

groupAlphaNoCurves = -10
groupZuluNoCurves = -20
groupsHaveCurvesNotEven = 0
groupAlphaAtEven = 1
groupZuluAtEven = 2

_n: int = 0

# ----------------- support functions ---------------------------------------------------------------------------------
@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def getEstimatedLengthCurveLocationsAnalyzed(indexTransferMatrix: int, safetyMultiplicand: float = 1.2, safetyAddend: int = 1000) -> int:
	"""Estimate the total number of non-unique curveLocations that will be computed from the existing curveLocations.

	Parameters
	----------
	curveLocationsDistinct : int
		The number of distinct `curveLocations` in the current iteration of the transfer matrix.
	indexTransferMatrix : int
		The current index in the transfer matrix algorithm.
	safetyMultiplicand : float = 1.05
		A multiplicative safety factor to ensure the estimate is slightly larger than necessary.
	safetyAddend : int = 100
		An additive safety factor to ensure the estimate is slightly larger than necessary.

	Returns
	-------
	estimatedLengthCurveLocationsAnalyzed : int
		An estimate of the total number of non-unique `curveLocations` that will be computed from the existing `curveLocations`.

	Notes
	-----
	`int` truncates, so `+ 1` to round up.

	In the 'reference' directory, I have a Jupyter notebook that derives this formula.
	"""
	return int(predict_less_than_max(indexTransferMatrix) * safetyMultiplicand) + safetyAddend + 1

def predict_less_than_max(indexTransferMatrix: int) -> float:
	"""Predict."""
	n = float(_n)
	b = max(0.0, min(float(indexTransferMatrix), n))
	x = b / n
	x1 = x
	x2 = x**2
	x3 = x**3
	x4 = x**4
	vals: list[float] = []
	vals.append(n)  # n
	vals.append(n**2)  # n2
	vals.append(x1)  # x1
	vals.append(x2)  # x2
	vals.append(x3)  # x3
	vals.append(x4)  # x4
	vals.append(n * x1)  # nx1
	vals.append(n * x2)  # nx2
	vals.append(n * x3)  # nx3
	z_hat = -24.817496909
	z_hat += (-0.242059151721) * vals[0]
	z_hat += (-2.663008305e-06) * vals[1]
	z_hat += (183.309622727) * vals[2]
	z_hat += (-522.471797116) * vals[3]
	z_hat += (679.612786304) * vals[4]
	z_hat += (-330.496162948) * vals[5]
	z_hat += (4.44033168706) * vals[6]
	z_hat += (-8.61595855772) * vals[7]
	z_hat += (4.75288013279) * vals[8]
	y_hat = exp(z_hat) - 1.0
	return max(y_hat, 0.0)

def getLocatorGroupAlpha(bitWidth: int) -> int:
	"""Compute an odd-parity bit-mask with `bitWidth` bits.

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
	return sum(1 << one for one in range(0, bitWidth, 2))

def getLocatorGroupZulu(bitWidth: int) -> int:
	"""Compute an even-parity bit-mask with `bitWidth` bits."""
	return sum(1 << one for one in range(1, bitWidth, 2))

def getMAXIMUMcurveLocations(indexTransferMatrix: int) -> int:
	"""Compute the maximum value of `curveLocations` for the current iteration of the transfer matrix."""
	return 1 << (2 * indexTransferMatrix + 4)

def outfitDictionaryCurveGroups(dictionaryCurveLocations: dict[int, int]) -> dict[tuple[int, int], int]:
	"""Outfit `dictionaryCurveGroups` so it may manage the computations for one iteration of the transfer matrix.

	`dictionaryCurveGroups` holds the input data, and `dictionaryCurveLocations` aggregates the output data as it is computed.

	Parameters
	----------
	dictionaryCurveLocations : dict[int, int]
		A dictionary of `curveLocations` to `distinctCrossings`.

	Returns
	-------
	dictionaryCurveGroups : dict[tuple[int, int], int]
		A dictionary of `(groupAlpha, groupZulu)` to `distinctCrossings`.
	"""
	bitWidth: int = max(dictionaryCurveLocations.keys()).bit_length()
	locatorGroupAlpha: int = getLocatorGroupAlpha(bitWidth)
	locatorGroupZulu: int = getLocatorGroupZulu(bitWidth)
	return {(curveLocations & locatorGroupAlpha, (curveLocations & locatorGroupZulu) >> 1): distinctCrossings
		for curveLocations, distinctCrossings in dictionaryCurveLocations.items()}

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

# ----------------- counting functions --------------------------------------------------------------------------------

def countBigInt(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int]) -> tuple[int, dict[int, int]]:
	"""Count meanders with matrix transfer algorithm using Python primitive `int` contained in a Python primitive `dict`.

	Parameters
	----------
	indexTransferMatrix : int
		The current index in the transfer matrix algorithm.
	dictionaryCurveLocations : dict[int, int]
		A dictionary of `curveLocations` to `distinctCrossings`.

	Returns
	-------
	matrixMeandersState : tuple[int, dict[int, int]]
		The state of the algorithm computation: the current `indexTransferMatrix`, `curveLocations`, and `distinctCrossings`.

	Notes
	-----
	The algorithm is sophisticated, but this implementation is straightforward. Compute each index one at a time, compute each
	`curveLocations` one at a time, and compute each type of analysis one at a time.
	"""
	dictionaryCurveGroups: dict[tuple[int, int], int] = {}

# TODO garbage collection balance maximizing `walkDyckPath` cache hits with `dictionaryCurveLocations` and `dictionaryCurveGroups`
# memory usage.

	while (indexTransferMatrix > 0
		and ((max(dictionaryCurveLocations.keys()).bit_length() > bitWidthCurveLocationsMaximum)
		or (max(dictionaryCurveLocations.values()).bit_length() > bitWidthDistinctCrossingsMaximum))):

		indexTransferMatrix -= 1

		MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(indexTransferMatrix)
		dictionaryCurveGroups = outfitDictionaryCurveGroups(dictionaryCurveLocations)
		dictionaryCurveLocations = {}
# TODO is `dictionaryCurveLocations.clear()` better for garbage collection?

		for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
			groupAlphaCurves: bool = groupAlpha > 1
			groupZuluCurves: bool = groupZulu > 1
			groupAlphaIsEven = groupZuluIsEven = 0

			# simple
			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			if curveLocationAnalysis < MAXIMUMcurveLocations:
				dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 1)) << 1)
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupZuluCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			# aligned
			if groupAlphaCurves and groupZuluCurves and (groupAlphaIsEven or groupZuluIsEven):
				if groupAlphaIsEven and not groupZuluIsEven:
					groupAlpha ^= walkDyckPath(groupAlpha)  # noqa: PLW2901
				elif groupZuluIsEven and not groupAlphaIsEven:
					groupZulu ^= walkDyckPath(groupZulu)  # noqa: PLW2901

				curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

	return (indexTransferMatrix, dictionaryCurveLocations)

def countPandas(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int]) -> tuple[int, dict[int, int]]:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame.

	indexTransferMatrix : int
		The current index in the transfer matrix algorithm.
	dictionaryCurveLocations : dict[int, int]
		A dictionary of `curveLocations` to `distinctCrossings`.

	Returns
	-------
	matrixMeandersState : tuple[int, dict[int, int]]
		The state of the algorithm computation: the current `indexTransferMatrix`, `curveLocations`, and `distinctCrossings`.
	"""
	def aggregateCurveLocations(MAXIMUMcurveLocations: int) -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations, indexStartAnalyzed

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = 0

		indexStopAnalyzed: int = indexStartAnalyzed + int((dataframeCurveLocations['analyzed'] > 0).sum())

		if indexStopAnalyzed > indexStartAnalyzed:
			currentLengthAnalyzed: int = len(dataframeAnalyzed.index)
			if currentLengthAnalyzed < indexStopAnalyzed:
				dataframeAnalyzed = dataframeAnalyzed.reindex(range(indexStopAnalyzed), fill_value=0)
				warn(
					f"matrixMeanders.countPandas: expanded dataframeAnalyzed to length {indexStopAnalyzed} to avoid overflow; n={_n}, indexTransferMatrix={indexTransferMatrix}, indexStopAnalyzed={indexStopAnalyzed}",
					stacklevel=2
				)

			dataframeAnalyzed.loc[indexStartAnalyzed:indexStopAnalyzed - 1, ['analyzed', 'distinctCrossings']] = (
				dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0), ['analyzed', 'distinctCrossings']].to_numpy(dtype=datatypeCurveLocations, copy=False)
			)

			indexStartAnalyzed = indexStopAnalyzed

		dataframeCurveLocations.loc[:, 'analyzed'] = 0

	def analyzeCurveLocationsAligned(MAXIMUMcurveLocations: int) -> None:
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

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.

		Notes
		-----
		"IsEven" vs "AtEven":
		- `groupAlphaIsEven` ≅ `groupAlphaAtEven`
		- `groupZuluIsEven` ≅ `groupZuluAtEven`
		- Semantically, "IsEven" is the evaluation of a single value, while "AtEven" is the evaluation of multiple values telling
		us where the even numbers are *at*, hence "AtEven".
		- Pragmatically, I need to avoid a name collision between global `groupAlphaIsEven` and local `groupAlphaIsEven` in
		`countBigInt`.

		The 'dropModify' column in `dataframeCurveLocations` controls what is dropped and what is modified. If a row fails `if
		groupAlpha > 1 and groupZulu > 1`, its 'dropModify' value will be between -20 and -8. If a row passes that check but fails
		`(groupAlphaIsEven or groupZuluIsEven)`, meaning both values are odd numbers, its 'dropModify' value will be 0. Rows with
		`['dropModify'] <= 0]` get dropped.

		If `groupAlphaIsEven` then 'dropModify' += 1. If `groupZuluIsEven` then 'dropModify' += 2. So the remaining rows:
		- `['dropModify'] == 1`: `groupAlphaIsEven` and not `groupZuluIsEven`
		- `['dropModify'] == 2`: `groupZuluIsEven` and not `groupAlphaIsEven`
		- `['dropModify'] == 3`: `groupAlphaIsEven` and `groupZuluIsEven`
		"""
		nonlocal dataframeCurveLocations

		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations.loc[dataframeCurveLocations['dropModify'] <= groupsHaveCurvesNotEven].index) # if groupAlphaCurves and groupZuluCurves and (groupAlphaIsEven or groupZuluIsEven)

		# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
		dataframeCurveLocations.loc[dataframeCurveLocations['dropModify'] == groupAlphaAtEven, 'groupAlpha'] = datatypeCurveLocations(flipTheExtra_0b1AsUfunc(dataframeCurveLocations.loc[dataframeCurveLocations['dropModify'] == groupAlphaAtEven, 'groupAlpha']))

		# if groupZuluIsEven and not groupAlphaIsEven, modifyGroupZuluPairedToOdd
		dataframeCurveLocations.loc[dataframeCurveLocations['dropModify'] == groupZuluAtEven, 'groupZulu'] = datatypeCurveLocations(flipTheExtra_0b1AsUfunc(dataframeCurveLocations.loc[dataframeCurveLocations['dropModify'] == groupZuluAtEven, 'groupZulu']))

		dataframeCurveLocations.loc[:, 'groupAlpha'] //= 2**2 # (groupAlpha >> 2)
		dataframeCurveLocations.loc[:, 'groupZulu'] //= 2**2 # (groupZulu >> 2)
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**1 # ((groupZulu ...) << 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupAlpha']
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # (groupZulu ...) | (groupAlpha ...)

		aggregateCurveLocations(MAXIMUMcurveLocations)

	def analyzeCurveLocationsAlpha(MAXIMUMcurveLocations: int) -> None:
		"""Compute `curveLocations` from `groupAlpha`.

		Formula
		-------
		```python
		if groupAlpha > 1:
			curveLocations = ((1 - (groupAlpha & 1)) << 1) | (groupZulu << 3) | (groupAlpha >> 2)
		```

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupAlpha']
		dataframeCurveLocations.loc[:, 'analyzed'] &= 1 # (groupAlpha & 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupAlpha ...))

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] == 1, 'dropModify'] += groupAlphaAtEven # groupAlphaIsEven

		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**1 # ((groupAlpha ...) << 1)
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**3 # (groupZulu << 3)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # ... | (groupZulu ...)
		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # ... | (groupAlpha >> 2)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupAlpha'] # ... | (groupAlpha)
		dataframeCurveLocations.loc[:, 'analyzed'] //= 2**2 # (... >> 2)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupAlpha'] <= 1, ['analyzed', 'dropModify']] = [0, groupAlphaNoCurves] # if groupAlpha > 1

		aggregateCurveLocations(MAXIMUMcurveLocations)
		computeCurveGroups(alpha=False)

	def analyzeCurveLocationsSimple(MAXIMUMcurveLocations: int) -> None:
		"""Compute curveLocations with the 'simple' bridges formula.

		Formula
		-------
		```python
		curveLocations = ((groupAlpha | (groupZulu << 1)) << 2) | 3
		```

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.

		Notes
		-----
		Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the last bits are '0b00'. '0 + 3'
		is '0b11', and '0b00 | 0b11' is also '0b11'.

		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['groupAlpha']
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**1 # (groupZulu << 1)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # ((groupAlpha | (groupZulu ...))
		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # (... << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] += 3 # (...) | 3

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

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupZulu']
		dataframeCurveLocations.loc[:, 'analyzed'] &= 1 # (groupZulu & 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupZulu ...))

		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1), 'dropModify'] += groupZuluAtEven # groupZuluIsEven

		dataframeCurveLocations.loc[:, 'groupAlpha'] *= 2**2 # (groupAlpha << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupAlpha'] # ... | (groupAlpha ...)
		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**1 # ... | (groupZulu >> 1)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # ... | (groupZulu)
		dataframeCurveLocations.loc[:, 'analyzed'] //= 2**1 # (... >> 1)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupZulu'] <= 1, ['analyzed', 'dropModify']] = [0, groupZuluNoCurves] # if groupZulu > 1

		aggregateCurveLocations(MAXIMUMcurveLocations)
		computeCurveGroups(zulu=False)

	def computeCurveGroups(*, alpha: bool = True, zulu: bool = True) -> None:
		"""Compute `groupAlpha` and `groupZulu` with 'bit-masks' on `curveLocations`.

		Parameters
		----------
		alpha : bool = True
			Should column `groupAlpha` be computed?

		zulu : bool = True
			Should column `groupZulu` be computed?

		3L33T H@X0R
		-----------
		- `groupAlpha`: odd-parity bit-masked `curveLocations`
		- `groupZulu`: even-parity bit-masked `curveLocations`
		"""
		nonlocal dataframeCurveLocations
		bitWidth: int = int(dataframeCurveLocations['curveLocations'].max()).bit_length()
		if alpha:
			dataframeCurveLocations['groupAlpha'] = dataframeCurveLocations['curveLocations']
			dataframeCurveLocations.loc[:, 'groupAlpha'] &= getLocatorGroupAlpha(bitWidth)
		if zulu:
			dataframeCurveLocations['groupZulu'] = dataframeCurveLocations['curveLocations']
			dataframeCurveLocations.loc[:, 'groupZulu'] &= getLocatorGroupZulu(bitWidth)
			dataframeCurveLocations.loc[:, 'groupZulu'] //= 2**1 # (groupZulu >> 1)

	def outfitDataframeAnalyzed(indexTransferMatrix: int) -> None:
		nonlocal dataframeAnalyzed, indexStartAnalyzed
		lengthDataframe: int = max(100000, getEstimatedLengthCurveLocationsAnalyzed(indexTransferMatrix))
		dataframeAnalyzed = dataframeAnalyzed.reindex(range(lengthDataframe), fill_value=0)
		indexStartAnalyzed = 0

	def outfitDataframeCurveLocations() -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations = dataframeCurveLocations.iloc[0:0]
		dataframeCurveLocations['curveLocations'] = dataframeAnalyzed['analyzed']
		dataframeCurveLocations['distinctCrossings'] = dataframeAnalyzed['distinctCrossings']
		dataframeCurveLocations['dropModify'] = groupsHaveCurvesNotEven
		dataframeCurveLocations['analyzed'] = 0

		dataframeAnalyzed = dataframeAnalyzed.iloc[0:0]
		computeCurveGroups()

	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=list(dictionaryCurveLocations.keys()), dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=list(dictionaryCurveLocations.values()), dtype=datatypeDistinctCrossings)
		}, dtype=datatypeCurveLocations
	)
	del dictionaryCurveLocations

	dataframeCurveLocations = pandas.DataFrame({
		'curveLocations': pandas.Series(name='curveLocations', data=0, dtype=datatypeCurveLocations)
		, 'groupAlpha': pandas.Series(name='groupAlpha', data=0, dtype=datatypeCurveLocations)
		, 'groupZulu': pandas.Series(name='groupZulu', data=0, dtype=datatypeCurveLocations)
		, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=0, dtype=datatypeDistinctCrossings)
		, 'dropModify': pandas.Series(name='dropModify', data=groupsHaveCurvesNotEven, dtype=numpy.int8)
		}
	)

	indexStartAnalyzed: int = 0

	while (indexTransferMatrix > 0
		and (int(dataframeAnalyzed['analyzed'].max()).bit_length() <= bitWidthCurveLocationsMaximum)
		and (int(dataframeAnalyzed['distinctCrossings'].max()).bit_length() <= bitWidthDistinctCrossingsMaximum)):

		indexTransferMatrix -= 1

		MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(indexTransferMatrix)

		outfitDataframeCurveLocations()
		goByeBye()

		outfitDataframeAnalyzed(indexTransferMatrix)
		goByeBye()

		analyzeCurveLocationsSimple(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsAlpha(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsZulu(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsAligned(MAXIMUMcurveLocations)
		goByeBye()

		dataframeAnalyzed = dataframeAnalyzed.iloc[0:indexStartAnalyzed].groupby('analyzed', sort=False)['distinctCrossings'].aggregate('sum').reset_index()
		print(_n, indexTransferMatrix, indexStartAnalyzed, sep=',')

	return (indexTransferMatrix, dataframeAnalyzed.set_index('analyzed')['distinctCrossings'].to_dict())

# ----------------- doTheNeedful --------------------------------------------------------------------------------------

def doTheNeedful(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int]) -> int:
	"""Compute a(n) meanders with the transfer matrix algorithm.

	Parameters
	----------
	indexTransferMatrix : int
		The current index in the transfer matrix algorithm.
	dictionaryCurveLocations : dict[int, int]
		A dictionary of `curveLocations` to `distinctCrossings`.

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
	global _n  # noqa: PLW0603
	_n = indexTransferMatrix + 1
	while indexTransferMatrix > 0:
		bitWidthCurveLocations: int = max(dictionaryCurveLocations.keys()).bit_length()
		bitWidthDistinctCrossings: int = max(dictionaryCurveLocations.values()).bit_length()

		goByeBye()

		if (bitWidthCurveLocations > bitWidthCurveLocationsMaximum) or (bitWidthDistinctCrossings > bitWidthDistinctCrossingsMaximum):
			# sys.stdout.write(f"countBigInt({indexTransferMatrix}).\t")
			indexTransferMatrix, dictionaryCurveLocations = countBigInt(indexTransferMatrix, dictionaryCurveLocations)
		else:
			# sys.stdout.write(f"countPandas({indexTransferMatrix}).\t")
			indexTransferMatrix, dictionaryCurveLocations = countPandas(indexTransferMatrix, dictionaryCurveLocations)

	return sum(dictionaryCurveLocations.values())

@cache
def A000682(n: int) -> int:
	"""Compute A000682(n)."""
	if n & 0b1:
		curveLocations: int = 5
	else:
		curveLocations = 1
	listCurveLocations: list[int] = [(curveLocations << 1) | curveLocations]

	MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(n-1)
	while listCurveLocations[-1] < MAXIMUMcurveLocations:
		curveLocations = (curveLocations << 4) | 0b101 # == curveLocations * 2**4 + 5
		listCurveLocations.append((curveLocations << 1) | curveLocations)
	return doTheNeedful(n - 1, dict.fromkeys(listCurveLocations, 1))

@cache
def A005316(n: int) -> int:
	"""Compute A005316(n)."""
	if n & 0b1:
		dictionaryCurveLocations: dict[int, int] = {15: 1}
	else:
		dictionaryCurveLocations = {22: 1}
	return doTheNeedful(n - 1, dictionaryCurveLocations)
