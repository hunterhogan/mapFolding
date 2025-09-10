"""Count meanders with matrix transfer algorithm."""
from functools import cache
from gc import collect as goByeBye
from mapFolding.reference.A005316facts import (
	bucketsIf_k_EVEN_by_nLess_k as rowsA005316If_k_EVEN_by_nLess_k,
	bucketsIf_k_ODD_by_nLess_k as rowsA005316If_k_ODD_by_nLess_k)
from math import e, exp
from warnings import warn
import numpy
import pandas

# ----------------- environment configuration -------------------------------------------------------------------------
datatypeCurveLocations = datatypeDistinctCrossings = numpy.uint64

_bitWidthOfFixedSizeInteger: int = numpy.dtype(datatypeCurveLocations).itemsize * 8 # bits

_offsetNecessary: int = 3 # For example, `groupZulu << 3`.
_offsetSafety: int = 1 # I don't have mathematical proof of how many extra bits I need.
_offset: int = _offsetNecessary + _offsetSafety
bitWidthCurveLocationsMaximum: int = _bitWidthOfFixedSizeInteger - _offset

del _offsetNecessary, _offsetSafety, _offset

_offsetNecessary: int = 0 # I don't know of any.
_offsetEstimation: int = 3 # See reference directory.
_offsetSafety: int = 1
_offset: int = _offsetNecessary + _offsetEstimation + _offsetSafety

bitWidthDistinctCrossingsMaximum: int = _bitWidthOfFixedSizeInteger - _offset

del _offsetNecessary, _offsetEstimation, _offsetSafety, _offset
del _bitWidthOfFixedSizeInteger

_n: int = 0
_oeisID: str | None = None

# ----------------- lookup tables -------------------------------------------------------------------------------------

rowsMaximumBy_kOfMatrix: dict[int, int] = {1:3, 2:12, 3:40, 4:125, 5:392, 6:1254, 7:4087, 8:13623, 9:46181
	, 10:159137, 11:555469, 12:1961369, 13:6991893, 14:25134208}

# ----------------- support functions ---------------------------------------------------------------------------------
@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def getLengthDataframeAnalyzed(k: int, safetyMultiplicand: float = 1.3, safetyAddend: int = 100000) -> int:
	"""Estimate the total number of non-unique curveLocations that will be computed from the existing curveLocations."""
	nLess_k: int = _n - k

	length: int = -10000
	rowsEstimated: float = 0
	rowsKnown: int = -10
	if (k in rowsMaximumBy_kOfMatrix) and (k*2+2 <= nLess_k):
		rowsKnown = rowsMaximumBy_kOfMatrix[k]
	elif (_oeisID == 'A005316') and (k > nLess_k):
		if k & 1: # k is odd
			if nLess_k in rowsA005316If_k_ODD_by_nLess_k:
				rowsKnown = rowsA005316If_k_ODD_by_nLess_k[nLess_k]
			elif nLess_k & 1: # k is odd; n-k is odd
				nLess_k = nLess_k//2 + 1
				rowsEstimated = 7.498868 + 0.8513946*e**(1.527641*nLess_k)
			else: # k is odd; n-k is even
				nLess_k = nLess_k//2
				rowsEstimated = -13.50074 + 2.175569*e**(1.528578*nLess_k)
		elif nLess_k in rowsA005316If_k_EVEN_by_nLess_k: # k is even
			rowsKnown = rowsA005316If_k_EVEN_by_nLess_k[nLess_k]
		elif nLess_k & 1: # k is even; n-k is odd
			nLess_k = nLess_k//2 + 1
			rowsEstimated = 6.399027 + 1.042025*e**(1.526213*nLess_k)
		else: # k is even; n-k is even
			nLess_k = nLess_k//2
			rowsEstimated = 1.826639 + 1.867297*e**(1.525544*nLess_k)
		rowsEstimated = int(rowsEstimated)
	elif k <= max(rowsMaximumBy_kOfMatrix.keys()):
		rowsKnown = rowsMaximumBy_kOfMatrix[k]
	else:
		rowsEstimated = predict_less_than_max(k - 1)
	if rowsEstimated:
		length = int(rowsEstimated * safetyMultiplicand) + safetyAddend + 1
	else:
		length = rowsKnown
	return length

def predict_less_than_max(k: int) -> float:
	"""Predict."""
# TODO replace this old estimate.
	n = _n
	b = max(0.0, min(float(k), n))
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

def getMAXIMUMcurveLocations(kOfMatrix: int) -> int:
	"""Compute the maximum value of `curveLocations` for the current iteration of the transfer matrix."""
	return 1 << (2 * kOfMatrix + 4)

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

def countBigInt(kOfMatrix: int, dictionaryCurveLocations: dict[int, int]) -> tuple[int, dict[int, int]]:
	"""Count meanders with matrix transfer algorithm using Python primitive `int` contained in a Python primitive `dict`.

	Parameters
	----------
	kOfMatrix : int
		The current index in the transfer matrix algorithm.
	dictionaryCurveLocations : dict[int, int]
		A dictionary of `curveLocations` to `distinctCrossings`.

	Returns
	-------
	matrixMeandersState : tuple[int, dict[int, int]]
		The state of the algorithm computation: the current `kOfMatrix`, `curveLocations`, and `distinctCrossings`.

	Notes
	-----
	The algorithm is sophisticated, but this implementation is straightforward. Compute each index one at a time, compute each
	`curveLocations` one at a time, and compute each type of analysis one at a time.
	"""
	dictionaryCurveGroups: dict[tuple[int, int], int] = {}

	while (kOfMatrix > 0
		and ((max(dictionaryCurveLocations.keys()).bit_length() > bitWidthCurveLocationsMaximum)
		or (max(dictionaryCurveLocations.values()).bit_length() > bitWidthDistinctCrossingsMaximum))):

		kOfMatrix -= 1

		MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(kOfMatrix)
		dictionaryCurveGroups = outfitDictionaryCurveGroups(dictionaryCurveLocations)
		dictionaryCurveLocations.clear()

		for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
			groupAlphaCurves: bool = groupAlpha > 1
			groupZuluHasCurves: bool = groupZulu > 1
			groupAlphaIsEven = groupZuluIsEven = 0

			# simple
			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			if curveLocationAnalysis < MAXIMUMcurveLocations:
				dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 1)) << 1)
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			if groupZuluHasCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			# aligned
			if groupAlphaCurves and groupZuluHasCurves and (groupAlphaIsEven or groupZuluIsEven):
				if groupAlphaIsEven and not groupZuluIsEven:
					groupAlpha ^= walkDyckPath(groupAlpha)  # noqa: PLW2901
				elif groupZuluIsEven and not groupAlphaIsEven:
					groupZulu ^= walkDyckPath(groupZulu)  # noqa: PLW2901

				curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
				if curveLocationAnalysis < MAXIMUMcurveLocations:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

		if _n >= 45: # for data collection
			print(_n, kOfMatrix+1, len(dictionaryCurveLocations), sep=',')  # noqa: T201

	return (kOfMatrix, dictionaryCurveLocations)

def countPandas(kOfMatrix: int, dictionaryCurveLocations: dict[int, int]) -> tuple[int, dict[int, int]]:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame.

	kOfMatrix : int
		The current index in the transfer matrix algorithm.
	dictionaryCurveLocations : dict[int, int]
		A dictionary of `curveLocations` to `distinctCrossings`.

	Returns
	-------
	matrixMeandersState : tuple[int, dict[int, int]]
		The state of the algorithm computation: the current `kOfMatrix`, `curveLocations`, and `distinctCrossings`.
	"""
	def aggregateCurveLocations(MAXIMUMcurveLocations: int) -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations, indexStartAnalyzed

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = 0

		indexStopAnalyzed: int = indexStartAnalyzed + int((dataframeCurveLocations['analyzed'] > 0).sum())

		goByeBye()

		if indexStopAnalyzed > indexStartAnalyzed:
			if len(dataframeAnalyzed.index) < indexStopAnalyzed:
				dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)
				warn(f"Lengthened `dataframeAnalyzed` to {indexStopAnalyzed=}; n={_n}, {kOfMatrix=}.", stacklevel=2)

			dataframeAnalyzed.loc[indexStartAnalyzed:indexStopAnalyzed - 1, ['analyzed', 'distinctCrossings']] = (
				dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0), ['analyzed', 'distinctCrossings']].to_numpy(dtype=datatypeCurveLocations, copy=False)
			)

			indexStartAnalyzed = indexStopAnalyzed

		dataframeCurveLocations.loc[:, 'analyzed'] = 0

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

		Notes
		-----
		"IsEven" vs "AtEven":
		- `groupAlphaIsEven` ≅ `groupAlphaAtEven`
		- `groupZuluIsEven` ≅ `groupZuluAtEven`
		- Semantically, "IsEven" is the evaluation of a single value, while "AtEven" is the evaluation of multiple values telling
		us where the even numbers are *at*, hence "AtEven".
		- Pragmatically, I need to avoid a name collision between global `groupAlphaIsEven` and local `groupAlphaIsEven` in
		`countBigInt`.
		"""
		nonlocal dataframeCurveLocations

# NOTE drop unqualified rows

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
		ImaGroupZulpha &= getLocatorGroupAlpha(bitWidth) # Ima `groupAlpha`.

		dataframeCurveLocations = dataframeCurveLocations.loc[(ImaGroupZulpha > 1)] # if groupAlphaHasCurves

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
		ImaGroupZulpha &= getLocatorGroupZulu(bitWidth) # Ima `groupZulu`.
		ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

		dataframeCurveLocations = dataframeCurveLocations.loc[(ImaGroupZulpha > 1)] # if groupZuluHasCurves

		ImaGroupZulpha = ImaGroupZulpha.loc[(ImaGroupZulpha > 1)] # decrease size to match dataframeCurveLocations
		ImaGroupZulpha &= 1 # (groupZulu & 1)
		ImaGroupZulpha = 1 - ImaGroupZulpha # (1 - (groupZulu ...))
		dataframeCurveLocations.loc[:, 'analyzed'] = ImaGroupZulpha.copy()

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
		ImaGroupZulpha &= getLocatorGroupAlpha(bitWidth) # Ima `groupAlpha`.
		ImaGroupZulpha &= 1 # (groupAlpha & 1)
		ImaGroupZulpha = 1 - ImaGroupZulpha # (1 - (groupAlpha ...))
		ImaGroupZulpha = ImaGroupZulpha.astype(bool)

		dataframeCurveLocations = dataframeCurveLocations.loc[(ImaGroupZulpha) | (dataframeCurveLocations.loc[:, 'analyzed'])] # if (groupAlphaIsEven or groupZuluIsEven)

# NOTE Above this line, I am only using the current minimum of data structures: i.e., no selectors.

# NOTE unqualified rows are dropped
# NOTE modify rows

# NOTE `selectorGroupZuluAtEven` until I can figure out how to eliminate it.
		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
		ImaGroupZulpha &= getLocatorGroupZulu(bitWidth) # Ima `groupZulu`.
		ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)
		ImaGroupZulpha &= 1 # (groupZulu & 1)
		ImaGroupZulpha = 1 - ImaGroupZulpha # (1 - (groupZulu ...))
		selectorGroupZuluAtEven = ImaGroupZulpha.astype(bool)

		# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
		ImaGroupZulpha &= getLocatorGroupAlpha(bitWidth) # Ima `groupAlpha`.

		ImaGroupZulpha.loc[(~selectorGroupZuluAtEven)] = datatypeCurveLocations( # pyright: ignore[reportCallIssue, reportArgumentType]
			flipTheExtra_0b1AsUfunc(ImaGroupZulpha.loc[(~selectorGroupZuluAtEven)]))
		ImaGroupZulpha //= 2**2 # (groupAlpha >> 2)

		del selectorGroupZuluAtEven
		goByeBye()

		dataframeCurveLocations.loc[:, 'analyzed'] = ImaGroupZulpha # (groupAlpha ...)

# NOTE `selectorGroupAlphaAtEven` until I can figure out how to eliminate it.
		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
		ImaGroupZulpha &= getLocatorGroupAlpha(bitWidth) # Ima `groupAlpha`.
		ImaGroupZulpha &= 1 # (groupAlpha & 1)
		ImaGroupZulpha = 1 - ImaGroupZulpha # (1 - (groupAlpha ...))
		selectorGroupAlphaAtEven = ImaGroupZulpha.astype(bool)

		# if groupZuluIsEven and not groupAlphaIsEven, modifyGroupZuluPairedToOdd
		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
		ImaGroupZulpha &= getLocatorGroupZulu(bitWidth) # Ima `groupZulu`.
		ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

		ImaGroupZulpha.loc[(~selectorGroupAlphaAtEven)] = datatypeCurveLocations( # pyright: ignore[reportCallIssue, reportArgumentType]
			flipTheExtra_0b1AsUfunc(ImaGroupZulpha.loc[(~selectorGroupAlphaAtEven)]))

		del selectorGroupAlphaAtEven
		goByeBye()

# NOTE finish compute curveLocations

		ImaGroupZulpha //= 2**2 # (groupZulu >> 2)
		ImaGroupZulpha *= 2**1 # ((groupZulu ...) << 1)

		dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupZulu ...)

		del ImaGroupZulpha

	def analyzeCurveLocationsAlpha() -> None:
		"""Compute `curveLocations` from `groupAlpha`.

		Formula
		-------
		```python
		if groupAlpha > 1:
			curveLocations = ((1 - (groupAlpha & 1)) << 1) | (groupZulu << 3) | (groupAlpha >> 2)
		```
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['curveLocations']
		dataframeCurveLocations.loc[:, 'analyzed'] &= getLocatorGroupAlpha(bitWidth)

		dataframeCurveLocations.loc[:, 'analyzed'] &= 1 # (groupAlpha & 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupAlpha ...))

		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**1 # ((groupAlpha ...) << 1)

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
		ImaGroupZulpha &= getLocatorGroupZulu(bitWidth) # Ima `groupZulu`.
		ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

		ImaGroupZulpha *= 2**3 # (groupZulu << 3)
		dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupZulu ...)
		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # ... | (groupAlpha >> 2)

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
		ImaGroupZulpha &= getLocatorGroupAlpha(bitWidth) # Ima `groupAlpha`.

		dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupAlpha)
		dataframeCurveLocations.loc[:, 'analyzed'] //= 2**2 # (... >> 2)

		dataframeCurveLocations.loc[(ImaGroupZulpha <= 1), 'analyzed'] = 0 # if groupAlpha > 1

		del ImaGroupZulpha

	def analyzeCurveLocationsSimple() -> None:
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
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['curveLocations']
		dataframeCurveLocations.loc[:, 'analyzed'] &= getLocatorGroupAlpha(bitWidth)

		groupZulu = dataframeCurveLocations['curveLocations'].copy()
		groupZulu &= getLocatorGroupZulu(bitWidth)
		groupZulu //= 2**1 # (groupZulu >> 1)
		groupZulu *= 2**1 # (groupZulu << 1)

		dataframeCurveLocations.loc[:, 'analyzed'] |= groupZulu # ((groupAlpha | (groupZulu ...))

		del groupZulu

		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # (... << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] += 3 # (...) | 3

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
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['curveLocations']
		dataframeCurveLocations.loc[:, 'analyzed'] &= getLocatorGroupZulu(bitWidth)
		dataframeCurveLocations.loc[:, 'analyzed'] //= 2**1 # (groupZulu >> 1)

		dataframeCurveLocations.loc[:, 'analyzed'] &= 1 # (groupZulu & 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupZulu ...))

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
		ImaGroupZulpha &= getLocatorGroupAlpha(bitWidth) # Ima `groupAlpha`.

		ImaGroupZulpha *= 2**2 # (groupAlpha << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupAlpha ...)

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
		ImaGroupZulpha &= getLocatorGroupZulu(bitWidth) # Ima `groupZulu`.
		ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

		ImaGroupZulpha //= 2**1 # (groupZulu >> 1)

		dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupZulu ...)

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
		ImaGroupZulpha &= getLocatorGroupZulu(bitWidth) # Ima `groupZulu`.
		ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

		dataframeCurveLocations.loc[ImaGroupZulpha <= 1, 'analyzed'] = 0 # if groupZulu > 1

		del ImaGroupZulpha

	def outfitDataframeAnalyzed(kOfMatrix: int) -> None:
		nonlocal dataframeAnalyzed, indexStartAnalyzed
		dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(getLengthDataframeAnalyzed(kOfMatrix)), fill_value=0)
		indexStartAnalyzed = 0

	def outfitDataframeCurveLocations() -> None:
		nonlocal bitWidth, dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations = dataframeCurveLocations.iloc[0:0]
		dataframeCurveLocations['curveLocations'] = dataframeAnalyzed['analyzed']
		dataframeCurveLocations['distinctCrossings'] = dataframeAnalyzed['distinctCrossings']
		dataframeCurveLocations['analyzed'] = 0
		bitWidth = int(dataframeCurveLocations['curveLocations'].max()).bit_length()

		dataframeAnalyzed = dataframeAnalyzed.iloc[0:0]

	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=list(dictionaryCurveLocations.keys()), dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=list(dictionaryCurveLocations.values()), dtype=datatypeDistinctCrossings)
		}, dtype=datatypeCurveLocations
	)
	del dictionaryCurveLocations

	dataframeCurveLocations = pandas.DataFrame({
		'curveLocations': pandas.Series(name='curveLocations', data=0, dtype=datatypeCurveLocations)
		, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=0, dtype=datatypeDistinctCrossings)
		}, dtype=datatypeCurveLocations
	)

	bitWidth: int = 0
	indexStartAnalyzed: int = 0

	while (kOfMatrix > 0
		and (int(dataframeAnalyzed['analyzed'].max()).bit_length() <= bitWidthCurveLocationsMaximum)
		and (int(dataframeAnalyzed['distinctCrossings'].max()).bit_length() <= bitWidthDistinctCrossingsMaximum)):

		outfitDataframeCurveLocations()
		goByeBye()

		outfitDataframeAnalyzed(kOfMatrix)
		goByeBye()

		kOfMatrix -= 1
		MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(kOfMatrix)

		analyzeCurveLocationsSimple()
		aggregateCurveLocations(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsAlpha()
		aggregateCurveLocations(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsZulu()
		aggregateCurveLocations(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsAligned()
		aggregateCurveLocations(MAXIMUMcurveLocations)
		dataframeCurveLocations = dataframeCurveLocations.iloc[0:0]
		goByeBye()

		dataframeAnalyzed = dataframeAnalyzed.iloc[0:indexStartAnalyzed].groupby('analyzed', sort=False)['distinctCrossings'].aggregate('sum').reset_index()
		if _n >= 45:  # for data collection
			print(_n, kOfMatrix+1, indexStartAnalyzed, sep=',')  # noqa: T201

	return (kOfMatrix, dataframeAnalyzed.set_index('analyzed')['distinctCrossings'].to_dict())

# ----------------- doTheNeedful --------------------------------------------------------------------------------------

def doTheNeedful(kOfMatrix: int, dictionaryCurveLocations: dict[int, int]) -> int:
	"""Compute a(n) meanders with the transfer matrix algorithm.

	Parameters
	----------
	kOfMatrix : int
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
	while kOfMatrix > 0:
		bitWidthCurveLocations: int = max(dictionaryCurveLocations.keys()).bit_length()
		bitWidthDistinctCrossings: int = max(dictionaryCurveLocations.values()).bit_length()

		goByeBye()

		if (bitWidthCurveLocations > bitWidthCurveLocationsMaximum) or (bitWidthDistinctCrossings > bitWidthDistinctCrossingsMaximum):
			kOfMatrix, dictionaryCurveLocations = countBigInt(kOfMatrix, dictionaryCurveLocations)
		else:
			kOfMatrix, dictionaryCurveLocations = countPandas(kOfMatrix, dictionaryCurveLocations)

	return sum(dictionaryCurveLocations.values())

@cache
def A000682(n: int) -> int:
	"""Compute A000682(n)."""
	global _n, _oeisID  # noqa: PLW0603
	_n = n
	_oeisID = 'A000682'

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
	global _n, _oeisID  # noqa: PLW0603
	_n = n
	_oeisID = 'A005316'

	if n & 0b1:
		dictionaryCurveLocations: dict[int, int] = {15: 1}
	else:
		dictionaryCurveLocations = {22: 1}
	return doTheNeedful(n - 1, dictionaryCurveLocations)
