# ruff: noqa
"""Count meanders with matrix transfer algorithm.

Notes
-----
- Odd/even of `groupAlpha` == the odd/even of `curveLocations`. Proof: `groupAlphaIsEven = curveLocations & 1 & 1 ^ 1`.
- Odd/even of `groupZulu` == `curveLocations` second-least significant bit. So `groupZuluIsEven = bool(curveLocations & 2 ^ 2)`.
"""
from functools import cache
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding.reference.A005316facts import bucketsIf_k_EVEN_by_nLess_k, bucketsIf_k_ODD_by_nLess_k
from math import exp, log
from typing import NamedTuple
from warnings import warn
import dataclasses
import math
import numpy
import pandas

@dataclasses.dataclass
class MatrixMeandersState:
	"""State."""

	n: int
	oeisID: str
	kOfMatrix: int
	dictionaryCurveLocations: dict[int, int]

	datatypeCurveLocations: type = numpy.uint64
	datatypeDistinctCrossings: type = numpy.uint64

	bitWidthCurveLocationsMaximum: int | None = None
	bitWidthDistinctCrossingsMaximum: int | None = None

	bitWidth: int = 0
	indexStartAnalyzed: int = 0

	def __post_init__(self) -> None:
		"""Post init."""
		if self.bitWidthCurveLocationsMaximum is None:
			_bitWidthOfFixedSizeInteger: int = numpy.dtype(self.datatypeCurveLocations).itemsize * 8 # bits

			_offsetNecessary: int = 3 # For example, `groupZulu << 3`.
			_offsetSafety: int = 1 # I don't have mathematical proof of how many extra bits I need.
			_offset: int = _offsetNecessary + _offsetSafety

			self.bitWidthCurveLocationsMaximum = _bitWidthOfFixedSizeInteger - _offset

			del _bitWidthOfFixedSizeInteger, _offsetNecessary, _offsetSafety, _offset

		if self.bitWidthDistinctCrossingsMaximum is None:
			_bitWidthOfFixedSizeInteger: int = numpy.dtype(self.datatypeDistinctCrossings).itemsize * 8 # bits

			_offsetNecessary: int = 0 # I don't know of any.
			_offsetEstimation: int = 3 # See reference directory.
			_offsetSafety: int = 1
			_offset: int = _offsetNecessary + _offsetEstimation + _offsetSafety

			self.bitWidthDistinctCrossingsMaximum = _bitWidthOfFixedSizeInteger - _offset

			del _bitWidthOfFixedSizeInteger, _offsetNecessary, _offsetEstimation, _offsetSafety, _offset

	@property
	def MAXIMUMcurveLocations(self) -> int:
		"""Compute the maximum value of `curveLocations` for the current iteration of the transfer matrix."""
		return 1 << (2 * self.kOfMatrix + 4)

	@property
	def locatorGroupAlpha(self) -> int:
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
		return sum(1 << one for one in range(0, self.bitWidth, 2))

	@property
	def locatorGroupZulu(self) -> int:
		"""Compute an even-parity bit-mask with `bitWidth` bits."""
		return sum(1 << one for one in range(1, self.bitWidth, 2))

# ----------------- support functions ---------------------------------------------------------------------------------
@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def getBucketsTotal(state: MatrixMeandersState, safetyMultiplicand: float = 1.2) -> int:
	"""Estimate the total number of non-unique curveLocations that will be computed from the existing curveLocations.

	Notes
	-----
	Subexponential bucketsTotal unified estimator parameters (derived in reference notebook).

	The model is: log(buckets) = intercept + bN*log(n) + bK*log(k) + bD*log(n-k) + g_r*(k/n) + g_r2*(k/n)^2 + g_s*((n-k)/n) + offset(subseries)
	Subseries key: f"{oeisID}_kOdd={int(kIsOdd)}_dOdd={int(nLess_kIsOdd)}" with a reference subseries offset of zero.
	These coefficients intentionally remain in-source (SSOT) to avoid runtime JSON parsing overhead and to support reproducibility.
	"""
	class ImaKey(NamedTuple):
		"""keys for dictionaries."""

		oeisID: str
		kIsOdd: bool
		nLess_kIsOdd: bool

	dictionaryExponentialCoefficients: dict[ImaKey, float] = {
		(ImaKey(oeisID='', kIsOdd=False, nLess_kIsOdd=True)): 0.834,
		(ImaKey(oeisID='', kIsOdd=False, nLess_kIsOdd=False)): 1.5803,
		(ImaKey(oeisID='', kIsOdd=True, nLess_kIsOdd=True)): 1.556,
		(ImaKey(oeisID='', kIsOdd=True, nLess_kIsOdd=False)): 1.8047,
	}

	logarithmicOffsets: dict[ImaKey, float] ={
		(ImaKey('A000682', kIsOdd=False, nLess_kIsOdd=False)): 0.0,
		(ImaKey('A000682', kIsOdd=False, nLess_kIsOdd=True)): -0.07302547148212568,
		(ImaKey('A000682', kIsOdd=True, nLess_kIsOdd=False)): -0.00595307513938792,
		(ImaKey('A000682', kIsOdd=True, nLess_kIsOdd=True)): -0.012201222865243722,
		(ImaKey('A005316', kIsOdd=False, nLess_kIsOdd=False)): -0.6392728422078733,
		(ImaKey('A005316', kIsOdd=False, nLess_kIsOdd=True)): -0.6904925299923548,
		(ImaKey('A005316', kIsOdd=True, nLess_kIsOdd=False)): 0.0,
		(ImaKey('A005316', kIsOdd=True, nLess_kIsOdd=True)): 0.0,
	}

	logarithmicParameters: dict[str, float] = {
		'intercept': -166.1750299793178,
		'log(n)': 1259.0051001675547,
		'log(k)': -396.4306071056408,
		'log(nLess_k)': -854.3309503739766,
		'k/n': 716.530410654819,
		'(k/n)^2': -2527.035113444166,
		'normalized k': -882.7054406339189,
	}

	bucketsTotalMaximumBy_kOfMatrix: dict[int, int] = {1:3, 2:12, 3:40, 4:125, 5:392, 6:1254, 7:4087, 8:13623, 9:46181
		, 10:159137, 11:555469, 12:1961369, 13:6991893, 14:25134208}

	xCommon = 1.57

	nLess_k: int = state.n - state.kOfMatrix
	kIsOdd: bool = bool(state.kOfMatrix & 1)
	nLess_kIsOdd: bool = bool(nLess_k & 1)
	kIsEven: bool = not kIsOdd
	bucketsTotal: int = -8

	"""NOTE temporary notes
	I have a fault in my thinking. bucketsTotal increases as k decreases until ~0.4k, then bucketsTotal decreases rapidly to 1. I
	have ignored the decreasing side. In the formulas for estimation, I didn't differentiate between increasing and decreasing.
	So, I probably need to refine the formulas. I guess I need to add checks to the if/else monster, too.

	While buckets is increasing:
	3 types of estimates:
	1. Exponential growth.
	2. Logarithmic growth.
	3. Hard ceiling.

	"""

	# If I know bucketsTotal is maxed out.
	if state.kOfMatrix <= ((state.n - 1 - (state.kOfMatrix % 2)) // 3):
		if (state.kOfMatrix in bucketsTotalMaximumBy_kOfMatrix):
			bucketsTotal = bucketsTotalMaximumBy_kOfMatrix[state.kOfMatrix]
		else:
			c = 0.95037
			r = 3.3591258254
			if kIsOdd:
				c = 0.92444
				r = 3.35776

			bucketsTotal = int(c * r**state.kOfMatrix * safetyMultiplicand)

	# Exponential growth.
	elif state.kOfMatrix > nLess_k:
		# If I already know bucketsTotal.
		if (state.oeisID == 'A005316') and kIsOdd and (nLess_k in bucketsIf_k_ODD_by_nLess_k):
			bucketsTotal = bucketsIf_k_ODD_by_nLess_k[nLess_k]
		# If I already know bucketsTotal.
		elif (state.oeisID == 'A005316') and kIsEven and (nLess_k in bucketsIf_k_EVEN_by_nLess_k):
			bucketsTotal = bucketsIf_k_EVEN_by_nLess_k[nLess_k]
		# If I can estimate bucketsTotal during exponential growth.
		elif state.kOfMatrix > nLess_k:
			xInstant: int = math.ceil(nLess_k / 2)
			A000682adjustStartingCurveLocations: float = 0.25
			startingConditionsCoefficient: float = dictionaryExponentialCoefficients[ImaKey('', kIsOdd, nLess_kIsOdd)]
			if kIsOdd and nLess_kIsOdd:
				A000682adjustStartingCurveLocations = 0.0
			if state.oeisID == 'A000682': # NOTE Net effect is between `*= n` and `*= n * 2.2` if n=46.
				startingConditionsCoefficient *= state.n * (((state.n // 2) + 2) ** A000682adjustStartingCurveLocations)
			bucketsTotal = int(startingConditionsCoefficient * math.exp(xCommon * xInstant))

	# If `kOfMatrix` is low, use maximum bucketsTotal. 1. Can't underestimate. 2. Skip computation that can underestimate. 3. The
	# potential difference in memory use is relatively small.
	elif state.kOfMatrix <= max(bucketsTotalMaximumBy_kOfMatrix.keys()):
		bucketsTotal = bucketsTotalMaximumBy_kOfMatrix[state.kOfMatrix]

	# Logarithmic growth, power-law + ratio curvature + subseries offset
	elif state.kOfMatrix > ((state.n - (state.n % 3)) // 3):  # noqa: ERA001
		xPower: float = (0
			+ logarithmicParameters['intercept']
			+ logarithmicParameters['log(n)'] * log(state.n)
			+ logarithmicParameters['log(k)'] * log(state.kOfMatrix)
			+ logarithmicParameters['log(nLess_k)'] * log(nLess_k)
			+ logarithmicParameters['k/n'] * (state.kOfMatrix / state.n)
			+ logarithmicParameters['(k/n)^2'] * (state.kOfMatrix / state.n)**2
			+ logarithmicParameters['normalized k'] * nLess_k / state.n
			+ logarithmicOffsets[ImaKey(state.oeisID, kIsOdd, nLess_kIsOdd)]
		)

		bucketsTotal = int(exp(xPower * safetyMultiplicand))

	else:
		message = "I shouldn't be here."
		raise SystemError(message)
	return bucketsTotal

def outfitDictionaryCurveGroups(state: MatrixMeandersState) -> dict[tuple[int, int], int]:
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

# ----------------- counting functions --------------------------------------------------------------------------------

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

	while (state.kOfMatrix > 0
		and ((max(state.dictionaryCurveLocations.keys()).bit_length() > raiseIfNone(state.bitWidthCurveLocationsMaximum))
		or (max(state.dictionaryCurveLocations.values()).bit_length() > raiseIfNone(state.bitWidthDistinctCrossingsMaximum)))):

		state.kOfMatrix -= 1

		dictionaryCurveGroups = outfitDictionaryCurveGroups(state)
		state.dictionaryCurveLocations.clear()
		LessThanMaximumTotal: int = 0 # for data collection

		for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
			groupAlphaCurves: bool = groupAlpha > 1
			groupZuluHasCurves: bool = groupZulu > 1
			groupAlphaIsEven = groupZuluIsEven = 0

			# simple
			curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
			if curveLocationAnalysis < state.MAXIMUMcurveLocations:
				state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings
				LessThanMaximumTotal += 1

			if groupAlphaCurves:
				curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 1)) << 1)
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings
					LessThanMaximumTotal += 1

			if groupZuluHasCurves:
				curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings
					LessThanMaximumTotal += 1

			# aligned
			if groupAlphaCurves and groupZuluHasCurves and (groupAlphaIsEven or groupZuluIsEven):
				if groupAlphaIsEven and not groupZuluIsEven:
					groupAlpha ^= walkDyckPath(groupAlpha)  # noqa: PLW2901
				elif groupZuluIsEven and not groupAlphaIsEven:
					groupZulu ^= walkDyckPath(groupZulu)  # noqa: PLW2901

				curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
				if curveLocationAnalysis < state.MAXIMUMcurveLocations:
					state.dictionaryCurveLocations[curveLocationAnalysis] = state.dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings
					LessThanMaximumTotal += 1

		if state.n >= 45: # for data collection
			print(state.n, state.kOfMatrix+1, LessThanMaximumTotal, sep=',')  # noqa: T201

	return state

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

	while (state.kOfMatrix > 0
		and (int(dataframeAnalyzed['analyzed'].max()).bit_length() <= raiseIfNone(state.bitWidthCurveLocationsMaximum))
		and (int(dataframeAnalyzed['distinctCrossings'].max()).bit_length() <= raiseIfNone(state.bitWidthDistinctCrossingsMaximum))):

# ruff: noqa: B023

		def aggregateCurveLocations() -> None:
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

# NOTE Step 1: drop unqualified rows

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

# NOTE Step 2: modify rows

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

# NOTE Step 3: compute curveLocations
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
					dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)
					warn(f"Lengthened `dataframeAnalyzed` to {indexStopAnalyzed=}; n={state.n}, {state.kOfMatrix=}.", stacklevel=2)

				dataframeAnalyzed.loc[state.indexStartAnalyzed:indexStopAnalyzed - 1, ['analyzed', 'distinctCrossings']] = (
					dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0), ['analyzed', 'distinctCrossings']
								].to_numpy(dtype=state.datatypeCurveLocations, copy=False)
				)

				state.indexStartAnalyzed = indexStopAnalyzed

			del indexStopAnalyzed

		dataframeCurveLocations = pandas.DataFrame({
			'curveLocations': pandas.Series(name='curveLocations', data=dataframeAnalyzed['analyzed'], copy=True, dtype=state.datatypeCurveLocations)
			, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=state.datatypeCurveLocations)
			, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=dataframeAnalyzed['distinctCrossings'], copy=True, dtype=state.datatypeDistinctCrossings)
			} # pyright: ignore[reportUnknownArgumentType]
		)

		state.bitWidth = int(dataframeCurveLocations['curveLocations'].max()).bit_length()

		del dataframeAnalyzed
		goByeBye()

		length: int = getBucketsTotal(state)
		dataframeAnalyzed = pandas.DataFrame({
			'analyzed': pandas.Series(0, pandas.RangeIndex(length), dtype=state.datatypeCurveLocations, name='analyzed')
			, 'distinctCrossings': pandas.Series(0, pandas.RangeIndex(length), dtype=state.datatypeDistinctCrossings, name='distinctCrossings')
			}, index=pandas.RangeIndex(length), columns=['analyzed', 'distinctCrossings'], dtype=state.datatypeCurveLocations # pyright: ignore[reportUnknownArgumentType]
		)

		state.indexStartAnalyzed = 0

		state.kOfMatrix -= 1

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

# ----------------- doTheNeedful --------------------------------------------------------------------------------------

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
		bitWidthCurveLocations: int = max(state.dictionaryCurveLocations.keys()).bit_length()
		bitWidthDistinctCrossings: int = max(state.dictionaryCurveLocations.values()).bit_length()

		goByeBye()

		if (bitWidthCurveLocations > raiseIfNone(state.bitWidthCurveLocationsMaximum)) or (bitWidthDistinctCrossings > raiseIfNone(state.bitWidthDistinctCrossingsMaximum)):
			state = countBigInt(state)
		else:
			state = countPandas(state)

	return sum(state.dictionaryCurveLocations.values())

@cache
def A000682(n: int) -> int:
	"""Compute A000682(n)."""
	oeisID = 'A000682'

	kOfMatrix: int = n - 1

	if n & 0b1:
		curveLocations: int = 5
	else:
		curveLocations = 1
	listCurveLocations: list[int] = [(curveLocations << 1) | curveLocations]

	MAXIMUMcurveLocations: int = 1 << (2 * kOfMatrix + 4)
	while listCurveLocations[-1] < MAXIMUMcurveLocations:
		curveLocations = (curveLocations << 4) | 0b101 # == curveLocations * 2**4 + 5
		listCurveLocations.append((curveLocations << 1) | curveLocations)

	dictionaryCurveLocations=dict.fromkeys(listCurveLocations, 1)

	state = MatrixMeandersState(n, oeisID, kOfMatrix, dictionaryCurveLocations)

	return doTheNeedful(state)

@cache
def A005316(n: int) -> int:
	"""Compute A005316(n)."""
	oeisID = 'A005316'

	kOfMatrix: int = n - 1

	if n & 0b1:
		dictionaryCurveLocations: dict[int, int] = {15: 1}
	else:
		dictionaryCurveLocations = {22: 1}

	state = MatrixMeandersState(n, oeisID, kOfMatrix, dictionaryCurveLocations)

	return doTheNeedful(state)
