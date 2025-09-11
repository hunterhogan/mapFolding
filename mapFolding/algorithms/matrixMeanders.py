"""Count meanders with matrix transfer algorithm."""
from functools import cache
from hunterMakesPy import raiseIfNone
from mapFolding.reference.A005316facts import bucketsIf_k_EVEN_by_nLess_k, bucketsIf_k_ODD_by_nLess_k
from mapFolding.trim_memory import trim_ram as goByeBye
from math import e, exp
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

# ----------------- lookup tables -------------------------------------------------------------------------------------

bucketsTotalMaximumBy_kOfMatrix: dict[int, int] = {1:3, 2:12, 3:40, 4:125, 5:392, 6:1254, 7:4087, 8:13623, 9:46181
	, 10:159137, 11:555469, 12:1961369, 13:6991893, 14:25134208}

# ----------------- support functions ---------------------------------------------------------------------------------
@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def getBucketsTotal(state: MatrixMeandersState, safetyMultiplicand: float = 1.3, safetyAddend: int = 100000) -> int:
	"""Estimate the total number of non-unique curveLocations that will be computed from the existing curveLocations."""
	xCommon = 1.57

	nLess_k: int = state.n - state.kOfMatrix

	kIsOdd: bool = bool(state.kOfMatrix & 1)
	kIsEven: bool = not kIsOdd
	nLess_kIsOdd: bool = bool(nLess_k & 1)
	nLess_kIsEven: bool = not nLess_kIsOdd

	length: int = -10000
	bucketsEstimated: float = 0
	bucketsTotal: int = -10
	xInstant = nLess_k // 2 + 1
	startingConditionsCoefficient = 1.8047

	# If I know bucketsTotal is maxed out.
	if ((state.kOfMatrix in bucketsTotalMaximumBy_kOfMatrix) and (state.kOfMatrix >= ((state.n - (state.n % 3)) // 3))):
		bucketsTotal = bucketsTotalMaximumBy_kOfMatrix[state.kOfMatrix]
	# If I already know bucketsTotal.
	elif (state.oeisID == 'A005316') and (state.kOfMatrix > nLess_k) and kIsOdd and (nLess_k in bucketsIf_k_ODD_by_nLess_k):
		bucketsTotal = bucketsIf_k_ODD_by_nLess_k[nLess_k]
	# If I already know bucketsTotal.
	elif (state.oeisID == 'A005316') and (state.kOfMatrix > nLess_k) and kIsEven and (nLess_k in bucketsIf_k_EVEN_by_nLess_k):
		bucketsTotal = bucketsIf_k_EVEN_by_nLess_k[nLess_k]
	# If I can estimate bucketsTotal during exponential growth with a formula.
	elif (state.oeisID == 'A005316') and (state.kOfMatrix > nLess_k):
		if kIsEven and nLess_kIsOdd:
			startingConditionsCoefficient = 0.834
			xInstant = nLess_k // 2 + 1
		elif kIsEven and nLess_kIsEven:
			startingConditionsCoefficient = 1.5803
			xInstant = nLess_k // 2
		elif kIsOdd and nLess_kIsOdd:
			startingConditionsCoefficient = 1.556
			xInstant = nLess_k // 2 + 1
		elif kIsOdd and nLess_kIsEven:
			startingConditionsCoefficient = 1.8047
			xInstant = nLess_k // 2
		else:
			message = "I shouldn't be here."
			raise SystemError(message)
		bucketsTotal = int(startingConditionsCoefficient * math.exp(xCommon * xInstant))
# TODO elif (state.oeisID == 'A005316') and (state.kOfMatrix < ((state.n - (state.n % 3)) // 3)):
	elif state.kOfMatrix <= max(bucketsTotalMaximumBy_kOfMatrix.keys()):
		bucketsTotal = bucketsTotalMaximumBy_kOfMatrix[state.kOfMatrix]
	else:
		bucketsEstimated = predict_less_than_max(state)
	if bucketsEstimated:
		length = int(bucketsEstimated * safetyMultiplicand) + safetyAddend + 1
	else:
		length = bucketsTotal
	return length

def predict_less_than_max(state: MatrixMeandersState) -> float:
	"""Predict."""
# TODO replace this old estimate.
	n = float(state.n)
	b = max(0.0, min(float(state.kOfMatrix-1), n))
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
	def recordCurveLocations() -> None:
		nonlocal dataframeAnalyzed

		indexStopAnalyzed: int = state.indexStartAnalyzed + int((dataframeCurveLocations['analyzed'] > 0).sum())

		goByeBye()

		if indexStopAnalyzed > state.indexStartAnalyzed:
			if len(dataframeAnalyzed.index) < indexStopAnalyzed:
				dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)
				warn(f"Lengthened `dataframeAnalyzed` to {indexStopAnalyzed=}; n={state.n}, {state.kOfMatrix=}.", stacklevel=2)

			dataframeAnalyzed.loc[state.indexStartAnalyzed:indexStopAnalyzed - 1, ['analyzed', 'distinctCrossings']] = (
				dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0), ['analyzed', 'distinctCrossings']].to_numpy(dtype=state.datatypeCurveLocations, copy=False)
			)

			state.indexStartAnalyzed = indexStopAnalyzed

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

		ImaGroupZulpha = ImaGroupZulpha.loc[(ImaGroupZulpha > 1)] # decrease size to match dataframeCurveLocations
		ImaGroupZulpha &= 1 # (groupZulu & 1)
		ImaGroupZulpha ^= 1 # (1 - (groupZulu ...))
		dataframeCurveLocations.loc[:, 'analyzed'] = ImaGroupZulpha

		del ImaGroupZulpha

		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
		ImaGroupZulpha &= state.locatorGroupAlpha # Ima `groupAlpha`.
		ImaGroupZulpha &= 1 # (groupAlpha & 1)
		ImaGroupZulpha ^= 1 # (1 - (groupAlpha ...))
		ImaGroupZulpha = ImaGroupZulpha.astype(bool)

		dataframeCurveLocations = dataframeCurveLocations.loc[(ImaGroupZulpha) | (dataframeCurveLocations.loc[:, 'analyzed'])] # if (groupAlphaIsEven or groupZuluIsEven)

		del ImaGroupZulpha

# NOTE Step 2: modify rows

		# Make a selector for groupZuluAtEven, so you can modify groupAlpha
		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
		ImaGroupZulpha &= state.locatorGroupZulu # Ima `groupZulu`.
		ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)
		ImaGroupZulpha &= 1 # (groupZulu & 1)
		ImaGroupZulpha ^= 1 # (1 - (groupZulu ...))
		ImaGroupZulpha = ImaGroupZulpha.astype(bool)

		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['curveLocations'] # Ima `groupAlpha`.
		dataframeCurveLocations.loc[:, 'analyzed'] &= state.locatorGroupAlpha # (groupAlpha)

		# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
		dataframeCurveLocations.loc[(~ImaGroupZulpha), 'analyzed'] = state.datatypeCurveLocations( # pyright: ignore[reportCallIssue, reportArgumentType]
			flipTheExtra_0b1AsUfunc(dataframeCurveLocations.loc[(~ImaGroupZulpha), 'analyzed']))

		del ImaGroupZulpha

# NOTE Above this line, I am only using the current minimum of data structures: i.e., no selectors.

# TODO `selectorGroupAlphaAtEven` until I can figure out how to eliminate it.
		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupAlpha`.
		ImaGroupZulpha &= state.locatorGroupAlpha # Ima `groupAlpha`.
		ImaGroupZulpha &= 1 # (groupAlpha & 1)
		ImaGroupZulpha ^= 1 # (1 - (groupAlpha ...))
		selectorGroupAlphaAtEven: pandas.Series = ImaGroupZulpha.astype(bool)

		del ImaGroupZulpha

		# if groupZuluIsEven and not groupAlphaIsEven, modifyGroupZuluPairedToOdd
		ImaGroupZulpha = dataframeCurveLocations['curveLocations'].copy() # Ima `groupZulu`.
		ImaGroupZulpha &= state.locatorGroupZulu # Ima `groupZulu`.
		ImaGroupZulpha //= 2**1 # Ima `groupZulu` (groupZulu >> 1)

		ImaGroupZulpha.loc[(~selectorGroupAlphaAtEven)] = state.datatypeCurveLocations( # pyright: ignore[reportCallIssue, reportArgumentType]
			flipTheExtra_0b1AsUfunc(ImaGroupZulpha.loc[(~selectorGroupAlphaAtEven)]))

		del selectorGroupAlphaAtEven
		goByeBye()

# NOTE Step 3: compute curveLocations
		dataframeCurveLocations.loc[:, 'analyzed'] //= 2**2 # (groupAlpha >> 2)

		ImaGroupZulpha //= 2**2 # (groupZulu >> 2)
		ImaGroupZulpha *= 2**1 # ((groupZulu ...) << 1)

		dataframeCurveLocations.loc[:, 'analyzed'] |= ImaGroupZulpha # ... | (groupZulu ...)
		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

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
		dataframeCurveLocations.loc[:, 'analyzed'] &= state.locatorGroupAlpha

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
		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

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
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['curveLocations']
		dataframeCurveLocations.loc[:, 'analyzed'] &= state.locatorGroupZulu
		dataframeCurveLocations.loc[:, 'analyzed'] //= 2**1 # (groupZulu >> 1)

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
		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= state.MAXIMUMcurveLocations, 'analyzed'] = 0

		del ImaGroupZulpha

	def outfitDataframeAnalyzed() -> None:
		nonlocal dataframeAnalyzed
		dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(getBucketsTotal(state)), fill_value=0)

	def outfitDataframeCurveLocations() -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations = dataframeCurveLocations.iloc[0:0]
		dataframeCurveLocations['curveLocations'] = dataframeAnalyzed['analyzed']
		dataframeCurveLocations['distinctCrossings'] = dataframeAnalyzed['distinctCrossings']
		dataframeCurveLocations['analyzed'] = 0
		state.bitWidth = int(dataframeCurveLocations['curveLocations'].max()).bit_length()

		dataframeAnalyzed = dataframeAnalyzed.iloc[0:0]

	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=state.dictionaryCurveLocations.keys(), dtype=state.datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=state.dictionaryCurveLocations.values(), dtype=state.datatypeDistinctCrossings)
		}, dtype=state.datatypeCurveLocations
	)
	state.dictionaryCurveLocations.clear()

	dataframeCurveLocations = pandas.DataFrame({
		'curveLocations': pandas.Series(name='curveLocations', data=0, dtype=state.datatypeCurveLocations)
		, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=state.datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=0, dtype=state.datatypeDistinctCrossings)
		}, dtype=state.datatypeCurveLocations # pyright: ignore[reportUnknownArgumentType]
	)

	while (state.kOfMatrix > 0
		and (int(dataframeAnalyzed['analyzed'].max()).bit_length() <= raiseIfNone(state.bitWidthCurveLocationsMaximum))
		and (int(dataframeAnalyzed['distinctCrossings'].max()).bit_length() <= raiseIfNone(state.bitWidthDistinctCrossingsMaximum))):

		outfitDataframeCurveLocations()
		goByeBye()

		outfitDataframeAnalyzed()
		state.indexStartAnalyzed = 0
		goByeBye()

		state.kOfMatrix -= 1

		analyzeCurveLocationsSimple()
		recordCurveLocations()
		goByeBye()
		analyzeCurveLocationsAlpha()
		recordCurveLocations()
		goByeBye()
		analyzeCurveLocationsZulu()
		recordCurveLocations()
		goByeBye()
		analyzeCurveLocationsAligned()
		recordCurveLocations()
		dataframeCurveLocations = dataframeCurveLocations.iloc[0:0]
		goByeBye()

		dataframeAnalyzed = dataframeAnalyzed.iloc[0:state.indexStartAnalyzed].groupby('analyzed', sort=False)['distinctCrossings'].aggregate('sum').reset_index()
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
