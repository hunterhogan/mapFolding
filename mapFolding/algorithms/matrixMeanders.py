"""Count meanders with matrix transfer algorithm."""
from functools import cache
from gc import collect as goByeBye
from tqdm import tqdm
import gc
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

datatypeCurveLocations = numpy.uint64
datatypeDistinctCrossings = numpy.uint64

tqdmDelay: float = 1.9
tqdmDisable: bool = False
tqdmInitial: int = 0
tqdmLeave: bool = False
tqdmTotal: int = 0

# ----------------- support functions ---------------------------------------------------------------------------------

@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

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
	"""NOTE `gc.set_threshold`: Low numbers nullify the `walkDyckPath` cache."""
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

def countBigInt(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int], indexTransferMatrixMinimum: int = 0) -> tuple[int, dict[int, int]]:
	"""Count meanders with matrix transfer algorithm using Python primitive `int` contained in a Python primitive `dict`.

	Parameters
	----------
	indexTransferMatrix : int
		The current index in the transfer matrix algorithm.
	dictionaryCurveLocations : dict[int, int]
		A dictionary of `curveLocations` to `distinctCrossings`.
	indexTransferMatrixMinimum : int = 0
		The last index value to compute, even if the full algorithm computation is incomplete.

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

	gc.disable()  # Disable the garbage collector inside this loop to maximize the `walkDyckPath` cache hits.

	for indexTransferMatrix in tqdm(list(range(indexTransferMatrix -1, indexTransferMatrixMinimum -1, -1)), total=tqdmTotal, initial=tqdmInitial, leave=tqdmLeave, delay=tqdmDelay, disable=tqdmDisable):  # noqa: B020, PLR1704
		if (indexTransferMatrixMinimum > 0) and (max(dictionaryCurveLocations.keys()).bit_length() <= bitWidthCurveLocationsMaximum):
			indexTransferMatrix += 1  # noqa: PLW2901
			sys.stdout.write(f"Switching at {indexTransferMatrix =}, not {indexTransferMatrixMinimum =}.")
			break
		MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(indexTransferMatrix)
		dictionaryCurveGroups = outfitDictionaryCurveGroups(dictionaryCurveLocations)
		dictionaryCurveLocations = {}

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

	gc.enable()  # Re-enable the garbage collector.

	return (indexTransferMatrix, dictionaryCurveLocations)

def countPandas(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int], indexTransferMatrixMinimum: int = 0) -> tuple[int, dict[int, int]]:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame.

	indexTransferMatrix : int
		The current index in the transfer matrix algorithm.
	dictionaryCurveLocations : dict[int, int]
		A dictionary of `curveLocations` to `distinctCrossings`.
	indexTransferMatrixMinimum : int = 0
		The last index value to compute, even if the full algorithm computation is incomplete.

	Returns
	-------
	matrixMeandersState : tuple[int, dict[int, int]]
		The state of the algorithm computation: the current `indexTransferMatrix`, `curveLocations`, and `distinctCrossings`.
	"""
	def aggregateCurveLocations(MAXIMUMcurveLocations: int) -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = 0
		dataframeAnalyzed = pandas.concat(
			[dataframeAnalyzed
			, dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] > 0), ['analyzed', 'distinctCrossings']].reset_index()
		], ignore_index=True).groupby('analyzed', sort=False)['distinctCrossings'].aggregate('sum').reset_index()

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

		Parameters
		----------
		MAXIMUMcurveLocations : int
			Maximum value of `curveLocations` for the current iteration of `bridges`.
		"""
		nonlocal dataframeCurveLocations

		# if groupAlphaIsEven or groupZuluIsEven
		# NOTE Options. The CPU cost is the same. Pay to drop. Pay for a mask. Pay to evaluate and set to 0. I assume memory is cheaper to drop.
		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations[dataframeCurveLocations['alignAt'] == 'oddBoth'].index)

		# if groupAlphaCurves and groupZuluCurves
		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations[(dataframeCurveLocations['groupAlpha'] <= 1) | (dataframeCurveLocations['groupZulu'] <= 1)].index)

		# if groupAlphaIsEven and not groupZuluIsEven, modifyGroupAlphaPairedToOdd
		dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenAlpha', 'groupAlpha'] = datatypeCurveLocations(flipTheExtra_0b1AsUfunc(dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenAlpha', 'groupAlpha'])) # 32s

		# if groupZuluIsEven and not groupAlphaIsEven, modifyGroupZuluPairedToOdd
		dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenZulu', 'groupZulu'] = datatypeCurveLocations(flipTheExtra_0b1AsUfunc(dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenZulu', 'groupZulu'])) # 31s

		dataframeCurveLocations.loc[:, 'groupAlpha'] //= 2**2 # (groupAlpha >> 2)
		dataframeCurveLocations.loc[:, 'groupZulu'] //= 2**2 # (groupZulu >> 2)
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**1 # ((groupZulu ...) << 1)
		dataframeCurveLocations.loc[:, 'analyzed'] = dataframeCurveLocations['groupAlpha']
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # (groupZulu ...) | (groupAlpha ...)

		aggregateCurveLocations(MAXIMUMcurveLocations) # 106

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

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] == 1, 'alignAt'] = 'evenAlpha' # groupAlphaIsEven

		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**1 # ((groupAlpha ...) << 1)
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**3 # (groupZulu << 3)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # ... | (groupZulu ...)
# NOTE potential memory optimization
		dataframeCurveLocations.loc[:, 'analyzed'] |= (dataframeCurveLocations['groupAlpha'] // 2**2) # ... | (groupAlpha >> 2)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupAlpha'] <= 1, 'analyzed'] = 0 # if groupAlpha > 1

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
		About substituting `+= 3` for `|= 3`:

		- Givens
			1. "n" is a Python `int` >= 0
			2. "0bk" = `bin(n)`

		- Claims
			1. n * 2**2 == n << 2
			2. bin(n * 2**2) == 0bk00
			3. 0b11 = 0b00 | 0b11
			4. 0bk11 = 0bk00 | 0b11
			5. 0b11 = 0bk11 - 0bk00
			6. 0b11 == int(3)

		- Therefore
			- For any non-zero integer, 0bk00, the operation 0bk00 | 0b11 is equivalent to 0bk00 + 0b11.
			- I hope my substitution is valid!

		Why substitute? I've been having problems implementing bitwise operations in pandas, so I am avoiding them until I learn
		how to implement them in pandas.
		"""
		nonlocal dataframeCurveLocations
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['groupAlpha']
		dataframeCurveLocations.loc[:, 'groupZulu'] *= 2**1 # (groupZulu << 1)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupZulu'] # ((groupAlpha | (groupZulu ...))
		dataframeCurveLocations.loc[:, 'analyzed'] *= 2**2 # (... << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] += 3 # (...) | 3

		aggregateCurveLocations(MAXIMUMcurveLocations) # 34
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

		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1) & (dataframeCurveLocations['alignAt'] == 'evenAlpha'), 'alignAt'] = 'evenBoth' # groupZuluIsEven
		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1) & (dataframeCurveLocations['alignAt'] == 'oddBoth'), 'alignAt'] = 'evenZulu' # groupZuluIsEven

		dataframeCurveLocations.loc[:, 'groupAlpha'] *= 2**2 # (groupAlpha << 2)
		dataframeCurveLocations.loc[:, 'analyzed'] |= dataframeCurveLocations['groupAlpha'] # ... | (groupAlpha ...)
# NOTE potential memory optimization
		dataframeCurveLocations.loc[:, 'analyzed'] |= (dataframeCurveLocations['groupZulu'] // 2**1) # ... | (groupZulu >> 1)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupZulu'] <= 1, 'analyzed'] = 0 # if groupZulu > 1

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

	def outfitDataframeCurveLocations() -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations = dataframeCurveLocations.iloc[0:0]
		dataframeCurveLocations['curveLocations'] = dataframeAnalyzed['analyzed']
		dataframeCurveLocations['distinctCrossings'] = dataframeAnalyzed['distinctCrossings']
		dataframeCurveLocations['alignAt'] = 'oddBoth'
		dataframeCurveLocations['analyzed'] = 0
		dataframeAnalyzed = dataframeAnalyzed.iloc[0:0]
		computeCurveGroups()

	CategoriesAlignAt = pandas.CategoricalDtype(categories=['evenAlpha', 'evenZulu', 'evenBoth', 'oddBoth'], ordered=False)

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

	for indexTransferMatrix in tqdm(list(range(indexTransferMatrix -1, indexTransferMatrixMinimum -1, -1)), total=tqdmTotal, initial=tqdmInitial, leave=tqdmLeave, delay=tqdmDelay, disable=tqdmDisable):  # noqa: B020, PLR1704
		if int(dataframeAnalyzed['distinctCrossings'].max()).bit_length() > bitWidthDistinctCrossingsMaximum:
			indexTransferMatrix += 1  # noqa: PLW2901
			sys.stdout.write(f"Switching at {indexTransferMatrix =}, not {indexTransferMatrixMinimum =}.")
			break
		MAXIMUMcurveLocations: int = getMAXIMUMcurveLocations(indexTransferMatrix)
		outfitDataframeCurveLocations()
		goByeBye()

		analyzeCurveLocationsSimple(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsAlpha(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsZulu(MAXIMUMcurveLocations)
		goByeBye()
		analyzeCurveLocationsAligned(MAXIMUMcurveLocations)
		goByeBye()

	return (indexTransferMatrix, dataframeAnalyzed.set_index('analyzed')['distinctCrossings'].to_dict())

# ----------------- doTheNeedful --------------------------------------------------------------------------------------

def doTheNeedful(indexTransferMatrix: int, dictionaryCurveLocations: dict[int, int]) -> int:
	"""Compute a(n) meanders with the transfer matrix algorithm.

	Parameters
	----------
	n : int
		The index in the OEIS ID sequence.
	dictionaryCurveLocations : dict[int, int]
		A dictionary mapping curve locations to their counts.

	Returns
	-------
	a(n) : int
		The computed value of a(n).

	Making sausage
	--------------

	As first computed by Iwan Jensen in 2000, A000682(41) = 6664356253639465480.
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex
	See also https://oeis.org/A000682

	I'm sure you instantly observed that A000682(41) = (6664356253639465480).bit_length() = 63 bits. And A005316(44) =
	(18276178714484582264).bit_length() = 64 bits.

	If you ask NumPy 2.3, "What is your relationship with integers with more than 64 bits?"
	NumPy will say, "It's complicated."

	Therefore, to take advantage of the computational excellence of NumPy when computing A000682(n) for n > 41, I must make some
	adjustments at the total count approaches 64 bits.

	The second complication is bit-packed integers. I use a loop that starts at `bridges = n` and decrements (`bridges -= 1`)
	`until bridges = 0`. If `bridges > 29`, some of the bit-packed integers have more than 64 bits. "Hey NumPy, can I use
	bit-packed integers with more than 64 bits?" NumPy: "It's complicated." Therefore, while `bridges` is decrementing, I don't
	use NumPy until I believe the bit-packed integers will be less than 64 bits.

	A third factor that works in my favor is that peak memory usage occurs when all types of integers are well under 64-bits wide.

	In total, to compute a(n) for "large" n, I use three-stages.
	1. I use Python primitive `int` contained in a Python primitive `dict`.
	2. When the bit width of the bit-packed integers connected to `bridges` is small enough to use `numpy.uint64`, I switch to NumPy for the heavy lifting.
	3. When `distinctCrossings` subtotals might exceed 64 bits, I must switch back to Python primitives.
	"""
	global tqdmInitial, tqdmTotal  # noqa: PLW0603
	tqdmTotal = indexTransferMatrix
	tqdmInitial = tqdmTotal - indexTransferMatrix
	indexTransferMatrixMinimum = max(0, (indexTransferMatrix + 1 - 41 + 2))

	bitWidth: int = max(dictionaryCurveLocations.keys()).bit_length()
	# NOTE Stage 1 if `curveLocations` bit-width is too wide for `numpy.uint64`.
	if bitWidth > bitWidthCurveLocationsMaximum:
		indexTransferMatrixMinimumEstimatedA000682 = 28
		indexTransferMatrix, dictionaryCurveLocations = countBigInt(indexTransferMatrix, dictionaryCurveLocations, indexTransferMatrixMinimumEstimatedA000682)
		tqdmInitial = tqdmTotal - indexTransferMatrix
		goByeBye()

	# NOTE Stage 2 Goldilocks
	indexTransferMatrix, dictionaryCurveLocations = countPandas(indexTransferMatrix, dictionaryCurveLocations, indexTransferMatrixMinimum)

	# NOTE Stage 3 if `distinctCrossings` bit-width is too wide for `numpy.uint64`.
	if indexTransferMatrix > 0:
		tqdmInitial = tqdmTotal - indexTransferMatrix
		goByeBye()
		indexTransferMatrix, dictionaryCurveLocations = countBigInt(indexTransferMatrix, dictionaryCurveLocations)
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
