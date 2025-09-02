"""Count meanders with matrix transfer algorithm using pandas DataFrame."""
from functools import cache
import numpy
import pandas

datatypeCurveLocationsHARDCODED = numpy.uint64
datatypeDistinctCrossingsHARDCODED = numpy.uint64

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

flipTheExtra_0b1 = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def count(bridges: int, dictionaryCurveLocations: dict[int, int]) -> int:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame."""
	def computeCurveGroups() -> None:
		nonlocal dataframeCurveLocations
		bitWidth: int = int(dataframeCurveLocations['curveLocations'].max()).bit_length()
		locatorGroupAlpha: int = sum(1 << one for one in range(0, bitWidth, 2))
		locatorGroupZulu: int = sum(1 << one for one in range(1, bitWidth, 2))

		dataframeCurveLocations['groupAlpha'] = dataframeCurveLocations['curveLocations'] & locatorGroupAlpha
		dataframeCurveLocations['groupZulu'] = dataframeCurveLocations['curveLocations'] & locatorGroupZulu
		dataframeCurveLocations['groupZulu'] //= 2**1

	def outfitDataframeCurveLocations() -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations['curveLocations'] = dataframeAnalyzed['analyzed'].copy()
		dataframeCurveLocations['distinctCrossings'] = dataframeAnalyzed['distinctCrossings'].copy()
		dataframeCurveLocations = dataframeCurveLocations.drop_duplicates(['curveLocations', 'distinctCrossings'])
		dataframeCurveLocations['alignAt'] = 'oddBoth'
		dataframeAnalyzed['analyzed'] = 0
		dataframeAnalyzed['distinctCrossings'] = 0
		dataframeAnalyzed = dataframeAnalyzed.drop_duplicates()
		computeCurveGroups()

	def analyzeCurveLocations(MAXIMUMcurveLocations: int) -> None:
		nonlocal dataframeAnalyzed, dataframeCurveLocations
		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = 0

		dataframeAnalyzed = dataframeAnalyzed.set_index('analyzed')['distinctCrossings'].add(
			dataframeCurveLocations.groupby('analyzed')['distinctCrossings'].sum()
		).reset_index().astype({'analyzed': datatypeCurveLocations, 'distinctCrossings': datatypeDistinctCrossings})

		dataframeAnalyzed = dataframeAnalyzed.drop(dataframeAnalyzed[dataframeAnalyzed['analyzed'] == 0].index)

		dataframeCurveLocations['analyzed'] = 0

	datatypeCurveLocations = datatypeCurveLocationsHARDCODED
	datatypeDistinctCrossings = datatypeDistinctCrossingsHARDCODED

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

	for countdown in range((bridges - 1), (0 - 1), -1):
		MAXIMUMcurveLocations: int = 1 << (2 * countdown + 4)
		outfitDataframeCurveLocations()

# ------------------------------------- curveLocationsSimple ----------------------------------------------------------
# NOTE ((groupAlpha | (groupZulu << 1)) << 2) | 3
		dataframeCurveLocations['groupZulu'] *= 2**1
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['groupAlpha'] | dataframeCurveLocations['groupZulu']
		dataframeCurveLocations['analyzed'] *= 2**2
		dataframeCurveLocations['analyzed'] += 0b11

		analyzeCurveLocations(MAXIMUMcurveLocations)

		computeCurveGroups()

# ------------------------------------- curveLocationsAlpha -----------------------------------------------------------
# NOTE ((1 - (groupAlpha & 1)) << 1) | (groupZulu << 3) | (groupAlpha >> 2)
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['groupAlpha'] & 1 # (groupAlpha & 1)
		dataframeCurveLocations['analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupAlpha ...))

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] == 1, 'alignAt'] = 'evenAlpha' # groupAlphaIsEven

		dataframeCurveLocations['analyzed'] *= 2**1 # ((groupAlpha ...) << 1)
		dataframeCurveLocations['groupZulu'] *= 2**3 # (groupZulu << 3)
		dataframeCurveLocations['analyzed'] |= dataframeCurveLocations['groupZulu'] # ... | (groupZulu ...)
		dataframeCurveLocations['analyzed'] |= (dataframeCurveLocations['groupAlpha'] // 2**2) # ... | (groupAlpha >> 2)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupAlpha'] <= 1, 'analyzed'] = 0 # if groupAlpha > 1

		analyzeCurveLocations(MAXIMUMcurveLocations)

		computeCurveGroups()

# ------------------------------------- curveLocationsZulu ------------------------------------------------------------
# NOTE (1 - (groupZulu & 1)) | (groupAlpha << 2) | (groupZulu >> 1)
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['groupZulu'] & 1 # (groupZulu & 1)
		dataframeCurveLocations['analyzed'] = 1 - dataframeCurveLocations['analyzed'] # (1 - (groupZulu ...))

		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1) & (dataframeCurveLocations['alignAt'] == 'evenAlpha'), 'alignAt'] = 'evenBoth' # groupZuluIsEven
		dataframeCurveLocations.loc[(dataframeCurveLocations['analyzed'] == 1) & (dataframeCurveLocations['alignAt'] == 'oddBoth'), 'alignAt'] = 'evenZulu' # groupZuluIsEven

		dataframeCurveLocations['groupAlpha'] *= 2**2 # (groupAlpha << 2)
		dataframeCurveLocations['analyzed'] |= dataframeCurveLocations['groupAlpha'] # ... | (groupAlpha ...)
		dataframeCurveLocations['analyzed'] |= (dataframeCurveLocations['groupZulu'] // 2**1) # ... | (groupZulu >> 1)
		dataframeCurveLocations.loc[dataframeCurveLocations['groupZulu'] <= 1, 'analyzed'] = 0 # if groupZulu > 1

		analyzeCurveLocations(MAXIMUMcurveLocations)

		computeCurveGroups()

# ------------------------------------- aligning ----------------------------------------------------------------------
		# if groupAlphaCurves and groupZuluCurves
		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations[(dataframeCurveLocations['groupAlpha'] <= 1) & (dataframeCurveLocations['groupZulu'] <= 1)].index)

		# if groupAlphaIsEven or groupZuluIsEven
		dataframeCurveLocations = dataframeCurveLocations.drop(dataframeCurveLocations[dataframeCurveLocations['alignAt'] == 'oddBoth'].index)

# ------------------------------------- modifyGroupAlphaPairedToOdd ---------------------------------------------------
		# if groupAlphaIsEven and not groupZuluIsEven
		dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenAlpha', 'groupAlpha'] = datatypeCurveLocations(flipTheExtra_0b1(dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenAlpha', 'groupAlpha']))

# ------------------------------------- modifyGroupZuluPairedToOdd ----------------------------------------------------
		# if groupZuluIsEven and not groupAlphaIsEven
		dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenZulu', 'groupZulu'] = datatypeCurveLocations(flipTheExtra_0b1(dataframeCurveLocations.loc[dataframeCurveLocations['alignAt'] == 'evenZulu', 'groupZulu']))

# ------------------------------------- curveLocationsAligned ---------------------------------------------------------
# NOTE (groupAlpha >> 2) | ((groupZulu >> 2) << 1)
		dataframeCurveLocations['groupAlpha'] //= 2**2 # (groupAlpha >> 2)
		dataframeCurveLocations['groupZulu'] //= 2**2 # (groupZulu >> 2)
		dataframeCurveLocations['groupZulu'] *= 2**1 # ((groupZulu ...) << 1)
		dataframeCurveLocations['analyzed'] = dataframeCurveLocations['groupAlpha'] | dataframeCurveLocations['groupZulu'] # (groupZulu ...) | (groupAlpha ...)

		analyzeCurveLocations(MAXIMUMcurveLocations)

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
