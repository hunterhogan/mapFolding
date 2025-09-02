# ruff: noqa
# ruff: noqa: D100 D103 ERA001 F841
"""
You should never modify something you are iterating over. This is not guaranteed to work in all cases.

See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html
"""
from functools import cache
import numpy
import pandas

datatypeCurveLocationsHARDCODED = numpy.uint64
datatypeDistinctCrossingsHARDCODED = numpy.uint64

locator64GroupAlpha = numpy.uint64(0x5555555555555555)
locator64GroupZulu = numpy.uint64(0xaaaaaaaaaaaaaaaa)

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
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
	return intWithExtra_0b1 ^ walkDyckPath(intWithExtra_0b1)

flipTheExtra_0b1 = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)

def count(bridges: int, dictionaryCurveLocations: dict[int, int]) -> int:

	def computeCurveGroups() -> None:
		dataframeCurveLocations['groupAlpha'] = dataframeCurveLocations['curveLocations'] & locator64GroupAlpha
		dataframeCurveLocations['groupZulu'] = dataframeCurveLocations['curveLocations'] & locator64GroupZulu
		dataframeCurveLocations['groupZulu'] //= 2**1

	def outfitDataframeCurveLocations() -> None:
		dataframeCurveLocations['curveLocations'] = dataframeAnalyzed['analyzed']
		dataframeCurveLocations['distinctCrossings'] = dataframeAnalyzed['distinctCrossings']
		dataframeCurveLocations['alignAt'] = 'oddBoth'
		dataframeAnalyzed['analyzed'] = None
		dataframeAnalyzed['distinctCrossings'] = None
		computeCurveGroups()

	datatypeCurveLocations = datatypeCurveLocationsHARDCODED
	datatypeDistinctCrossings = datatypeDistinctCrossingsHARDCODED

	CategoriesAlignAt = pandas.CategoricalDtype(categories=['evenAlpha', 'evenZulu', 'evenBoth', 'oddBoth'], ordered=False)

	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=list(dictionaryCurveLocations.keys()), dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=list(dictionaryCurveLocations.values()), dtype=datatypeDistinctCrossings)
		}
	)

	dataframeCurveLocations = pandas.DataFrame({
		'curveLocations': pandas.Series(name='curveLocations', data=None, dtype=datatypeCurveLocations)
		, 'groupAlpha': pandas.Series(name='groupAlpha', data=None, dtype=datatypeCurveLocations)
		, 'groupZulu': pandas.Series(name='groupZulu', data=None, dtype=datatypeCurveLocations)
		, 'analyzed': pandas.Series(name='analyzed', data=None, dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=None, dtype=datatypeDistinctCrossings)
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

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = None
		dataframeAnalyzed = dataframeCurveLocations.groupby('analyzed')['distinctCrossings'].sum().reset_index()

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
		dataframeCurveLocations.loc[dataframeCurveLocations['groupAlpha'] <= 1, 'analyzed'] = None # if groupAlpha > 1

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = None
		dataframeAnalyzed = dataframeCurveLocations.groupby('analyzed')['distinctCrossings'].sum().reset_index()

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
		dataframeCurveLocations.loc[dataframeCurveLocations['groupZulu'] <= 1, 'analyzed'] = None # if groupZulu > 1

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = None
		dataframeAnalyzed = dataframeCurveLocations.groupby('analyzed')['distinctCrossings'].sum().reset_index()

		computeCurveGroups()

# ------------------------------------- aligning ----------------------------------------------------------------------
		# if groupAlphaCurves and groupZuluCurves
		dataframeCurveLocations.drop(dataframeCurveLocations[(dataframeCurveLocations['groupAlpha'] <= 1) & (dataframeCurveLocations['groupZulu'] <= 1)].index, inplace=True)

		# if (groupAlphaIsEven or groupZuluIsEven)
		dataframeCurveLocations.drop(dataframeCurveLocations[dataframeCurveLocations['alignAt'] == 'oddBoth'].index, inplace=True)

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

		dataframeCurveLocations.loc[dataframeCurveLocations['analyzed'] >= MAXIMUMcurveLocations, 'analyzed'] = None
		dataframeAnalyzed = dataframeCurveLocations.groupby('analyzed')['distinctCrossings'].sum().reset_index()

	return int(dataframeAnalyzed['distinctCrossings'].sum())

def doTheNeedful(bridges: int, dictionaryCurveLocations: dict[int, int]) -> int:
	return count(bridges, dictionaryCurveLocations)

def A000682getCurveLocations(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)
	curveStart: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveStart << 1) | curveStart]
	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveStart = (curveStart << 4) | 0b101
		listCurveLocations.append((curveStart << 1) | curveStart)
	return dict.fromkeys(listCurveLocations, 1)

@cache
def A000682(n: int) -> int:
	return doTheNeedful(n - 1, A000682getCurveLocations(n - 1))

def A005316getCurveLocations(n: int) -> dict[int, int]:
	if n & 0b1:
		return {22: 1}
	else:
		return {15: 1}

@cache
def A005316(n: int) -> int:
	return doTheNeedful(n - 1, A005316getCurveLocations(n - 1))
