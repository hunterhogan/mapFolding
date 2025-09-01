# ruff: noqa
# ruff: noqa: D100 D103 ERA001 F841

from functools import cache
from numpy import e
import numpy
import pandas

locatorGroupAlpha: int = 0x55555555555555555555555555555555
locator64GroupAlpha: int = 0x5555555555555555
locatorGroupZulu: int = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
locator64GroupZulu: int = 0xaaaaaaaaaaaaaaaa

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

def count(bridges: int, dictionaryCurveLocations: dict[int, int]) -> int: # pyright: ignore[reportReturnType]
	datatypeCurveLocationsHARDCODED = 'UInt64'
	datatypeDistinctCrossingsHARDCODED = 'UInt64'

	def aggregateCurveLocations() -> None:
# NOTE You know this doesn't cause an exception, but you don't know that it works.
		# zero-out groupAlpha and groupZulu, groupby curveLocations, sum distinctCrossings, zero-out duplicate rows
		theDataframe['groupAlpha'] = pandas.NA
		theDataframe['groupZulu'] = pandas.NA
		theDataframe['distinctCrossings'].update(theDataframe.groupby(by='curveLocations', as_index=False, sort=False, group_keys=False)['distinctCrossings'].transform(func='sum'))
		theDataframe[theDataframe.duplicated()] = pandas.NA

	def computeCurveGroups() -> None:
		if datatypeCurveLocations == 'UInt64':
			bitwiseAlpha = locator64GroupAlpha
			bitwiseZulu = locator64GroupZulu
		else:
			bitwiseAlpha = locatorGroupAlpha
			bitwiseZulu = locatorGroupZulu

		numpy.bitwise_and(theDataframe['curveLocations'], bitwiseAlpha, out=theDataframe['groupAlpha']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.bitwise_and(theDataframe['curveLocations'], bitwiseZulu, out=theDataframe['groupZulu']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.right_shift(theDataframe['groupZulu'], 1, out=theDataframe['groupZulu']) # pyright: ignore[reportArgumentType, reportCallIssue]
		theDataframe['curveLocations'] = pandas.NA

	rowsEstimated = 0.71 * e**(0.4668 * bridges+1)

	datatypeCurveLocations = datatypeCurveLocationsHARDCODED
	datatypeDistinctCrossings = datatypeDistinctCrossingsHARDCODED
	datatypeDataFrame = datatypeCurveLocations if datatypeCurveLocations == datatypeDistinctCrossings else object

	theDataframe = pandas.DataFrame({
		'curveLocations': pandas.Series(name='curveLocations', data=dictionaryCurveLocations.keys(), dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(name='distinctCrossings', data=dictionaryCurveLocations.values(), dtype=datatypeDistinctCrossings)
		, 'groupAlpha': pandas.Series(name='groupAlpha', data=pandas.NA, dtype=datatypeCurveLocations)
		, 'groupZulu': pandas.Series(name='groupZulu', data=pandas.NA, dtype=datatypeCurveLocations)
		}, dtype=datatypeDataFrame
	)

	aggregateCurveLocations()

	computeCurveGroups()

	for Z0Z_bridges in range((bridges - 1), (0 - 1), -1):
		MAXIMUMcurveLocations = 1 << (2 * Z0Z_bridges + 4)

# ------------------------------------- curveLocationsSimple ----------------------------------------------------------
		numpy.left_shift(theDataframe['groupZulu'], 1, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.bitwise_or(theDataframe['groupAlpha'], theDataframe['curveLocations'], out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.left_shift(theDataframe['curveLocations'], 2, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.bitwise_or(theDataframe['curveLocations'], 3, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		theDataframe.loc[theDataframe['curveLocations'] >= MAXIMUMcurveLocations, 'curveLocations'] = pandas.NA

# ------------------------------------- curveLocationsAlpha -----------------------------------------------------------
# TODO NOTE I'm writing this logic as if `theDataframe['curveLocations']` were still empty.
		theDataframe['curveLocations'].update(theDataframe['groupZulu'])
# TODO I read something in the pandas docs that makes me think pandas will simply skip NaN. If true, then setting to NaN is the equivalent of mask=False.
		theDataframe.loc[theDataframe['curveLocations'] <= 1, 'curveLocations'] = pandas.NA # = 0
		numpy.left_shift(theDataframe['curveLocations'], 3, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		# (groupZulu << 3)
		# Right now, the THREE least significant bits are 000. So, if I can confine operations to only those three bits, I can do anything I want.






		numpy.bitwise_and(theDataframe['curveLocations'], 1, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.subtract(1, theDataframe['curveLocations'], out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]

		numpy.left_shift(theDataframe['curveLocations'], 2, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.bitwise_or(theDataframe['curveLocations'], theDataframe['groupAlpha'], out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.right_shift(theDataframe['curveLocations'], 2, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]


		"""
		if groupAlpha > 1:
			curveLocations = (groupAlpha >> 2) | (groupZulu << 3) | ((1 - (groupAlpha & 1)) << 1)

		((1 - (groupAlpha & 1)) << 1): maybe flip the second-least-significant bit

		selectCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupAlpha] > 1
		curveLocations: DataArray1D = arrayCurveGroups[selectCurves, columnGroupAlpha].copy()
		numpy.right_shift(curveLocations, 2, out=curveLocations)

		numpy.bitwise_or(curveLocations, numpy.left_shift(arrayCurveGroups[selectCurves, columnGroupZulu], 3), out=curveLocations)

		numpy.bitwise_or(curveLocations, numpy.left_shift(numpy.subtract(1, numpy.bitwise_and(arrayCurveGroups[selectCurves, columnGroupAlpha], 1)), 1), out=curveLocations)

		"""

# ------------------------------------- curveLocationsZulu ------------------------------------------------------------
# TODO NOTE I'm writing this logic as if `theDataframe['curveLocations']` were still empty.
		theDataframe['curveLocations'].update(theDataframe['groupAlpha'])
# TODO I read something in the pandas docs that makes me think pandas will simply skip NaN. If true, then setting to NaN is the equivalent of mask=False.
		theDataframe.loc[theDataframe['curveLocations'] <= 1, 'curveLocations'] = pandas.NA # = 0
		numpy.left_shift(theDataframe['curveLocations'], 2, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		# (groupAlpha << 2)


		numpy.bitwise_and(theDataframe['curveLocations'], 1, out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]
		numpy.subtract(1, theDataframe['curveLocations'], out=theDataframe['curveLocations']) # pyright: ignore[reportArgumentType, reportCallIssue]


		# curveLocations = (groupAlpha << 2) | (groupZulu >> 1) | (1 - (groupZulu & 1))
		#  Any x | right shift, `x | (groupZulu >> 1)`, can be restructured, `((x << 1) | groupZulu) >> 1`, so that the order of operations allows `out=x` at every step.
		# (and is the same as `groupZulu // 2`)
		# For left shift, I've only figured out a method that uses left-shift-many bits of temporary storage, which is better than 64 bits but it's not 0.
		# `| (1 - (groupZulu & 1))` merely means "maybe flip the least significant bit", so there might be another way to compute it without extra memory.



	# The loop
		# do the first three conditional computations on each row: curveLocationsAlpha, and curveLocationsZulu
			# some of the computations produce a new curveLocations associated with that row
			# each row starts with an _empty_ curveLocations and a value in distinctCrossings
			# if the computation produces a new curveLocations and the curveLocations column is still empty, put it in the empty
				# curveLocations: the distinctCrossings is exactly what we need associated with that curveLocations
			# if the computation produces a new curveLocations and the curveLocations column is _not_ empty, create a new row: put
				# the new curveLocations in curveLocations column and copy the value in distinctCrossings to the new row
		# The last computation, curveLocationsAligned, modifies some values in columns groupAlpha and groupZulu, so it must be last
			# As before, fill an empty curveLocations or create a new row
		# groupby curveLocations, sum distinctCrossings, keep one row per curveLocations value
		# `computeCurveGroups()`
# TODO I assume it will be impossible to avoid ALL temporary arrays, so what should I do to minimize the negative effects? How do
# I, for example, ensure the memory is freed as soon as possible?
	# Finish
		# When bridges = 0, after groupby, there will only be one row
		# return distinctCrossings

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
