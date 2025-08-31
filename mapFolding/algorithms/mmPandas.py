# ruff: noqa
# ruff: noqa: D100 D103 ERA001 F841

from functools import cache
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

	def computeCurveGroups() -> None:
		if datatypeCurveLocations == 'UInt64':
			bitwiseAlpha = locator64GroupAlpha
			bitwiseZulu = locator64GroupZulu
		else:
			bitwiseAlpha = locatorGroupAlpha
			bitwiseZulu = locatorGroupZulu

		numpy.bitwise_and(theDataframe['curveLocations'], bitwiseAlpha, out=theDataframe['groupAlpha']) # pyright: ignore[reportCallIssue, reportArgumentType]
		numpy.bitwise_and(theDataframe['curveLocations'], bitwiseZulu, out=theDataframe['groupZulu']) # pyright: ignore[reportCallIssue, reportArgumentType]
		numpy.right_shift(theDataframe['groupZulu'], 1, out=theDataframe['groupZulu']) # pyright: ignore[reportCallIssue, reportArgumentType]
		theDataframe['curveLocations'] = None

	datatypeCurveLocations = datatypeCurveLocationsHARDCODED
	datatypeDistinctCrossings = datatypeDistinctCrossingsHARDCODED
	datatypeDataFrame = datatypeCurveLocations if datatypeCurveLocations == datatypeDistinctCrossings else object

	theDataframe = pandas.DataFrame({
		'curveLocations': pandas.Series(dictionaryCurveLocations.keys(), dtype=datatypeCurveLocations)
		, 'distinctCrossings': pandas.Series(dictionaryCurveLocations.values(), dtype=datatypeDistinctCrossings)
		, 'groupAlpha': pandas.Series(data=None, dtype=datatypeCurveLocations)
		, 'groupZulu': pandas.Series(data=None, dtype=datatypeCurveLocations)
		}, dtype=datatypeDataFrame
	)

# TODO Add call to groupby

	computeCurveGroups()

	# memory usage is extremely important
	# The loop
		# Controlled by decrementing bridges to 0
		# do the first three conditional computations on each row: curveLocationsSimple, curveLocationsAlpha, and curveLocationsZulu
			# some of the computations produce a new curveLocations associated with that row
			# each row starts with an _empty_ curveLocations and a value in distinctCrossings
			# if the computation produces a new curveLocations and the curveLocations column is still empty, put it in the empty
				# curveLocations: the distinctCrossings is exactly what we need associated with that curveLocations
			# if the computation produces a new curveLocations and the curveLocations column is _not_ empty, create a new row: put
				# the new curveLocations in curveLocations column and copy the value in distinctCrossings to the new row
		# The last computation, curveLocationsAligned, modifies some values in columns groupAlpha and groupZulu, so it must be last
			# As before, fill an empty curveLocations or create a new row
		# groupby curveLocations, sum distinctCrossings, keep one row per curveLocations value
		# compute groupAlpha from curveLocations
		# compute groupZulu from curveLocations
		# curveLocations = None
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
