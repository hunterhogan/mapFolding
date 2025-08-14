"""Semi-meanders.

n = 3   `startingCurveLocations` keys = 3
n = 4   `startingCurveLocations` keys = 4
n = 5   `startingCurveLocations` keys = 4
n = 6   `startingCurveLocations` keys = 5
n = 7   `startingCurveLocations` keys = 5
n = 8   `startingCurveLocations` keys = 6
n = 9   `startingCurveLocations` keys = 6
n = 10  `startingCurveLocations` keys = 7
n = 11  `startingCurveLocations` keys = 7
n = 12  `startingCurveLocations` keys = 8
n = 13  `startingCurveLocations` keys = 8
n = 14  `startingCurveLocations` keys = 9
n = 15  `startingCurveLocations` keys = 9
n = 16  `startingCurveLocations` keys = 10
n = 17  `startingCurveLocations` keys = 10
n = 18  `startingCurveLocations` keys = 11
n = 19  `startingCurveLocations` keys = 11
n = 20  `startingCurveLocations` keys = 12
n = 21  `startingCurveLocations` keys = 12
n = 22  `startingCurveLocations` keys = 13
n = 23  `startingCurveLocations` keys = 13
n = 24  `startingCurveLocations` keys = 14
n = 25  `startingCurveLocations` keys = 14
n = 26  `startingCurveLocations` keys = 15
n = 27  `startingCurveLocations` keys = 15
n = 28  `startingCurveLocations` keys = 16
n = 29  `startingCurveLocations` keys = 16
n = 30  `startingCurveLocations` keys = 17
n = 31  `startingCurveLocations` keys = 17
n = 32  `startingCurveLocations` keys = 18
n = 33  `startingCurveLocations` keys = 18
n = 34  `startingCurveLocations` keys = 19
n = 35  `startingCurveLocations` keys = 19
n = 36  `startingCurveLocations` keys = 20
n = 37  `startingCurveLocations` keys = 20
n = 38  `startingCurveLocations` keys = 21
n = 39  `startingCurveLocations` keys = 21
n = 40  `startingCurveLocations` keys = 22
n = 41  `startingCurveLocations` keys = 22
n = 42  `startingCurveLocations` keys = 23
n = 43  `startingCurveLocations` keys = 23
n = 44  `startingCurveLocations` keys = 24
n = 45  `startingCurveLocations` keys = 24
n = 46  `startingCurveLocations` keys = 25
n = 47  `startingCurveLocations` keys = 25
n = 48  `startingCurveLocations` keys = 26
n = 49  `startingCurveLocations` keys = 26
n = 50  `startingCurveLocations` keys = 27
n = 51  `startingCurveLocations` keys = 27
n = 52  `startingCurveLocations` keys = 28
n = 53  `startingCurveLocations` keys = 28
n = 54  `startingCurveLocations` keys = 29
n = 55  `startingCurveLocations` keys = 29
n = 56  `startingCurveLocations` keys = 30
n = 57  `startingCurveLocations` keys = 30
n = 58  `startingCurveLocations` keys = 31
n = 59  `startingCurveLocations` keys = 31
n = 60  `startingCurveLocations` keys = 32
n = 61  `startingCurveLocations` keys = 32
"""

# TODO figure out how to call the correct module
# In other situations, I use a so-called dispatcher amd that has helped make code transformation easier, too.
from mapFolding._oeisFormulas.matrixMeanders import count  # noqa: ERA001
from mapFolding._oeisFormulas.matrixMeanders64 import count as count64

def initializeA000682(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM = 1 << (2 * n + 4)

	curveSeed: int = 5 - (n & 0b1) * 4
	listCurveLocations = [(curveSeed << 1) | curveSeed]

	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveSeed = (curveSeed << 4) | 0b101
		listCurveLocations.append((curveSeed << 1) | curveSeed)

	return dict.fromkeys(listCurveLocations, 1)

def A000682(n: int) -> int:
	# count64(n - 1, initializeA000682(n - 1))
	# print()
	return count(n - 1, initializeA000682(n - 1))
	# return count64(n - 1, initializeA000682(n - 1))
