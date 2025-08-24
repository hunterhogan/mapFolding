"""Semi-meanders."""
from functools import cache
from mapFolding._oeisFormulas.matrixMeanders import doTheNeedful

def initializeA000682(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)

	curveSeed: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveSeed << 1) | curveSeed]

	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveSeed = (curveSeed << 4) | 0b101
		listCurveLocations.append((curveSeed << 1) | curveSeed)

	return dict.fromkeys(listCurveLocations, 1)

@cache
def A000682(n: int) -> int:
	return doTheNeedful(n - 1, initializeA000682(n - 1))
