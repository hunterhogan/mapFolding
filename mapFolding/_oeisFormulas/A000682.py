from mapFolding._oeisFormulas.matrixMeanders import count, curveMaximum

def initializeA000682(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM = curveMaximum[n].curveLocationsMAXIMUM

	curveSeed: int = 5 - (n & 0b1) * 4
	listCurveLocations = [(curveSeed << 1) | curveSeed]

	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveSeed = (curveSeed << 4) | 0b101
		listCurveLocations.append((curveSeed << 1) | curveSeed)

	return dict.fromkeys(listCurveLocations, 1)

def A000682(n: int) -> int:
	return count(n - 1, initializeA000682(n - 1))

