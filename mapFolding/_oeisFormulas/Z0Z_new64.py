"""Semi-meanders."""
# pyright: reportUnusedImport=false
from mapFolding._oeisFormulas.matrixMeanders import count
from mapFolding._oeisFormulas.matrixMeanders64 import count64
from mapFolding._oeisFormulas.matrixMeandersMimic import countMimic
from mapFolding._oeisFormulas.Z0Z_oeisMeanders import dictionaryOEISMeanders
import sys
import time

def initializeA000682(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)

	curveSeed: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveSeed << 1) | curveSeed]

	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveSeed = (curveSeed << 4) | 0b101
		listCurveLocations.append((curveSeed << 1) | curveSeed)

	return dict.fromkeys(listCurveLocations, 1)

def A000682(n: int) -> int:
	count64Maximum = 29
	bridgesMinimum = 0
	distinctCrossingsMaximum = 40
	distinctCrossingsSlack = 2
	dictionaryCurveLocations: dict[int, int] = initializeA000682(n - 1)
	if n >= count64Maximum:
		if n >= distinctCrossingsMaximum:
			bridgesMinimum = n - distinctCrossingsMaximum + distinctCrossingsSlack
		n, dictionaryCurveLocations = countMimic(n - 1, dictionaryCurveLocations, count64Maximum)
		n += 1
	n, dictionaryCurveLocations = count64(n - 1, dictionaryCurveLocations, bridgesMinimum)
	if n > 0:
		n += 1
		return count(n - 1, dictionaryCurveLocations)
	return sum(dictionaryCurveLocations.values())

# ruff: noqa: ERA001

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(booleanColor:=(foldsTotal == dictionaryOEISMeanders[oeisID]['valuesKnown'][n]))}\t" # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
			f"\033[{(not booleanColor)*91}m"
			f"{n}\t"
			# f"{foldsTotal}\t"
			# f"{dictionaryOEISMeanders[oeisID]['valuesKnown'][n]=}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			# f"{description}\t"
			"\033[0m\n"
		)
	oeisID = 'A000682'
	for n in range(35,47):

		timeStart = time.perf_counter()
		foldsTotal = A000682(n)
		# sys.stdout.write(f"{n} {foldsTotal} {time.perf_counter() - timeStart:.2f}\n")
		_write()
