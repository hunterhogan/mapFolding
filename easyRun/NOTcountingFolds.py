# ruff: noqa
from mapFolding import dictionaryOEIS
from mapFolding.basecamp import NOTcountingFolds
import sys
import time

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match:=countTotal == dictionaryOEIS[oeisID]['valuesKnown'][n])}\t"
			f"\033[{(not match)*91}m"
			f"{n}\t"
			f"{countTotal}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			"\033[0m\n"
		)

	CPUlimit: bool | float | int | None = -2
	# oeisID: str | None = None
	oeis_n: int | None = None
	flow: str | None = None

	oeisID = 'A001010'
	oeisID = 'A007822'

	flow = 'theorem2Trimmed'
	flow = 'theorem2Numba'
	flow = 'algorithm'
	flow = 'asynchronous'

	for n in range(2,8):
	# for n in range(8,10):

		timeStart = time.perf_counter()
		countTotal = NOTcountingFolds(oeisID, n, flow, CPUlimit)

		_write()
