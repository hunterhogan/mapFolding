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

	CPUlimit: bool | float | int | None = 3
	# oeisID: str | None = None
	oeis_n: int | None = None
	flow: str | None = None

	oeisID = 'A001010'
	oeisID = 'A007822'

	flow = 'algorithm'
	flow = 'asynchronous'
	flow = 'theorem2Numba'
	flow = 'theorem2Trimmed'

	# for n in range(7,10):
	for n in range(3,8):

		timeStart = time.perf_counter()
		countTotal = NOTcountingFolds(oeisID, n, flow, CPUlimit)

		_write()
