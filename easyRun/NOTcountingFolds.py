# ruff: noqa
from collections import ChainMap
from mapFolding import dictionaryOEIS, dictionaryOEISMapFolding
from mapFolding.basecamp import NOTcountingFolds
import sys
import time

dictionaryONE = ChainMap(dictionaryOEISMapFolding, dictionaryOEIS) # pyright: ignore[reportArgumentType]

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match:=countTotal == dictionaryONE[oeisID]['valuesKnown'][n])}\t"
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
	oeisID = 'A000136'

	flow = 'asynchronous'
	flow = 'theorem2Numba'
	flow = 'theorem2Trimmed'
	flow = 'algorithm'
	flow = 'eliminationParallel'
	flow = 'elimination'

	for n in range(3,12):
	# for n in range(3,8):

		timeStart = time.perf_counter()
		countTotal = NOTcountingFolds(oeisID, n, flow, CPUlimit)

		_write()

r"""
deactivate && C:\apps\mapFolding\.vtail\Scripts\activate.bat && title good && cls
title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\NOTcountingFolds.py & title I'm done
"""

