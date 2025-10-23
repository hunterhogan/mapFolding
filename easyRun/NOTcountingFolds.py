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
	flow: str | None = None

	oeisID = 'A007822'
	oeisID = 'A000136'

	flow = 'algorithm'
	flow = 'theorem2'
	flow = 'eliminationParallel'
	flow = 'elimination_combi'
	flow = 'constraintPropagation'
	flow = 'elimination'

	sys.stdout.write(f"\033[{30+int(oeisID,11)%8};{40+int(oeisID,12)%8}m{oeisID} ")
	sys.stdout.write(f"\033[{31+int(flow,35)%7};{41+int(flow,36)%7}m{flow}")
	sys.stdout.write("\033[0m\n")

	nList: list[int] = []
	nList.extend(range(7, 11))
	# nList.extend(range(9, 13))
	# nList.extend(range(11, 15))
	# nList.extend(range(13, 17))

	for n in dict.fromkeys(nList):

		timeStart = time.perf_counter()
		countTotal = NOTcountingFolds(oeisID, n, flow, CPUlimit)

		_write()

r"""
deactivate && C:\apps\mapFolding\.vtail\Scripts\activate.bat && title good && cls
title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\NOTcountingFolds.py & title I'm done
"""
