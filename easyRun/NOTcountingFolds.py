# ruff: noqa
from collections import ChainMap
from mapFolding import (
	ansiColorGreenOnBlack, ansiColorReset, ansiColors, ansiColorYellowOnRed, dictionaryOEIS, dictionaryOEISMapFolding)
from mapFolding.basecamp import NOTcountingFolds
import sys
import time

dictionaryONE = ChainMap(dictionaryOEISMapFolding, dictionaryOEIS) # pyright: ignore[reportArgumentType]

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match:=countTotal == dictionaryONE[oeisID]['valuesKnown'][n])}\t"
			f"{(ansiColorYellowOnRed, ansiColorGreenOnBlack)[match]}"
			f"{n}\t"
			f"{countTotal}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			f"{ansiColorReset}\n"
		)

	CPUlimit: bool | float | int | None = -2
	flow: str | None = None

	oeisID = 'A007822'
	oeisID = 'A000136'

	flow = 'algorithm'
	flow = 'theorem2'

	sys.stdout.write(f"{ansiColors[int(oeisID,36)%len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow,36)%len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + '\n')

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
