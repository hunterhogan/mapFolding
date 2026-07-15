# ruff: noqa: ERA001, D100
from __future__ import annotations

from collections import ChainMap
from mapFolding import ansiColorReset, ansiColors
from mapFolding.oeis import countingMeanders, dictionaryOEIS, dictionaryOEISMapFolding
from typing import TYPE_CHECKING
import sys
import time

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation

dictionaryONE = ChainMap(dictionaryOEISMapFolding, dictionaryOEIS)  # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-argument-type]

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match := countTotal == dictionaryONE[oeisID]['valuesKnown'][n])}\t"  # ty:ignore[not-subscriptable]
			f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}"
			f"{n}\t"
			f"{countTotal}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			f"{ansiColorReset}\n"
		)

	CPUlimit: Limitation = -2
	flow: str | None = None

	oeisID = 'A007822'
	oeisID = 'A000136'

	flow = 'algorithm'
	flow = 'theorem2'

	sys.stdout.write(f"{ansiColors[int(oeisID, 36) % len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow, 36) % len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + '\n')

	nList: list[int] = []
	nList.extend(range(7, 11))
	# nList.extend(range(9, 13))
	# nList.extend(range(11, 15))
	# nList.extend(range(13, 17))

	for n in dict.fromkeys(nList):

		timeStart = time.perf_counter()
		countTotal = countingMeanders(oeisID, n, flow, None, CPUlimit=CPUlimit)

		_write()

r"""

title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\countingMeanders.py & title I'm done
"""
