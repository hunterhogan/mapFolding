# ruff: noqa
# pyright: basic
from mapFolding import dictionaryOEISMapFolding, eliminateFolds
from mapFolding._e.pinning2Dn import (
	pinLeaf首零Plus零, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPiles, pinPile首零Less零)
from mapFolding.dataBaskets import EliminationState
from os import PathLike
from pathlib import PurePath
import sys
import time

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match:=foldsTotal == dictionaryOEISMapFolding[oeisID]['valuesKnown'][n])}\t"
			f"\033[{(not match)*91}m"
			f"{n}\t"
			# f"{mapShape}\t"
			f"{foldsTotal}\t"
			f"{dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			"\033[0m\n"
		)

	pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
	oeisID: str = ''
	flow: str = ''
	CPUlimit: bool | float | int | None = -2
	state: EliminationState | None = None

	flow = 'elimination'
	flow = 'crease'
	flow = 'constraintPropagation'

	oeisID: str = 'A195646'
	oeisID: str = 'A001416'
	oeisID: str = 'A001415'
	oeisID: str = 'A000136'
	oeisID: str = 'A001418'
	oeisID: str = 'A001417'

	sys.stdout.write(f"\033[{30+int(oeisID,11)%8};{40+int(oeisID,12)%8}m{oeisID} ")
	sys.stdout.write(f"\033[{31+int(flow,35)%7};{41+int(flow,36)%7}m{flow}")
	sys.stdout.write("\033[0m\n")

	for n in range(5,6):

		mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)
		if oeisID == 'A001417' and n > 3:
			state = EliminationState(mapShape)
			state = pinPiles(state, 4)
			# state = pinPile首零Less零(state)
			# state = pinLeaf首零Plus零(state)
			# state = pinLeavesDimension一(state)
			# state = pinLeavesDimension二(state)
			# state = pinLeavesDimension首二(state)

		timeStart = time.perf_counter()
		foldsTotal: int = eliminateFolds(
						mapShape=mapShape
						, state=state
						, pathLikeWriteFoldsTotal=pathLikeWriteFoldsTotal
						, CPUlimit=CPUlimit
						, flow=flow)

		_write()

r"""
deactivate && C:\apps\mapFolding\.vtail\Scripts\activate.bat && title good && cls

title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\eliminateFolds.py & title I'm done
"""

# maps of 3 x 3 ... x 3, divisible by leavesTotal * 2^dimensionsTotal * factorial(dimensionsTotal)
