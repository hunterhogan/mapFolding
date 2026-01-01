# ruff: noqa
# pyright: basic
from mapFolding import ansiColorGreenOnBlack, ansiColorReset, ansiColors, ansiColorYellowOnRed, dictionaryOEISMapFolding
from mapFolding._e import between, oopsAllLeaves
from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensions import (
	pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPiles, pinPile首零Less零)
from os import PathLike
from pathlib import Path, PurePath
import sys
import time

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match:=foldsTotal == dictionaryOEISMapFolding[oeisID]['valuesKnown'][n])}\t"
			f"{(ansiColorYellowOnRed, ansiColorGreenOnBlack)[match]}"
			f"{n}\t"
			# f"{mapShape}\t"
			f"{foldsTotal}\t"
			f"{dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			f"{ansiColorReset}"
		)

	pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
	oeisID: str = ''
	flow: str = ''
	CPUlimit: bool | float | int | None = -4
	state: EliminationState | None = None

	flow = 'crease'
	flow = 'elimination'
	flow = 'constraintPropagation'

	oeisID: str = 'A195646'
	oeisID: str = 'A001416'
	oeisID: str = 'A000136'
	oeisID: str = 'A001418'
	oeisID: str = 'A001415'
	oeisID: str = 'A001417'

	sys.stdout.write(f"{ansiColors[int(oeisID,36)%len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow,36)%len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + '\n')

	for n in range(7,8):

		mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)
		if oeisID == 'A001417' and n > 3:
			state = EliminationState(mapShape)
			# state = pinPiles(state, 4)
			state = pinLeavesDimensions0零一(state)
			# state = pinPile首零Less零(state)
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

title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on mapFolding\_e\easyRun\eliminateFolds.py & title I'm done
"""

