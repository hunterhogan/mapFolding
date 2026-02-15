# ruff: noqa
# pyright: basic
from functools import partial
from itertools import filterfalse
from mapFolding import ansiColorReset, ansiColors
from mapFolding._e import PermutationSpace
from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import between吗, extractPinnedLeaves
from mapFolding._e.pin2上nDimensions import (
	pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零)
from mapFolding.oeis import dictionaryOEISMapFolding
from os import PathLike
from pathlib import Path, PurePath
from tqdm import tqdm
import csv
import sys
import time

if __name__ == "__main__":

	def _write() -> None:
		sys.stdout.write(
			f"{(match := foldsTotal == dictionaryOEISMapFolding[oeisID]['valuesKnown'][n])}\t"
			f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}"
			f"{n}\t"
			# f"{mapShape}\t"
			f"{foldsTotal}\t"
			f"{dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			f"{ansiColorReset}\n"
		)

	pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
	oeisID: str = ""
	flow: str = ""
	CPUlimit: bool | float | int | None = -2
	state: EliminationState | None = None

	flow = "elimination"
	flow = "constraintPropagation"
	flow = "crease"

	oeisID: str = "A195646"
	oeisID: str = "A000136"
	oeisID: str = "A001416"
	oeisID: str = "A001418"
	oeisID: str = "A001415"
	oeisID: str = "A001417"

	sys.stdout.write(f"{ansiColors[int(oeisID, 36) % len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow, 36) % len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + "\n")

	for n in range(4,6):
		mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]["getMapShape"](n)
		if oeisID == "A001417" and n > 3:
			state = EliminationState(mapShape)
			state = pinLeavesDimensions0零一(state)
			# state = pinPilesAtEnds(state, 4)
			# state = pinPile零Ante首零(state)
			# state = pinLeavesDimension二(state)
			# state = pinLeavesDimension首二(state)

		timeStart = time.perf_counter()
		foldsTotal: int = eliminateFolds(mapShape=mapShape, state=state, pathLikeWriteFoldsTotal=pathLikeWriteFoldsTotal, CPUlimit=CPUlimit, flow=flow)

		_write()

r"""
title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on mapFolding\_e\easyRun\eliminateFolds.py & title I'm done
"""

