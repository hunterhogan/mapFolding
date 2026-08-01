# ruff:file-ignore[commented-out-code]
# pyright: basic
from __future__ import annotations

from mapFolding import ansiColorReset, ansiColors
from mapFolding._e._2上nDimensional.pinIt import (
	pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零,
	pin首beans)
from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.oeis import getMapShape
from mapFolding.oeis._metadata import dictionaryOEIS
from typing import TYPE_CHECKING
import sys
import time

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike

if __name__ == "__main__":

	def _write() -> None:
		sys.stdout.write(
			f"{(match := foldsTotal == dictionaryOEIS[oeisID]['valuesKnown'][n])}\t"
			f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}"
			f"{n}\t"
			# f"{mapShape}\t"
			f"{foldsTotal}\t"
			f"{dictionaryOEIS[oeisID]['valuesKnown'][n]}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			f"{ansiColorReset}\n"
		)

	pathLikeWriteTotal: PathLike[str] | None = None
	oeisID: str = ""
	flow: str = ""
	CPUlimit: Limitation = -2
	state: EliminationState | None = None

	flow = "elimination"
	flow = "crease"
	flow = "constraintPropagation"

	oeisID = "A195646"
	oeisID = "A000136"
	oeisID = "A001418"
	oeisID = "A001416"
	oeisID = "A001415"
	oeisID = "A001417"

	sys.stdout.write(f"{ansiColors[int(oeisID, 36) % len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow, 36) % len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + "\n")

	for n in range(6, 8):
		mapShape: tuple[int, ...] = getMapShape(oeisID, n)
		timeStart: float = time.perf_counter()
		if oeisID == "A001417" and n > 3:
			state = EliminationState(mapShape)
			state = pinPilesAtEnds(state, 1)
			permutationSpace = state.listPermutationSpace.pop()
			for pile, leaf in [(2, 5), (3, 7)]:
				permutationSpace = permutationSpace.atPilePinLeaf(pile, leaf)
			state.listPermutationSpace.append(permutationSpace)
			# state = pinPile零Ante首零(state)
			state = pinPilesAtEnds(state, 3)
			# state = pinLeavesDimension首二(state)
			# state = pin3beans2(state)
			# state = pin首beans(state)
			# state = pinLeavesDimension一(state)
			# state = pinLeavesDimension二(state)
			# state = pinLeavesDimensions0零一(state)
			# state.listPermutationSpace.reverse()

		foldsTotal: int = eliminateFolds(mapShape=mapShape, state=state, pathLikeWriteTotal=pathLikeWriteTotal, CPUlimit=CPUlimit, flow=flow)

		_write()

r"""
title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on mapFolding\_e\easyRun\eliminateFolds.py & title I'm done
"""
