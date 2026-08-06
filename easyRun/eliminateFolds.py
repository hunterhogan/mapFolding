# pyright: reportUnusedImport=false
from __future__ import annotations

from mapFolding._e._2上nDimensional.pinIt import (
	pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零,
	pin首beans)
from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.beDRY import printEasyRunBenchmark, printEasyRunHeader
from mapFolding.oeis import makeMapShape
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from mapFolding.theTypes import OEISid
	from os import PathLike

if __name__ == "__main__":

	pathLikeWrite: PathLike[str] | None = None
	oeisID: OEISid = ""
	flow: str = ""
	CPUlimit: Limitation = -2
	state: EliminationState | None = None

	flow = "elimination"
	flow = "constraintPropagation"
	flow = "crease"

	oeisID = "A195646"
	oeisID = "A000136"
	oeisID = "A001418"
	oeisID = "A001416"
	oeisID = "A001415"
	oeisID = "A001417"

	printEasyRunHeader(oeisID, flow)

	for n in range(4, 6):
		mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
		timeStart: float = time.perf_counter()
		if oeisID == "A001417" and n > 3:
			state = EliminationState(mapShape)
			# state = pinPile零Ante首零(state)
			state = pinPilesAtEnds(state, 3)
			# state = pinLeavesDimension首二(state)
			# state = pin3beans2(state)
			# state = pin首beans(state)
			# state = pinLeavesDimension一(state)
			# state = pinLeavesDimension二(state)
			state = pinLeavesDimensions0零一(state)
			# state.listPermutationSpace.reverse()

		computed: int = eliminateFolds(mapShape=mapShape, state=state, pathLikeWrite=pathLikeWrite, CPUlimit=CPUlimit, flow=flow)

		printEasyRunBenchmark(oeisID, n, computed, timeStart, ratio=False)

r"""
title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on mapFolding\_e\easyRun\eliminateFolds.py & title I'm done
"""
