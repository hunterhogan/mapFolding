from __future__ import annotations

from mapFolding._e._2上nDimensional import pinIt
from mapFolding._e.basecamp import eliminateFolds
from mapFolding.oeis import makeMapShape, printEasyRunBenchmark, printEasyRunHeader
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from mapFolding._e.dataBaskets import StateElimination
	from mapFolding.theTypes import OEISid
	from os import PathLike

if __name__ == "__main__":

	pathLikeWrite: PathLike[str] | None = None
	oeisID: OEISid = ""
	flow: str = ""
	CPUlimit: Limitation = -2
	state: StateElimination | None = None

	flow = "constraintPropagation"
	flow = "crease"
	flow = "elimination"

	oeisID = "A195646"
	oeisID = "A001418"
	oeisID = "A001416"
	oeisID = "A001415"
	oeisID = "A001417"
	oeisID = "A000136"

	printEasyRunHeader(oeisID, flow)

	for n in range(2, 10):
		mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
		timeStart: float = time.perf_counter()
		if oeisID == "A001417" and 3 < n:  # pyright: ignore[reportUnnecessaryComparison]
			# state = StateElimination(mapShape)
			# state = pinIt.pinPile零Ante首零(state)
			# state = pinIt.pinPilesAtEnds(state, 3)
			# state = pinIt.pinLeavesDimension首二(state)
			# state = pinIt.pin3beans2(state)
			# state = pinIt.pin首beans(state)
			# state = pinIt.pinLeavesDimension一(state)
			# state = pinIt.pinLeavesDimension二(state)
			# state = pinIt.pinLeavesDimensions0零一(state)
			# state.boxOfPermutationSpace.reverse()
			pass

		computed: int = eliminateFolds(mapShape=mapShape, state=state, pathLikeWrite=pathLikeWrite, CPUlimit=CPUlimit, flow=flow)

		printEasyRunBenchmark(oeisID, n, computed, timeStart, ratio=False)

r"""
title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on mapFolding\_e\easyRun\eliminateFolds.py & title I'm done
"""
