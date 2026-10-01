from __future__ import annotations

from mapFolding import leafOrigin, pileOrigin
from mapFolding._e import getLookupChoicesLeaf
from mapFolding._e.algorithms.elimination import theorem2b
from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import PermutationSpace, StateElimination
from mapFolding._e.p2上nDimensional import pinIt  # pyright: ignore[reportUnusedImport]
from mapFolding._e.pinIt import atPileExcludeLeaf, excludeLeaf_rBeforeLeaf_k
from mapFolding._e.reduceIt import boxOfFunctionsReductionDEFAULT
from mapFolding.oeis import makeMapShape, printEasyRunBenchmark, printEasyRunHeader
from mapFolding.oeis._byFormulaLookup import _A000136, _A000682
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
	state: StateElimination | None = None

	flow = "crease"
	flow = "elimination"
	flow = "constraintPropagation"

	oeisID = "A195646"
	oeisID = "A001416"
	oeisID = "A001417"
	oeisID = "A001418"
	oeisID = "A001415"
	oeisID = "A000136"

	printEasyRunHeader(oeisID, flow)

	for n in range(12, 14):
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
		if oeisID == "A000136" and 3 < n:  # pyright: ignore[reportUnnecessaryComparison]
			state = StateElimination(mapShape)
			state.boxOfPermutationSpace.append(PermutationSpace({pileOrigin: leafOrigin}).updatePilesMissing(getLookupChoicesLeaf(state)))
			state = excludeLeaf_rBeforeLeaf_k(state, 1, 2)
			state.boxOfPermutationSpace = atPileExcludeLeaf(state.boxOfPermutationSpace, 1, 1)
			state = theorem2b(state)
			# print(state)
			# state = state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReductionDEFAULT)

		computed: int = eliminateFolds(mapShape=mapShape, state=state, pathLikeWrite=pathLikeWrite, CPUlimit=CPUlimit, flow=flow)

		if oeisID == "A000136" and 3 < n:  # pyright: ignore[reportUnnecessaryComparison]
			computed += 2 * _A000682(state.totalLeaves - 1) * state.totalLeaves  # pyright: ignore[reportOptionalMemberAccess] # ty: ignore[unresolved-attribute]

		printEasyRunBenchmark(oeisID, n, computed, timeStart, ratio=False)

r"""
title running && start "working" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on mapFolding\_e\easyRun\eliminateFolds.py & title I'm done
"""
