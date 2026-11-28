from __future__ import annotations

from mapFolding.basecamp import eliminateFolds
from mapFolding.kitTerminal import printEasyRunHeader
from mapFolding.oeis import makeMapShape, printEasyRunBenchmark
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.oeis import OEISid
	from os import PathLike

if __name__ == "__main__":

	pathLikeWrite: PathLike[str] | None = None
	oeisID: OEISid = "A001417"
	flow: str = "crease"
	CPUlimit: ConcurrencyLimit = 2

	printEasyRunHeader(oeisID, flow)
	ratio: int = 0

	# 'crease' requires (2,) * n with n >= 4.
	for n in range(4, 5):
		mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
		timeStart: float = time.perf_counter()
		computed: int = eliminateFolds(mapShape=mapShape, pathLikeWrite=pathLikeWrite, CPUlimit=CPUlimit, flow=flow)

		printEasyRunBenchmark(oeisID, n, computed, timeStart, ratio=ratio)
