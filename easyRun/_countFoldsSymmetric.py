"""You can use this script to benchmark map-folding algorithms against known OEIS values."""
from __future__ import annotations

from mapFolding.basecamp import countFoldsSymmetric
from mapFolding.oeis import makeMapShape, printEasyRunBenchmark, printEasyRunHeader
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from mapFolding.theTypes import OEISid
	from os import PathLike

if __name__ == '__main__':
	oeisID: OEISid = 'A007822'
	pathLikeWrite: PathLike[str] | None = None
	CPUlimit: Limitation = None
	flow = 'asynchronous'
	flow = 'theorem2'
	flow = 'theorem2Trimmed'
	flow = 'theorem2Numba'
	flow = 'algorithm'
	flow = 'algorithmNumba'

	printEasyRunHeader(oeisID, flow)

	for n in range(9, 11):

		mapShape: tuple[int, ...] = makeMapShape(oeisID, n)

		timeStart: float = time.perf_counter()
		totalFolds: int = countFoldsSymmetric(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit)

		printEasyRunBenchmark(oeisID, n, totalFolds, timeStart, ratio=False)
