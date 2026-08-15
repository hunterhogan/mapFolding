"""You can use this script to benchmark map-folding algorithms against known OEIS values.

This script iterates through multiple OEIS map-folding sequences and algorithm implementations,
verifying computed results against reference values and measuring execution time. The script
prints colorized output indicating whether computed values match expected values.
"""
from __future__ import annotations

from mapFolding.basecamp import countFolds
from mapFolding.oeis import makeMapShape, printEasyRunBenchmark, printEasyRunHeader
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike

if __name__ == '__main__':

	pathLikeWrite: PathLike[str] | None = None
	computationDivisions: int | str | None = None
	CPUlimit: Limitation = None
	flow = 'numba'
	flow = 'theorem2'
	flow = 'daoOfMapFolding'
	flow = 'theorem2Numba'
	flow = 'daoOfMapFoldingNumba'

	oeisID = 'A001416'
	oeisID = 'A195646'
	oeisID = 'A001418'
	oeisID = 'A001417'
	oeisID = 'A000136'
	oeisID = 'A001415'

	printEasyRunHeader(oeisID, flow)

	for n in range(3, 10):

		mapShape: tuple[int, ...] = makeMapShape(oeisID, n)

		timeStart: float = time.perf_counter()
		totalFolds: int = countFolds(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit, computationDivisions=computationDivisions)

		printEasyRunBenchmark(oeisID, n, totalFolds, timeStart, ratio=False)
