from __future__ import annotations

from mapFolding.algorithms.permutationsAnnex import doTheNeedful
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid

if __name__ == '__main__':
	flow = 'fast'

	boxOfOEISid: list[OEISid] = []

	if False:
		n: int = 2
	if True:
		# ruff: ignore[repeated-append]
		boxOfOEISid.append('A000560')
		boxOfOEISid.append('A005316')
		boxOfOEISid.append('A077055')
		boxOfOEISid.append('A001011')
		boxOfOEISid.append('A000136')
		boxOfOEISid.append('A000682')

	for oeisID in boxOfOEISid:
		printEasyRunHeader(oeisID, flow)

		for n in range(4, 5):

			timeStart: float = time.perf_counter()
			aOFn: int = doTheNeedful(oeisID, n)

			printEasyRunBenchmark(oeisID, n, aOFn, timeStart, ratio=False)
