from __future__ import annotations

from archive.permutationMeanders.stamp_meander import doTheNeedful
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
import time

if __name__ == '__main__':
	flow = 'SawadaLi'

	oeisID = 'A077055'
	oeisID = 'A000136'
	oeisID = 'A000682'
	oeisID = 'A001011'
	oeisID = 'A000560'
	oeisID = 'A005316'

	printEasyRunHeader(oeisID, flow)

	for n in range(2, 15):

		timeStart: float = time.perf_counter()
		aOFn: int = doTheNeedful(oeisID, n)

		printEasyRunBenchmark(oeisID, n, aOFn, timeStart, ratio=False)
