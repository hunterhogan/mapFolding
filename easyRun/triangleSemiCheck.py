from __future__ import annotations

from mapFolding.algorithms.matrixMeandersTriangle import doTheNeedful
from mapFolding.dataBaskets import StateMeanders
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from research.matrixMeanders.formulasTriangle import checkFormulas
from typing import TYPE_CHECKING
import gc
import time

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid

if __name__ == '__main__':
	flow: str = 'triangleSemi'
	oeisID: OEISid = 'A000682'
	kind: str = 'semi'
	printEasyRunHeader(oeisID, flow)

	boxOf_n: list[int] = []
	boxOf_n.extend(range(2, 24))
	boxOf_n.extend(range(24, 28))
	# boxOf_n.extend(range(28, 33))
	# boxOf_n.extend(range(33, 38))

	for n in boxOf_n:
		gc.collect()
		timeStart: float = time.perf_counter()
		state: StateMeanders = doTheNeedful(StateMeanders(n, kind))
		countTotal: int = sum(state.lookupMeanders.values()) + state.countAddend

		printEasyRunBenchmark(oeisID, n, countTotal, timeStart, ratio=False)

	checkFormulas()
