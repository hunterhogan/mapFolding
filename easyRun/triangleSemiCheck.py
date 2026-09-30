from __future__ import annotations

from mapFolding.basecamp import countMeanders
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from mapFolding.oeis.A400429 import checkFormulas
from typing import TYPE_CHECKING
import gc
import time

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateMeanders
	from mapFolding.theTypes import OEISid

if __name__ == '__main__':
	flow: str = 'prune'
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
		state: StateMeanders = countMeanders(kind, n, flow)
		countTotal: int = sum(state.lookupMeanders.values()) + state.countAddend

		printEasyRunBenchmark(oeisID, n, countTotal, timeStart, ratio=False)

	checkFormulas()
