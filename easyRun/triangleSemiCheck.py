from __future__ import annotations

from itertools import repeat
from mapFolding.algorithms.matrixMeandersTriangle import doTheNeedful
from mapFolding.dataBaskets import StateMeanders
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from operator import call
from research.matrixMeanders.formulasTriangle import A005315of0, boxOfDiagonals, checkFormulas
from typing import TYPE_CHECKING
import gc
import time

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid

if __name__ == '__main__':
	checkFormulas()
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
		diagonals: int = len(boxOfDiagonals)
		totalDiagonals: int = n // 2
		countTotal: int = 0
		if diagonals < totalDiagonals:
			arcCodeMAXIMUM: int = 1 << (2 * (n - 1) + 4)
			arcCode: int = 0b1 + (0b100 * (n & 1))
			boxOfArcCodes: list[int] = [(arcCode << 1) | arcCode]
			while boxOfArcCodes[-1] < (arcCodeMAXIMUM >> 8):
				arcCode = (arcCode << 4) | 0b0101
				boxOfArcCodes.append((arcCode << 1) | arcCode)
			lookupMeanders: dict[int, int] = dict.fromkeys(boxOfArcCodes[0:-diagonals], 1)
			state: StateMeanders = StateMeanders(n, kind, lookupMeanders=lookupMeanders)
			countTotal = doTheNeedful(state)

		if n <= 2:
			countTotal = A005315of0(n)
		else:
			countTotal += sum(map(call, boxOfDiagonals[0:totalDiagonals], repeat(n)))  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]

		printEasyRunBenchmark(oeisID, n, countTotal, timeStart, ratio=False)
