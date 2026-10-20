# pyright: reportUnnecessaryComparison=false
from __future__ import annotations

from mapFolding.basecamp import enumerateSequences
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from pprint import pprint
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import OEISid
	from os import PathLike
	from typing import Literal

if __name__ == '__main__':
	flow = 'permutations'
	flow = 'bilateralConcurrent'
	flow = 'bilateral'

	boxOfSequences: list[tuple[OEISid, Literal['stamps blank', 'stamps', 'meanders symmetric', 'meanders', 'semi-meanders symmetric', 'semi']]] = []
	pathLikeWrite: PathLike[str] | None = None
	CPUlimit: ConcurrencyLimit = None

	if False:
		n: int = 2
		boxOfSequences.append(('A001011', 'stamps blank'))
		boxOfSequences.append(('A000136', 'stamps'))
		boxOfSequences.append(('A077055', 'meanders symmetric'))
		boxOfSequences.append(('A005316', 'meanders'))
		boxOfSequences.append(('A000560', 'semi-meanders symmetric'))
	if True:
		boxOfSequences.append(('A000682', 'semi'))

	ratio: int = 0

	for oeisID, kind in boxOfSequences:
		printEasyRunHeader(oeisID, flow)

		for n in range(6, 7):

			timeStart: float = time.perf_counter()
			album = enumerateSequences(n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit)
			total: int = album if isinstance(album, int) else len(album)

			printEasyRunBenchmark(oeisID, n, total, timeStart, ratio=ratio)

			# pprint(album)
