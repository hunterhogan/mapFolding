# pyright: reportUnnecessaryComparison=false
from __future__ import annotations

from mapFolding.basecamp import enumerateSequences
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader, Z0Z_oeisName
from pprint import pprint
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import OEISid
	from os import PathLike
	from typing import LiteralString

if __name__ == '__main__':
	flow = 'permutations'
	flow = 'bilateralConcurrent'
	flow = 'bilateral'

	boxOfKind: list[LiteralString] = []
	pathLikeWrite: PathLike[str] | None = None
	CPUlimit: ConcurrencyLimit = None

	if False:
		n: int = 2
		boxOfKind.append('stamps blank')
		boxOfKind.append('stamps')
		boxOfKind.append('meander shapes')
		boxOfKind.append('meanders')
		# TODO Replace this A000560 symmetry name with its precise counting convention.
		boxOfKind.append('semi-meanders symmetric')
	if True:
		boxOfKind.append('semi')

	ratio: int = 0

	for kind in boxOfKind:
		oeisID: OEISid = Z0Z_oeisName(kind)
		printEasyRunHeader(oeisID, flow)

		for n in range(6, 7):

			timeStart: float = time.perf_counter()
			album = enumerateSequences(n, kind, flow, pathLikeWrite, CPUlimit=CPUlimit)
			total: int = album if isinstance(album, int) else len(album)

			printEasyRunBenchmark(oeisID, n, total, timeStart, ratio=ratio)

			# pprint(album)
