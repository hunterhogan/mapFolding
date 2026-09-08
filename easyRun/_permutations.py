# pyright: reportUnnecessaryComparison=false
from __future__ import annotations

from mapFolding.algorithms.permutations import doTheNeedful, StateStampMeander
from mapFolding.algorithms.permutationsBilateral import doTheNeedful as bilateral
from mapFolding.algorithms.permutationsBilateralConcurrent import doTheNeedful as bilateralConcurrent
from mapFolding.kitFilesystem import makePathFilenameCount, writeAlbum
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from os import PathLike

if __name__ == '__main__':
	flow = 'permutations'
	flow = 'bilateral'
	flow = 'bilateralConcurrent'

	boxOfOEISid: list[OEISid] = []
	pathLikeWrite: PathLike[str] | None = None

	if False:
		n: int = 2
		boxOfOEISid.append('A001011')
		boxOfOEISid.append('A000136')
		boxOfOEISid.append('A077055')
		boxOfOEISid.append('A005316')
		boxOfOEISid.append('A000560')
	if True:
		boxOfOEISid.append('A000682')

	for oeisID in boxOfOEISid:
		printEasyRunHeader(oeisID, flow)

		for n in range(17, 20):

			timeStart: float = time.perf_counter()
			# Until I figure out how to integrate into basecamp, this must be a proto-basecamp
			# Need CPUlimit argument.
			# Still too slow: try numba and/or codon.
			if flow == 'bilateralConcurrent':
				if oeisID == 'A000560' and 2 <= n:
					total: int = bilateralConcurrent(n, symmetric=True)
				else:
					total = bilateralConcurrent(n - 1, symmetric=False)
			elif flow == 'bilateral':
				if oeisID == 'A000560' and 2 <= n:
					total = bilateral(n, symmetric=True)
				else:
					total = bilateral(n - 1, symmetric=False)
			else:
				state: StateStampMeander = doTheNeedful(oeisID, n)
				total = state.total
				if pathLikeWrite is not None:
					writeAlbum(state.boxOfPermutations, makePathFilenameCount(pathLikeWrite, oeisID, n, suffix='.album'))

			printEasyRunBenchmark(oeisID, n, total, timeStart, ratio=False)
