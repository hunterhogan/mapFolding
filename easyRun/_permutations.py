# pyright: reportUnnecessaryComparison=false
from __future__ import annotations

from mapFolding.algorithms.permutations import doTheNeedful, StateStampMeander
from mapFolding.algorithms.permutationsBilateral import doTheNeedful as bilateral
from mapFolding.kitFilesystem import writeAlbum
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from pathlib import Path
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from os import PathLike

if __name__ == '__main__':
	flow = 'bilateral'
	flow = 'permutations'

	boxOfOEISid: list[OEISid] = []
	pathLikeWrite: PathLike[str] | None = None
	pathDirectoryWrite: Path | None = None

	if False:
		n: int = 2
		# ruff: ignore[repeated-append]
		boxOfOEISid.append('A001011')
		boxOfOEISid.append('A000136')
		boxOfOEISid.append('A077055')
		boxOfOEISid.append('A000560')
		boxOfOEISid.append('A000682')
	if True:
		boxOfOEISid.append('A005316')

	if pathLikeWrite is not None:
		pathDirectoryWrite = Path(pathLikeWrite)

	for oeisID in boxOfOEISid:
		printEasyRunHeader(oeisID, flow)

		for n in range(6, 17):

			timeStart: float = time.perf_counter()
			if flow == 'bilateral':
				if oeisID == 'A000560':
					total: int = bilateral(n, symmetric=True)
				elif n <= 2:
					total = 1
				else:
					total = bilateral(n - 1, symmetric=False)
			else:
				state: StateStampMeander = doTheNeedful(oeisID, n)
				total = state.total
				if pathDirectoryWrite is not None:
					writeAlbum(state.boxOfPermutations, pathDirectoryWrite / f'{oeisID}_{n}.csv')

			printEasyRunBenchmark(oeisID, n, total, timeStart, ratio=True)

# Until I figure out how to integrate into basecamp, this must be a proto-basecamp
# Create save to file option, but save the list of permutations as CSV using `writeAlbum`.
