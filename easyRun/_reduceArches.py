# pyright: reportUnnecessaryComparison=false
from __future__ import annotations

from itertools import chain
from mapFolding.algorithms.catalanArch import doTheNeedful
from mapFolding.kitFilesystem import writeStringToHere, writeTriangle
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from pathlib import Path
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from collections.abc import Callable
	from os import PathLike

def write(nStart: int, nStop: int, pathWrite: Path, counter: Callable[[int], list[int]] = doTheNeedful) -> Path:
	triangle: dict[int, list[int]] = {}
	pathFilename: Path = pathWrite / 'b287548ADDENDUM.txt'
	for n in tqdm(range(nStart, nStop)):
		triangle[n] = counter(n)
		writeTriangle(triangle, pathWrite / 'A287548ADDENDUM.csv')
		writeStringToHere(''.join(map('{} {}\n'.format, range(1, sum(map(len, triangle.values())) + 1),
			chain.from_iterable(triangle.values()))), pathFilename)
	return pathFilename

if __name__ == '__main__':
	pathLikeWrite: PathLike[str] | None = Path('/apps/mapFolding/research/archReduction')
	oeisID = 'A000682'

	flow = 'write'
	flow = 'reduce'
	nStart: int = 1
	nStop: int = 20

	printEasyRunHeader(oeisID, flow)

	if flow == 'reduce':
		for n in range(nStart, nStop):

			timeStart: float = time.perf_counter()
			triangleRow: list[int] = doTheNeedful(n)
			print(triangleRow)
			total: int = triangleRow[-1]

			printEasyRunBenchmark(oeisID, n, total, timeStart, ratio=False)

	else:
		write(nStart, nStop, pathLikeWrite)
