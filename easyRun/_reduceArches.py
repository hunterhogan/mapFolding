# pyright: reportUnnecessaryComparison=false
from __future__ import annotations

from hunterMakesPy import inclusive
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
	pathFilename: Path = pathWrite / 'b287548.txt'
	for n in tqdm(range(nStart, nStop + inclusive), initial=nStart - 1):
		triangle[n] = counter(n)
		writeTriangle(triangle, pathWrite / 'triangle.csv')
		writeStringToHere(''.join(map('{} {}\n'.format, range(1, sum(map(len, triangle.values())) + 1),
			chain.from_iterable(triangle.values()))), pathFilename)
	return pathFilename

if __name__ == '__main__':
	flow = 'write'
	flow = 'reduce'
	pathLikeWrite: PathLike[str] | None = Path('/apps/mapFolding/research/archReduction')
	oeisID = 'A000682'

	printEasyRunHeader(oeisID, flow)

	if flow == 'reduce':
		for n in range(1, 17):

			timeStart: float = time.perf_counter()
			total = doTheNeedful(n)[-1]

			printEasyRunBenchmark(oeisID, n, total, timeStart, ratio=False)

	else:
		write(1, 20, pathLikeWrite)
