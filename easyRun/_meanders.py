from __future__ import annotations

from mapFolding.basecamp import countMeanders
from mapFolding.beDRY import printEasyRunBenchmark, printEasyRunHeader
from pathlib import Path
from typing import TYPE_CHECKING
import gc
import sys
import time
import warnings

if TYPE_CHECKING:
	from os import PathLike

if __name__ == '__main__':
	if (3, 14) <= sys.version_info:
		warnings.filterwarnings("ignore", category=FutureWarning)

	pathLikeWrite: PathLike[str] | None = Path('/apps/mapFolding/mapFolding/jobs')
	pathLikeWrite = None
	flow = 'matrixMeanders'
	flow = 'matrixPandas'
	flow = 'matrixNumPy'

	for oeisID, kind in [
			('A005316', 'meanders'),
			# ('A000682', 'semi'),
		]:
		printEasyRunHeader(oeisID, flow)

		"""# Identifiers. improve
		"generate up to four targets."
		1. Adding a new loop.
		2. Dragging up a loop end.
		3. Dragging down a loop end.
		4. Connect ends across the line.
		"""

		nList: list[int] = []
		nList.extend(range(2, 10))
		nList.extend(range(10, 28))
		# nList.extend(range(28, 33))
		# nList.extend(range(33, 38))
		# nList.extend(range(38, 43))
		# nList.extend(range(43, 45))
		# nList.extend(range(45, 50))

		for n in nList:
			gc.collect()
			timeStart: float = time.perf_counter()
			countTotal: int = countMeanders(kind, n, flow, pathLikeWrite)  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]

			printEasyRunBenchmark(oeisID, n, countTotal, timeStart, ratio=False)

r"""

title running && start "meanders" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\meanders.py && title I'm done || title Error

"""
