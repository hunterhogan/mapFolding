from __future__ import annotations

from mapFolding.basecamp import countMeanders
from mapFolding.oeis import printEasyRunBenchmark, printEasyRunHeader
from mapFolding.theSSOT import settingsPackage
from pathlib import Path
from typing import TYPE_CHECKING
import gc
import sys
import time
import warnings

if TYPE_CHECKING:
	from os import PathLike
	from typing import LiteralString

if __name__ == '__main__':
	if (3, 14) <= sys.version_info:
		warnings.filterwarnings("ignore", category=FutureWarning)

	pathLikeWrite = Path(settingsPackage.pathPackage, 'jobs')
	pathLikeWrite: PathLike[str] | None = Path('/apps/mapFolding/mapFolding/jobs')
	pathLikeWrite = None
	flow = 'matrixPandas'
	flow = 'qq'
	flow = 'matrixNumPy'
	flow = 'matrixMeanders'

	literallyAnnoyingListOfLiteralStrings: list[tuple[LiteralString, LiteralString]] = [
			# ('A005315', 'closed'),
			# ('A005316', 'meanders'),
			('A000682', 'semi'),
		]

	for oeisID, kind in literallyAnnoyingListOfLiteralStrings:
		printEasyRunHeader(oeisID, flow)

		"""# Identifiers. improve
		"generate up to four targets."
		1. Adding a new loop. Simple analysis.
		2. Dragging up a loop end.
		3. Dragging down a loop end.
		4. Connect ends across the line.
		"""

		boxOf_n: list[int] = []
		boxOf_n.extend(range(2, 10))
		# boxOf_n.extend(range(10, 28))
		# boxOf_n.extend(range(28, 33))
		# boxOf_n.extend(range(33, 38))
		# boxOf_n.extend(range(38, 43))
		# boxOf_n.extend(range(43, 46))
		# boxOf_n.extend(range(46, 47))
		# boxOf_n.extend(range(47, 48))

		# boxOf_n.extend(range(10, 43, 2))
		# boxOf_n.extend(range(11, 43, 2))

		for n in boxOf_n:
			gc.collect()
			timeStart: float = time.perf_counter()
			countTotal: int = countMeanders(kind, n, flow, pathLikeWrite)
			if kind == 'semi' and False:  # ruff: ignore[expr-and-false]
				if 4 <= n:
					countTotal += 2
				if 6 <= n:
					countTotal += (n + 1)**2 // 2 - 10

			printEasyRunBenchmark(oeisID, n, countTotal, timeStart, ratio=True)

r"""

title running && start "meanders" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\_countMeanders.py && title I'm done || title Error

sudo nice -n -10 /home/hunte/mapFolding/.venv/bin/python -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on /home/hunte/mapFolding/easyRun/_countMeanders.py
"""
