# ruff:file-ignore[commented-out-code, undocumented-public-module, undocumented-public-function]
from __future__ import annotations

from hunterMakesPy import errorL33T
from mapFolding import ansiColorReset, ansiColors
from mapFolding.basecamp import countMeanders
from mapFolding.oeis import getValuesKnown
from typing import TYPE_CHECKING
import gc
import sys
import time
import warnings

if TYPE_CHECKING:
	from os import PathLike

def write() -> None:
	sys.stdout.write(
		f"{(match := (countTotal == totalKnown))}\t"
		f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}"
		f"{n}\t"
		f"{countTotal}\t"
		f"{totalKnown}\t"
		f"{totalKnown / countTotal}\t"
		f"{time.perf_counter() - timeStart:.2f}\t"
		f"{ansiColorReset}\n"
	)

if __name__ == '__main__':
	if (3, 14) <= sys.version_info:
		warnings.filterwarnings("ignore", category=FutureWarning)

	pathLikeWriteTotal: PathLike[str] | None = '/apps/mapFolding/mapFolding/jobs'  # pyright: ignore[reportAssignmentType] # ty: ignore[invalid-assignment]
	pathLikeWriteTotal = None
	flow = 'matrixMeanders'
	flow = 'matrixPandas'
	flow = 'matrixNumPy'

	for oeisID in [
			'A005316',
			# 'A000682',
				]:
		sys.stdout.write(f"\n{oeisID}\n")

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
			timeStart = time.perf_counter()
			countTotal = countMeanders(oeisID, n, flow, pathLikeWriteTotal)
			totalKnown = getValuesKnown(oeisID).get(n, -errorL33T)
			if 0 <= totalKnown:
				write()
			else:
				sys.stdout.write(f"{n} {countTotal} {time.perf_counter() - timeStart:.2f}\n")

r"""

title running && start "meanders" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\meanders.py && title I'm done || title Error

"""
