"""You can use this script to benchmark map-folding algorithms against known OEIS values.

This script iterates through multiple OEIS map-folding sequences and algorithm implementations,
verifying computed results against reference values and measuring execution time. The script
prints colorized output indicating whether computed values match expected values.
"""
from __future__ import annotations

from mapFolding import ansiColorReset, ansiColors
from mapFolding.basecamp import countFolds
from mapFolding.oeis import dictionaryOEISMapFolding
from typing import TYPE_CHECKING
import sys
import time

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike
	from pathlib import PurePath

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match := foldsTotal == dictionaryOEISMapFolding[oeisID]['valuesKnown'].get(n))}\t"
			f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}"
			f"{n}\t"
			f"{foldsTotal}\t"
			f"{dictionaryOEISMapFolding[oeisID]['valuesKnown'].get(n)}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			f"{ansiColorReset}\n"
		)

	listDimensions: Sequence[int] | None = None
	pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
	computationDivisions: int | str | None = None
	CPUlimit: Limitation = None
	flow = 'daoOfMapFolding'
	flow = 'numba'
	flow = 'theorem2'
	flow = 'theorem2Codon'
	flow = 'theorem2Numba'

	oeisID = 'A001416'
	oeisID = 'A001418'
	oeisID = 'A000136'
	oeisID = 'A195646'
	oeisID = 'A001417'
	oeisID = 'A001415'

	sys.stdout.write(f"{ansiColors[int(oeisID, 36) % len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow, 36) % len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + '\n')

	for n in range(3, 15):

		mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)

		timeStart = time.perf_counter()
		foldsTotal: int = countFolds(listDimensions
						, pathLikeWriteFoldsTotal
						, computationDivisions
						, CPUlimit=CPUlimit
						, mapShape=mapShape
						, flow=flow)

		_write()
