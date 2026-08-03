"""You can use this script to benchmark map-folding algorithms against known OEIS values.

This script iterates through multiple OEIS map-folding sequences and algorithm implementations,
verifying computed results against reference values and measuring execution time. The script
prints colorized output indicating whether computed values match expected values.
"""
from __future__ import annotations

from hunterMakesPy import errorL33T
from mapFolding import ansiColorReset, ansiColors
from mapFolding.basecamp import countFolds
from mapFolding.oeis import getMapShape, getValuesKnown
from typing import TYPE_CHECKING
import sys
import time

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match := foldsTotal == getValuesKnown(oeisID).get(n, -errorL33T))}\t"
			f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}"
			f"{n}\t"
			f"{foldsTotal}\t"
			f"{getValuesKnown(oeisID).get(n, -errorL33T)}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			f"{ansiColorReset}\n"
		)

	pathLikeWrite: PathLike[str] | None = None
	computationDivisions: int | str | None = None
	CPUlimit: Limitation = None
	flow = 'daoOfMapFolding'
	flow = 'numba'
	flow = 'theorem2'
	flow = 'theorem2Codon'
	flow = 'theorem2Numba'

	oeisID = 'A001416'
	oeisID = 'A001418'
	oeisID = 'A195646'
	oeisID = 'A001417'
	oeisID = 'A001415'
	oeisID = 'A000136'

	sys.stdout.write(f"{ansiColors[int(oeisID, 36) % len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow, 36) % len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + '\n')

	for n in range(3, 25):

		mapShape: tuple[int, ...] = getMapShape(oeisID, n)

		timeStart = time.perf_counter()
		foldsTotal: int = countFolds(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit, computationDivisions=computationDivisions)

		_write()
