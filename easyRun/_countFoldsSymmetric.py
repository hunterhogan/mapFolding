"""You can use this script to benchmark map-folding algorithms against known OEIS values."""
from __future__ import annotations

from hunterMakesPy import errorL33T
from mapFolding import ansiColorReset, ansiColors
from mapFolding.basecamp import countFoldsSymmetric
from mapFolding.oeis import getMapShape, getValuesKnown
from typing import TYPE_CHECKING
import sys
import time

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike

if __name__ == '__main__':
	oeisID: str = 'A007822'
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
	CPUlimit: Limitation = None
	flow = 'asynchronous'
	flow = 'theorem2'
	flow = 'theorem2Codon'
	flow = 'theorem2Trimmed'
	flow = 'theorem2Numba'
	flow = 'algorithm'

	sys.stdout.write(f"{ansiColors[int(oeisID, 36) % len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow, 36) % len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + '\n')

	for n in range(1, 6):

		mapShape: tuple[int, ...] = getMapShape(oeisID, n)

		timeStart: float = time.perf_counter()
		foldsTotal: int = countFoldsSymmetric(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit)

		_write()
