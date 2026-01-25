# ruff: noqa
# pyright: basic
from collections.abc import Sequence
from mapFolding import ansiColorReset, ansiColors, dictionaryOEISMapFolding
from mapFolding.basecamp import countFolds
from os import PathLike
from pathlib import PurePath
import sys
import time

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match:=foldsTotal == dictionaryOEISMapFolding[oeisID]['valuesKnown'][n])}\t"
			f"{(ansiColors.YellowOnRed, ansiColors.GreenOnBlack)[match]}"
			f"{n}\t"
			f"{foldsTotal}\t"
			f"{dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			f"{ansiColorReset}\n"
		)

	listDimensions: Sequence[int] | None = None
	pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
	computationDivisions: int | str | None = None
	CPUlimit: bool | float | int | None = None
	# mapShape: tuple[int, ...] | None = None
	flow = 'numba'
	flow = 'theorem2'
	flow = 'theorem2Numba'
	flow = 'daoOfMapFolding'

	oeisID: str = 'A195646'
	oeisID: str = 'A001416'
	oeisID: str = 'A001418'
	oeisID: str = 'A001417'
	oeisID: str = 'A000136'
	oeisID: str = 'A001415'

	sys.stdout.write(f"{ansiColors[int(oeisID,36)%len(ansiColors)]}{oeisID} ")
	sys.stdout.write(f"{ansiColors[int(flow,36)%len(ansiColors)]}{flow}")
	sys.stdout.write(ansiColorReset + '\n')

	for n in range(2):

		mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)

		timeStart = time.perf_counter()
		foldsTotal: int = countFolds(listDimensions=listDimensions
						, pathLikeWriteFoldsTotal=pathLikeWriteFoldsTotal
						, computationDivisions=computationDivisions
						, CPUlimit=CPUlimit
						, mapShape=mapShape
						, flow=flow)

		_write()
