# ruff: noqa
# pyright: basic
from collections.abc import Sequence
from mapFolding import countFolds, dictionaryOEISMapFolding
from os import PathLike
from pathlib import PurePath
import sys
import time

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(match:=foldsTotal == dictionaryOEISMapFolding[oeisID]['valuesKnown'][n])}\t"
			f"\033[{(not match)*91}m"
			f"{n}\t"
			f"{foldsTotal}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			"\033[0m\n"
		)

	listDimensions: Sequence[int] | None = None
	pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
	computationDivisions: int | str | None = None
	CPUlimit: bool | float | int | None = None
	# mapShape: tuple[int, ...] | None = None
	flow = 'daoOfMapFolding'
	flow = 'numba'
	flow = 'theorem2'
	flow = 'theorem2Numba'
	flow: str | None = 'theorem2Trimmed'


	oeisID: str = 'A001415'
	oeisID: str = 'A000136'
	for n in range(1,4):

		mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)

		timeStart = time.perf_counter()
		# foldsTotal: int = countFolds(listDimensions=None, pathLikeWriteFoldsTotal=None, computationDivisions=None, CPUlimit=None, mapShape=(2, 3), flow='theorem2Trimmed')
		foldsTotal: int = countFolds(listDimensions=listDimensions
						, pathLikeWriteFoldsTotal=pathLikeWriteFoldsTotal
						, computationDivisions=computationDivisions
						, CPUlimit=CPUlimit
						, mapShape=mapShape
						, flow=flow)

		_write()
