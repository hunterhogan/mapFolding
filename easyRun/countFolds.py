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
			f"{dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]}\t"
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

	oeisID: str = 'A195646'
	oeisID: str = 'A001418'
	oeisID: str = 'A000136'
	oeisID: str = 'A001416'
	oeisID: str = 'A001415'
	oeisID: str = 'A001417'

	sys.stdout.write(f"\033[{30+int(oeisID,11)%8};{40+int(oeisID,12)%8}m{oeisID} ")
	sys.stdout.write(f"\033[{31+int(flow,35)%7};{41+int(flow,36)%7}m{flow}")
	sys.stdout.write("\033[0m\n")

	for n in range(6,7):

		mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]['getMapShape'](n)

		timeStart = time.perf_counter()
		foldsTotal: int = countFolds(listDimensions=listDimensions
						, pathLikeWriteFoldsTotal=pathLikeWriteFoldsTotal
						, computationDivisions=computationDivisions
						, CPUlimit=CPUlimit
						, mapShape=mapShape
						, flow=flow)

		_write()
