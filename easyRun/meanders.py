# ruff: noqa
# pyright: basic
from mapFolding import dictionaryOEIS
from mapFolding.basecamp import NOTcountingFolds
import gc
import multiprocessing
import sys
import time
import warnings

def write() -> None:
	sys.stdout.write(
		f"{(booleanColor:=(countTotal == dictionaryOEIS[oeisID]['valuesKnown'][n]))}\t"
		f"\033[{(not booleanColor)*91}m"
		f"{n}\t"
		# f"{countTotal}\t"
		# f"{dictionaryOEISMeanders[oeisID]['valuesKnown'][n]}\t"
		f"{time.perf_counter() - timeStart:.2f}\t"
		"\033[0m\n"
	)

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')
	if sys.version_info >= (3, 14):
		warnings.filterwarnings("ignore", category=FutureWarning)

	flow = 'matrixPandas'
	flow = 'matrixMeanders'
	flow = 'matrixNumPy'

	for oeisID in [
			'A005316',
			'A000682',
				]:
		sys.stdout.write(f"\n{oeisID}\n")

		"""TODO Identifiers. improve
		"generate up to four targets."
		1. Adding a new loop.
		2. Dragging up a loop end.
		3. Dragging down a loop end.
		4. Connect ends across the line.

		flipTheExtra_0b1AsUfunc: what is extra?
		"""

		nList: list[int] = []
		nList.extend(range(2, 10))
		# nList.extend(range(10, 28))
		# nList.extend(range(28,33))
		# nList.extend(range(33,38))
		# nList.extend(range(38,43))
		# nList.extend(range(43,45))
		# nList.extend(range(47,57))

		for n in nList:
			gc.collect()
			timeStart = time.perf_counter()
			countTotal = NOTcountingFolds(oeisID, n, flow)
			if n < dictionaryOEIS[oeisID]['valueUnknown']:
				write()
			else:
				sys.stdout.write(f"{n} {countTotal} {time.perf_counter() - timeStart:.2f}\n")

r"""
deactivate && C:\apps\mapFolding\.vtail\Scripts\activate.bat && title good && cls
title running && start "meanders" /B /HIGH /wait py -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on easyRun\meanders.py && title I'm done || title Error

"""
