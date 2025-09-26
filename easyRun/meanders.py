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
		kOfMatrix: I don't think the paper uses 'k'. step?

		ReidemeisterMove?
		flipTheExtra_0b1AsUfunc: what is extra?

		"strand" is an interesting word.
		"""

		# for n in range(44,45):

		# for n in range(43,56):
		# for n in range(46,47):
		# for n in range(43,47):
		# for n in range(38,43):
		# for n in range(28,38):
		# for n in [*range(2, 10), *range(28,33)]:
		# for n in range(28,33):
		# for n in range(2, 28):
		# for n in range(2, 10):
		for n in range(1, 6):

			gc.collect()
			timeStart = time.perf_counter()
			countTotal = NOTcountingFolds(oeisID, n, flow)
			if n < dictionaryOEIS[oeisID]['valueUnknown']:
				write()
			else:
				sys.stdout.write(f"{n} {countTotal} {time.perf_counter() - timeStart:.2f}\n")

r"""
deactivate && C:\apps\mapFolding\.vtail\Scripts\activate.bat && title good && cls
title running && py Z0Z_aOFn.py && title I'm done || title Error

"""
