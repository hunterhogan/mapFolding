# pyright: reportUnusedImport=false
from mapFolding._oeisFormulas.A000682NumPy import A000682
from mapFolding._oeisFormulas.oeisIDbyFormula import A000136, A001010
from mapFolding._oeisFormulas.Z0Z_oeisMeanders import dictionaryOEISMeanders
from mapFolding.oeis import dictionaryOEISMapFolding
import gc
import sys
import time

# ruff: noqa: ERA001

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(booleanColor:=(foldsTotal == dictionaryOEISMeanders[oeisID]['valuesKnown'][n]))}\t" # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
			f"\033[{(not booleanColor)*91}m"
			f"{n}\t"
			# f"{foldsTotal}\t"
			# f"{dictionaryOEISMeanders[oeisID]['valuesKnown'][n]=}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			# f"{description}\t"
			"\033[0m\n"
		)
	oeisID = 'A001010'
	oeisID = 'A000136'
	oeisID = 'A000682'
	for n in range(44,47):
	# for n in range(30,38):
	# for n in range(3,30):

		# sys.stdout.write(f"{n = }\n")
		gc.collect()
		timeStart = time.perf_counter()
		foldsTotal = A000682(n)
		if n <= 45:
			_write()
		else:
			sys.stdout.write(f"{n} {foldsTotal} {time.perf_counter() - timeStart:.2f}\n")
