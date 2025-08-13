# pyright: reportUnusedImport=false
from mapFolding._oeisFormulas.A000136 import A000136
from mapFolding._oeisFormulas.A000682 import A000682
from mapFolding._oeisFormulas.A001010 import A001010
from mapFolding._oeisFormulas.Z0Z_oeisMeanders import dictionaryOEISMeanders
from mapFolding.oeis import dictionaryOEIS
import sys
import time

# ruff: noqa: ERA001

if __name__ == '__main__':
	def _write() -> None:
		sys.stdout.write(
			f"{(booleanColor:=(foldsTotal == dictionaryOEISMeanders[oeisID]['valuesKnown'][n]))}\t" # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
			f"\033[{(not booleanColor)*91}m"
			f"{n}\t"
			f"{foldsTotal}\t"
			# f"{dictionaryOEISMeanders[oeisID]['valuesKnown'][n]=}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			# f"{description}\t"
			"\033[0m\n"
		)
	oeisID = 'A001010'
	oeisID = 'A000136'
	oeisID = 'A000682'
	for n in range(3,17):

		# sys.stdout.write(f"{n = }\n")

		timeStart = time.perf_counter()
		foldsTotal = eval(oeisID)(n)  # noqa: S307
		# sys.stdout.write(f"{n} {foldsTotal} {time.perf_counter() - timeStart:.2f}\n")
		_write()
