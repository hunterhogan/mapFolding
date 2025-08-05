from mapFolding._oeisFormulas.A000136 import A000136
from mapFolding._oeisFormulas.A000682 import A000682
from mapFolding._oeisFormulas.Z0Z_oeisMeanders import dictionaryOEISMeanders
from mapFolding.oeis import dictionaryOEIS
import sys
import time

# ruff: noqa: ERA001

if __name__ == '__main__':
	oeisID = 'A000136'
	oeisID = 'A000682'
	for n in range(3, 13):

		# sys.stdout.write(f"{n = }\n")

		timeStart = time.perf_counter()
		foldsTotal = eval(oeisID)(n)
		# sys.stdout.write(f"{n} {foldsTotal} {time.perf_counter() - timeStart:.2f}\n")
		sys.stdout.write(f"{foldsTotal == dictionaryOEISMeanders[oeisID]['valuesKnown'][n]} {n} {foldsTotal} {time.perf_counter() - timeStart:.2f}\n") # pyright: ignore[reportIndexIssue]

