from mapFolding._oeisFormulas.A000136 import A000136
from mapFolding.oeis import dictionaryOEIS
import sys
import time

# ruff: noqa: ERA001

if __name__ == '__main__':
	oeisID = 'A000136'
	for n in range(3,30):

		# print(n)

		timeStart = time.perf_counter()
		foldsTotal = A000136(n)
		# sys.stdout.write(f"{n} {foldsTotal} {time.perf_counter() - timeStart:.2f}\n")
		sys.stdout.write(f"{foldsTotal == dictionaryOEIS[oeisID]['valuesKnown'][n]} {n} {foldsTotal} {time.perf_counter() - timeStart:.2f}\n")

