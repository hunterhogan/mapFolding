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
			f"{(match:=foldsTotal == dictionaryOEISMeanders[oeisID]['valuesKnown'][n])}\t"
			f"\033[{(not match)*91}m"
			f"{n}\t"
			f"{foldsTotal=}\t"
			f"{dictionaryOEISMeanders[oeisID]['valuesKnown'][n]=}\t"
			f"{time.perf_counter() - timeStart:.2f}\t"
			# f"{description}\t"
			"\033[0m\n"
		)
	oeisID = 'A000136'
	oeisID = 'A000682'
	oeisID = 'A001010'
	for n in range(2, 13):

		# sys.stdout.write(f"{n = }\n")

		timeStart = time.perf_counter()
		foldsTotal = eval(oeisID)(n)
		# sys.stdout.write(f"{n} {foldsTotal} {time.perf_counter() - timeStart:.2f}\n")
		_write()
