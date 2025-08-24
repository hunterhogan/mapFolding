from mapFolding import dictionaryOEISMapFolding, MetadataOEISidMeanders
from mapFolding.algorithms.oeisIDbyFormula import (
	A000136, A000560, A000682, A001010, A001011, A005315, A005316, A060206, A077460, A078591, A223094, A259702, A301620)
from mapFolding.oeis import getOEISidInformation, getOEISidValues
import sys

oeisIDsMeanders: list[str] = [
	'A000560',
	'A000682',
	'A001010',
	'A001011',
	'A005315',
	'A005316',
	'A060206',
	'A077460',
	'A078591',
	'A223094',
	'A259702',
	'A301620',
]

dictionaryOEISMeanders: dict[str, MetadataOEISidMeanders] = {
	oeisID: {
		'description': getOEISidInformation(oeisID)[0],
		'offset': getOEISidInformation(oeisID)[1],
		'valuesKnown': getOEISidValues(oeisID),
	}
	for oeisID in oeisIDsMeanders
}

# ruff: noqa: S101

rangeTest = range(3, 7)

if __name__ == '__main__':
	import time
	start: float = time.perf_counter()
	for n in rangeTest:

		assert A000136(n) == dictionaryOEISMapFolding['A000136']['valuesKnown'][n]
		assert A000560(n) == dictionaryOEISMeanders['A000560']['valuesKnown'][n]
		assert A000682(n) == dictionaryOEISMeanders['A000682']['valuesKnown'][n]
		assert A001010(n) == dictionaryOEISMeanders['A001010']['valuesKnown'][n]
		assert A001011(n) == dictionaryOEISMeanders['A001011']['valuesKnown'][n]
		assert A005315(n) == dictionaryOEISMeanders['A005315']['valuesKnown'][n]
		assert A005316(n) == dictionaryOEISMeanders['A005316']['valuesKnown'][n]
		assert A060206(n) == dictionaryOEISMeanders['A060206']['valuesKnown'][n]
		assert A077460(n) == dictionaryOEISMeanders['A077460']['valuesKnown'][n]
		assert A078591(n) == dictionaryOEISMeanders['A078591']['valuesKnown'][n]
		assert A223094(n) == dictionaryOEISMeanders['A223094']['valuesKnown'][n]
		assert A259702(n) == dictionaryOEISMeanders['A259702']['valuesKnown'][n]
		assert A301620(n) == dictionaryOEISMeanders['A301620']['valuesKnown'][n]

	sys.stdout.write(f"\nTrue for {str(rangeTest)}\n")
	sys.stdout.write(f"Time taken: {time.perf_counter() - start:.2f} seconds\n")
