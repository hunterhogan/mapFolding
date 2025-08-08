from mapFolding._oeisFormulas.A000560 import A000560
from mapFolding._oeisFormulas.A000682 import A000682
from mapFolding._oeisFormulas.A001010 import A001010
from mapFolding._oeisFormulas.A001011 import A001011
from mapFolding._oeisFormulas.A005315 import A005315
from mapFolding._oeisFormulas.A005316 import A005316
from mapFolding._oeisFormulas.A223094 import A223094
from mapFolding._oeisFormulas.A259702 import A259702
from mapFolding._oeisFormulas.A301620 import A301620
from mapFolding.oeis import getOEISidInformation, getOEISidValues
import sys

oeisIDsMeanders: list[str] = [
	'A000560',
	'A000682',
	'A001010',
	'A001011',
	'A005315',
	'A005316',
	'A223094',
	'A259702',
	'A301620',
]

dictionaryOEISMeanders: dict[str, dict[str, dict[int, int] | str | int]] = {
	oeisID: {
		'valuesKnown': getOEISidValues(oeisID),
		'description': getOEISidInformation(oeisID)[0],
		'offset': getOEISidInformation(oeisID)[1],
	}
	for oeisID in oeisIDsMeanders
}

# ruff: noqa: S101
# pyright: reportIndexIssue=false

rangeTest = range(5, 13)

if __name__ == '__main__':
	for n in rangeTest:

		assert A000560(n) == dictionaryOEISMeanders['A000560']['valuesKnown'][n]
		assert A000682(n) == dictionaryOEISMeanders['A000682']['valuesKnown'][n]
		assert A001010(n) == dictionaryOEISMeanders['A001010']['valuesKnown'][n]
		assert A001011(n) == dictionaryOEISMeanders['A001011']['valuesKnown'][n]
		assert A005315(n) == dictionaryOEISMeanders['A005315']['valuesKnown'][n]
		assert A005316(n) == dictionaryOEISMeanders['A005316']['valuesKnown'][n]
		assert A223094(n) == dictionaryOEISMeanders['A223094']['valuesKnown'][n]
		assert A259702(n) == dictionaryOEISMeanders['A259702']['valuesKnown'][n]
		assert A301620(n) == dictionaryOEISMeanders['A301620']['valuesKnown'][n]

	sys.stdout.write(f"\nTrue for {str(rangeTest)}\n")
