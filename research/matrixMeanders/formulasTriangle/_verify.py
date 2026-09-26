from __future__ import annotations

from fractions import Fraction
from functools import partial
from hunterMakesPy import ansiColor, ansiColorReset, errorL33T
from itertools import chain, count, filterfalse, groupby, islice, repeat, starmap
from mapFolding.dataStructures import makeLookupTriangle
from mapFolding.kitFilesystem import readDiagonal, readTriangle
from mapFolding.oeis import getValuesKnown
from operator import add, itemgetter
from research.matrixMeanders.formulasTriangle import A000136, A000682, A005315, A005316, A006661, A076876, A077054, A077460, boxOfDiagonals
from research.matrixMeanders.formulasTriangle._fromTriangleCells import calculateDiagonal2, calculateDiagonal3
from research.matrixMeanders.infoBooth import (
	makeFilenameDiagonal, pathData, pathFilenameTriangleSemiCommaSeparatedValues, pathFilenameTriangleSemiSubmissionOEIS,
	pathFilenameTriangleSemiText)
from textwrap import wrap
from typing import TYPE_CHECKING
import sys

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Mapping
	from mapFolding.theTypes import 形Triangle
	from pathlib import Path

type Report = tuple[bool, str]
type FormulaValue = int | Fraction | tuple[int, ...]

def printReport(description: str, *, match: bool) -> None:
	message: str = f"{(ansiColor.YellowOnRed, ansiColor.GreenOnBlack)[match]}{match}{ansiColorReset} {description}"
	sys.stdout.write(message + "\n")

def printReports(reports: Iterable[Report], width: int = 160) -> None:
	def printGroup(group: tuple[bool, Iterator[Report]]) -> None:
		match, entries = group
		descriptions: Iterable[str] = map(itemgetter(1), entries)
		if match:
			descriptions = wrap(' '.join(map(str.replace, descriptions, repeat(' '), repeat('\N{NO-BREAK SPACE}')))
				, width=width - len('True '), break_long_words=False, break_on_hyphens=False)
			descriptions = map(str.replace, descriptions, repeat('\N{NO-BREAK SPACE}'), repeat(' '))
		tuple(map(partial(printReport, match=match), descriptions))

	tuple(map(printGroup, groupby(reports, itemgetter(0))))

def checkFormula(description: str, formula: Callable[[int], FormulaValue], valuesKnown: Mapping[int, FormulaValue]) -> tuple[Report, ...]:
	def checkTerm(n: int, known: FormulaValue) -> Report:
		computed: FormulaValue = formula(n)
		return computed == known, f"{description}\t{n}\t{computed}\t{known}"

	checks: tuple[Report, ...] = tuple(map(checkTerm, valuesKnown.keys(), valuesKnown.values()))
	failures: tuple[Report, ...] = tuple(filterfalse(itemgetter(0), checks))
	return ((not failures, f"{description}: {len(checks) - len(failures)}/{len(checks)}."), *failures)

def getKnownValue(oeisID: str, n: int) -> int:
	return getValuesKnown(oeisID).get(n, -errorL33T)

def checkOEISFormula(oeisID: str, formula: Callable[[int], int | Fraction], domain: range, description: str | None = None) -> tuple[Report, ...]:
	return checkFormula(description or oeisID, formula, dict(zip(domain, map(partial(getKnownValue, oeisID), domain), strict=True)))

def getTerm(triangle: 形Triangle, column: int, n: int) -> int:
	return triangle[n][column]

def selectColumn(triangle: 形Triangle, column: int, domain: range) -> dict[int, int]:
	return dict(zip(domain, map(partial(getTerm, triangle, column), domain), strict=True))

def readSubmissionTriangle(pathFilename: Path) -> dict[int, tuple[int, ...]]:
	rowsFlatList: list[int] = []
	for line in pathFilename.read_text(encoding='utf-8').splitlines():
		if not line.startswith(('%S ', '%T ', '%U ')):
			continue
		rowsFlatList.extend(map(int, filter(None, line.split(maxsplit=2)[2].split(','))))
	rowsFlat: tuple[int, ...] = tuple(rowsFlatList)
	triangleSubmission: dict[int, list[int]] = makeLookupTriangle(rowsFlat, (n // 2 for n in count(2)), rowStart=2)
	return dict(zip(triangleSubmission, map(tuple, triangleSubmission.values()), strict=True))

def readSubmissionExamples(pathFilename: Path) -> dict[int, tuple[int, ...]]:
	def parseExample(line: str) -> tuple[int, tuple[int, ...]]:
		row, _, values = line.split(maxsplit=2)[2].rstrip('.;').partition(':')
		return int(row), tuple(map(int, values.split(',')))

	return dict(map(parseExample, filter(lambda line: line.startswith('%E A400429 ') and line[11].isdigit()
		, pathFilename.read_text(encoding='utf-8').splitlines())))

def readTriangleText(pathFilename: Path) -> dict[int, int]:
	def parseTerm(index: str, value: str) -> tuple[int, int]:
		return int(index), int(value)

	return dict(starmap(parseTerm, map(str.split, pathFilename.read_text(encoding='utf-8').splitlines())))

def checkA005315(triangle: 形Triangle) -> tuple[Report, ...]:
	return checkOEISFormula('A005315', partial(A005315, triangle=triangle)
		, range(1, min(max(triangle) // 2, max(getValuesKnown('A005315'))) + 1))

def checkDiagonal(次diagonal: int, 工diagonal: Callable[[int], int], *, triangle: 形Triangle, pathData: Path) -> tuple[Report, ...]:
	return tuple(chain(
		checkFormula(f"D_{次diagonal}", 工diagonal, selectColumn(triangle, -次diagonal, range(2 * 次diagonal, max(triangle) + 1)))
		, checkFormula(f"D_{次diagonal} / {makeFilenameDiagonal(次diagonal)}", 工diagonal
			, readDiagonal(pathData / makeFilenameDiagonal(次diagonal), 次diagonal, formatData='diagonalCSV'))
	))

def checkSubmission(triangle: 形Triangle, triangleSubmission: 形Triangle) -> tuple[Report, ...]:
	def calculateFirstColumn(n: int) -> int:
		return getKnownValue('A005315', (n + 1) // 2)

	def calculateSecondColumn(n: int) -> int:
		return getKnownValue('A005315', n + 1) - 2 * getKnownValue('A005316', 2 * n)

	triangleWithZeros: dict[int, tuple[int, ...]] = {1: (0, 0, 0)} | dict(zip(triangle, map(add, triangle.values(), repeat((0, 0))), strict=True))

	calculateA000136: Callable[[int], int] = partial(A000136, triangle=triangle)
	calculateA000682: Callable[[int], int] = partial(A000682, triangle=triangle)
	calculateA005316: Callable[[int], Fraction] = partial(A005316, triangle=triangleWithZeros)
	calculateA076876: Callable[[int], Fraction] = partial(A076876, triangle=triangleWithZeros)
	calculateA006661: Callable[[int], Fraction] = partial(A006661, triangle=triangleWithZeros)
	calculateA077054: Callable[[int], Fraction] = partial(A077054, triangle=triangleWithZeros)
	calculateA077460: Callable[[int], Fraction] = partial(A077460, triangle=triangleWithZeros)

	reports: tuple[Report, ...] = (
		*checkFormula('A400429 DATA', triangle.__getitem__, triangleSubmission)
		, *checkFormula('A400429 DATA row length', lambda n: len(triangleSubmission[n])
			, dict(zip(triangleSubmission, map(int.__floordiv__, triangleSubmission, repeat(2)), strict=True)))
		, ('%O A400429 2,2' in pathFilenameTriangleSemiSubmissionOEIS.read_text(encoding='utf-8').splitlines(), 'A400429 OFFSET 2,2')
		, *checkFormula('A400429 EXAMPLE', triangle.__getitem__, readSubmissionExamples(pathFilenameTriangleSemiSubmissionOEIS))
		, *checkOEISFormula('A000136', calculateA000136, range(2, max(triangle) + 1))
		, *checkOEISFormula('A000682', calculateA000682, range(2, max(triangle) + 1))
		, *checkFormula('T(n,1)', calculateFirstColumn, selectColumn(triangle, 0, range(2, max(triangle) + 1)))
		, *checkFormula('D_1 = 2 - 0^(n-2)', lambda n: 2 - 0**(n - 2)
			, selectColumn(triangle, -1, range(2, max(triangle) + 1)))
		, *checkFormula('D_2 draft formula', calculateDiagonal2, selectColumn(triangle, -2, range(4, max(triangle) + 1)))
		, *checkFormula('D_3 draft formula', calculateDiagonal3, selectColumn(triangle, -3, range(6, max(triangle) + 1)))
	)
	return (
		*reports
		, *checkOEISFormula('A005316', calculateA005316, range(5, max(triangle) + 1))
		, *checkOEISFormula('A076876', calculateA076876, range(6, min(max(triangle) - 2, max(getValuesKnown('A076876'))) + 1, 2))
		, *checkOEISFormula('A006661', calculateA006661, range(10, min(max(triangle) + 2, max(getValuesKnown('A006661'))) + 1, 2))
		, *checkOEISFormula('A077054', calculateA077054, range(2, (max(triangle) - 1) // 2 + 1))
		, *checkOEISFormula('A077460', calculateA077460, range(2, max(triangle) // 2 + 1, 2), 'A077460 even')
		, *checkOEISFormula('A077460', calculateA077460, range(3, max(triangle) // 2 + 1, 2), 'A077460 odd')
		, *checkFormula('T(2*n,2)', calculateSecondColumn, dict(zip(range(2, max(triangle) // 2 + 1)
			, selectColumn(triangle, 1, range(4, max(triangle) + 1, 2)).values(), strict=True)))
	)

def checkFormulas() -> None:
	triangle: dict[int, tuple[int, ...]] = readTriangle(pathFilenameTriangleSemiCommaSeparatedValues)
	triangleSubmission: dict[int, tuple[int, ...]] = readSubmissionTriangle(pathFilenameTriangleSemiSubmissionOEIS)
	triangleFlat: dict[int, int] = dict(enumerate(chain.from_iterable(triangle.values()), 2))
	triangleText: dict[int, int] = readTriangleText(pathFilenameTriangleSemiText)
	reports: tuple[Report, ...] = tuple(chain(
		((tuple(triangle) == tuple(range(2, max(triangle) + 1)), 'triangleSemi.csv consecutive rows')
			, (tuple(triangleText) == tuple(range(2, len(triangleText) + 2)), 'triangleSemi.txt consecutive indices')
			, (len(triangleText) >= len(triangleFlat), f'triangleSemi.txt extra partial-row terms: {len(triangleText) - len(triangleFlat)}'))
		, checkFormula('triangleSemi.csv row length', lambda n: len(triangle[n])
			, dict(zip(triangle, map(int.__floordiv__, triangle, repeat(2)), strict=True)))
		, checkFormula('triangleSemi.csv / triangleSemi.txt complete rows', lambda n: triangleFlat.get(n, -errorL33T)
			, dict(islice(triangleText.items(), len(triangleFlat))))
		, checkSubmission(triangle, triangleSubmission)
		, checkA005315(triangle)
		, chain.from_iterable(map(partial(checkDiagonal, triangle=triangle, pathData=pathData), range(1, len(boxOfDiagonals) + 1), boxOfDiagonals))
	))
	printReports(reports)
	if not all(map(itemgetter(0), reports)):
		raise SystemExit(1)

if __name__ == '__main__':
	checkFormulas()
