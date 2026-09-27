from __future__ import annotations

from fractions import Fraction
from functools import partial
from hunterMakesPy import ansiColor, ansiColorReset, errorL33T
from itertools import chain, filterfalse, groupby, repeat
from mapFolding.oeis import getTriangleDiagonal, getTriangleRows, getValuesKnown
from operator import add, itemgetter
from research.matrixMeanders.formulasTriangle import A000136, A000682, A005315, A005316, A006661, A076876, A077054, A077460, boxOfDiagonals
from research.matrixMeanders.formulasTriangle._fromTriangleCells import calculateDiagonal2, calculateDiagonal3
from textwrap import wrap
from typing import TYPE_CHECKING
import sys

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Mapping
	from mapFolding.theTypes import 形Triangle

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

def checkA005315(triangle: 形Triangle) -> tuple[Report, ...]:
	return checkOEISFormula('A005315', partial(A005315, triangle=triangle)
		, range(1, min(max(triangle) // 2, max(getValuesKnown('A005315'))) + 1))

def checkDiagonal(次diagonal: int, 工diagonal: Callable[[int], int]) -> tuple[Report, ...]:
	return checkFormula(f"D_{次diagonal} / A400429", 工diagonal, getTriangleDiagonal('A400429', 次diagonal))

def checkTriangle(triangle: 形Triangle) -> tuple[Report, ...]:
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
		*checkOEISFormula('A000136', calculateA000136, range(2, max(triangle) + 1))
		, *checkOEISFormula('A000682', calculateA000682, range(2, max(triangle) + 1))
		, *checkFormula('T(n,1)', calculateFirstColumn, selectColumn(triangle, 0, range(2, max(triangle) + 1)))
		, *checkFormula('D_1 = 2 - 0^(n-2)', lambda n: 2 - 0**(n - 2)
			, selectColumn(triangle, -1, range(2, max(triangle) + 1)))
		, *checkFormula('D_2 formula / A400429', calculateDiagonal2, getTriangleDiagonal('A400429', 2))
		, *checkFormula('D_3 formula / A400429', calculateDiagonal3, getTriangleDiagonal('A400429', 3))
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
	triangleRows: dict[int, list[int]] = getTriangleRows('A400429')
	triangle: dict[int, tuple[int, ...]] = dict(zip(triangleRows, map(tuple, triangleRows.values()), strict=True))
	triangle = dict(filter(lambda row: len(row[1]) == row[0] // 2, triangle.items()))
	triangleOfficial: dict[int, int] = getValuesKnown('A400429')
	reports: tuple[Report, ...] = tuple(chain(
		((tuple(triangle) == tuple(range(2, max(triangle) + 1)), 'A400429 consecutive complete rows')
			, (tuple(triangleOfficial) == tuple(range(2, len(triangleOfficial) + 2)), 'A400429 official consecutive indices')
			, (tuple(chain.from_iterable(triangleRows.values())) == tuple(triangleOfficial.values()), 'A400429 rows preserve all official terms'))
		, checkFormula('A400429 complete row length', lambda n: len(triangle[n])
			, dict(zip(triangle, map(int.__floordiv__, triangle, repeat(2)), strict=True)))
		, checkTriangle(triangle)
		, checkA005315(triangle)
		, chain.from_iterable(map(checkDiagonal, range(1, len(boxOfDiagonals) + 1), boxOfDiagonals))
	))
	printReports(reports)
	if not all(map(itemgetter(0), reports)):
		raise SystemExit(1)

if __name__ == '__main__':
	checkFormulas()
