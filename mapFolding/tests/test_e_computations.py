from mapFolding import dictionaryOEISMapFolding
from mapFolding._e.basecamp import eliminateFolds
from mapFolding.tests.conftest import mapShapeFromTestCase, standardizedEqualToCallableReturn, TestCase

def test_eliminateFolds(testCaseEliminateFolds: TestCase) -> None:
	"""Validate `eliminateFolds` and different flows produce valid results.

	Parameters
	----------
	eliminateFoldsTestCase : TestCase
		TestCase describing the OEIS index and flow to validate.
	"""
	mapShape: tuple[int, ...] = mapShapeFromTestCase(testCaseEliminateFolds)
	state = None
	pathLikeWriteFoldsTotal: None = None
	expected: int = dictionaryOEISMapFolding[testCaseEliminateFolds.oeisID]['valuesKnown'][testCaseEliminateFolds.n]
	standardizedEqualToCallableReturn(expected, eliminateFolds, mapShape, state, pathLikeWriteFoldsTotal, testCaseEliminateFolds.CPUlimit, testCaseEliminateFolds.flow)