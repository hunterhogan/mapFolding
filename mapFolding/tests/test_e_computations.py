from mapFolding import dictionaryOEISMapFolding
from mapFolding._e.algorithms.eliminationCrease import pileProcessingOrderDefault
from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.tests.conftest import mapShapeFromTestCase, standardizedEqualToCallableReturn, TestCase
from more_itertools import all_unique, unique_to_each
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence

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


def test_pileProcessingOrderDefault_properties(mapShape2上nDimensionsStandard: tuple[int, ...]) -> None:
	"""Verify pileProcessingOrderDefault returns complete, unique, in-range pile ordering."""
	state = EliminationState(mapShape=mapShape2上nDimensionsStandard)
	sequencePileOrder: Sequence[int] = pileProcessingOrderDefault(state)

	assert len(sequencePileOrder) == state.leavesTotal, (
		f"pileProcessingOrderDefault: expected {state.leavesTotal} piles for {mapShape2上nDimensionsStandard=}, "
		f"got {len(sequencePileOrder)}."
	)
	assert all_unique(sequencePileOrder), (
		f"pileProcessingOrderDefault: duplicate piles for {mapShape2上nDimensionsStandard=}."
	)
	pilesExpected: tuple[int, ...] = tuple(range(state.leavesTotal))
	pilesMissing, pilesExtra = unique_to_each(pilesExpected, sequencePileOrder)
	pilesMissing = tuple(pilesMissing)
	pilesExtra = tuple(pilesExtra)
	assert len(pilesMissing) == 0 and len(pilesExtra) == 0, (
		f"pileProcessingOrderDefault: missing or extra piles for {mapShape2上nDimensionsStandard=}. "
		f"Missing={pilesMissing}, extra={pilesExtra}."
	)
	assert all(0 <= pile < state.leavesTotal for pile in sequencePileOrder), (
		f"pileProcessingOrderDefault: out-of-range pile values for {mapShape2上nDimensionsStandard=}."
	)
