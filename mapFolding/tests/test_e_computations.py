from mapFolding import dictionaryOEISMapFolding
from mapFolding._e.algorithms.eliminationCrease import pileProcessingOrderDefault
from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.tests.conftest import standardizedEqualToCallableReturn
from more_itertools import all_unique, unique_to_each
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Sequence

@pytest.mark.parametrize(
	"oeisID, n, flow, CPUlimit",
	[
		pytest.param("A001417", 4, "crease", 0.25),

		pytest.param("A000136", 5, "constraintPropagation", 0.25),
		pytest.param("A001415", 5, "constraintPropagation", 0.25),
		pytest.param("A001416", 4, "constraintPropagation", 0.25),
		pytest.param("A001417", 4, "constraintPropagation", 0.25),
		pytest.param("A001418", 3, "constraintPropagation", 0.25),
		pytest.param("A195646", 2, "constraintPropagation", 0.25),
		# *[pytest.param(oeisID, metadata["offset"], flow, 1)
		# 	for oeisID, metadata in dictionaryOEISMapFolding.items()
		# 		for flow in ["elimination", "constraintPropagation"]]


	],
)
def test_eliminateFoldsMapShape(oeisID: str, n: int, flow: str, CPUlimit: float) -> None:
	"""Validate `eliminateFolds` and different flows produce valid results.

	Parameters
	----------
	oeisID : str
		OEIS identifier to validate.
	sequenceIndex : int
		Sequence index to validate.
	flow : str
		Computation flow to validate.
	processorLimit : float
		CPU limit for the computation.
	"""
	mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]["getMapShape"](n)
	state = None
	pathLikeWriteFoldsTotal: None = None
	expected: int = dictionaryOEISMapFolding[oeisID]['valuesKnown'][n]
	standardizedEqualToCallableReturn(expected, eliminateFolds, mapShape, state, pathLikeWriteFoldsTotal, CPUlimit, flow)

@pytest.mark.parametrize(
	"mapShape2上nDimensions",
	[
		pytest.param((2, 2, 2, 2), id="2d4"),
		pytest.param((2, 2, 2, 2, 2), id="2d5"),
		pytest.param((2, 2, 2, 2, 2, 2), id="2d6"),
	],
)
def test_pileProcessingOrderDefault_properties(mapShape2上nDimensions: tuple[int, ...]) -> None:
	"""Verify pileProcessingOrderDefault returns complete, unique, in-range pile ordering."""
	state = EliminationState(mapShape=mapShape2上nDimensions)
	sequencePileOrder: Sequence[int] = pileProcessingOrderDefault(state)

	assert len(sequencePileOrder) == state.leavesTotal, (
		f"pileProcessingOrderDefault: expected {state.leavesTotal} piles for {mapShape2上nDimensions=}, "
		f"got {len(sequencePileOrder)}."
	)
	assert all_unique(sequencePileOrder), (
		f"pileProcessingOrderDefault: duplicate piles for {mapShape2上nDimensions=}."
	)
	pilesExpected: tuple[int, ...] = tuple(range(state.leavesTotal))
	pilesMissing, pilesExtra = unique_to_each(pilesExpected, sequencePileOrder)
	pilesMissing = tuple(pilesMissing)
	pilesExtra = tuple(pilesExtra)
	assert len(pilesMissing) == 0 and len(pilesExtra) == 0, (
		f"pileProcessingOrderDefault: missing or extra piles for {mapShape2上nDimensions=}. "
		f"Missing={pilesMissing}, extra={pilesExtra}."
	)
	assert all(0 <= pile < state.leavesTotal for pile in sequencePileOrder), (
		f"pileProcessingOrderDefault: out-of-range pile values for {mapShape2上nDimensions=}."
	)
