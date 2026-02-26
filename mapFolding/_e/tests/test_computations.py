from mapFolding._e.basecamp import eliminateFolds
from mapFolding.oeis import dictionaryOEISMapFolding
from mapFolding.tests.conftest import standardizedEqualToCallableReturn
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import EliminationState

@pytest.mark.parametrize("expected, oeisID, n, flow, CPUlimit", [
	*[pytest.param(dictionaryOEISMapFolding[oeisID]['valuesKnown'][n], oeisID, n, "crease", 0.99) for oeisID, n in (('A001417', 4), ('A001417', 5))],
	*[pytest.param(dictionaryOEISMapFolding[oeisID]['valuesKnown'][n], oeisID, n, "constraintPropagation", 0.99) for oeisID, n in (("A000136", 5), ("A001415", 5), ("A001416", 4), ("A001417", 4), ("A001418", 3), ("A195646", 2))],
	*[pytest.param(dictionaryOEISMapFolding[oeisID]['valuesKnown'][n], oeisID, n, "elimination", 0.99) for oeisID, n in (("A000136", 3), ("A001415", 3), ("A001416", 2), ("A001417", 2), ("A001418", 2), ("A195646", 1))],
	*[pytest.param(dictionaryOEISMapFolding[oeisID]['valuesKnown'][dictionaryOEISMapFolding[oeisID]["offset"]], oeisID, dictionaryOEISMapFolding[oeisID]["offset"], "constraintPropagation", 1) for oeisID in ('A000136', 'A001415', 'A001416', 'A001418')],
	*[pytest.param(ValueError, oeisID, dictionaryOEISMapFolding[oeisID]["offset"], "constraintPropagation", 1) for oeisID in ('A001417', 'A195646')],
	*[pytest.param(metadata['valuesKnown'][metadata["offset"] + 1], oeisID, metadata["offset"] + 1, "constraintPropagation", 1) for oeisID, metadata in dictionaryOEISMapFolding.items()],
])
def test_eliminateFoldsMapShape(expected: int | type[Exception], oeisID: str, n: int, flow: str, CPUlimit: float) -> None:
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
	state: EliminationState | None = None
	pathLikeWriteFoldsTotal: None = None
	standardizedEqualToCallableReturn(expected, eliminateFolds, mapShape, state, pathLikeWriteFoldsTotal, CPUlimit, flow)
