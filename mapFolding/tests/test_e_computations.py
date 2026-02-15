from mapFolding._e.basecamp import eliminateFolds
from mapFolding.oeis import dictionaryOEISMapFolding
from mapFolding.tests.conftest import standardizedEqualToCallableReturn
import pytest

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
		# 		for flow in ["elimination"]]


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
