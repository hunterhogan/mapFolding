from __future__ import annotations

from gmpy2 import mpz
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import extractPinnedLeaves
from mapFolding._e.Z0Z_analysisPython.toolkit import beansWithoutCornbread
from typing import Protocol, TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from mapFolding._e import PermutationSpace
	from numpy.typing import NDArray

class PinningFunction(Protocol):
	__name__: str

	def __call__(self, state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState: ...

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
def test_pinningFunctions(loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]], pinningFunction2上nDimensional: PinningFunction, CPUlimitPinningTests: float, dimensionsTotal: int) -> None:
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state: EliminationState = EliminationState(mapShape=mapShape)
	arrayFoldings: NDArray[numpy.uint8] = loadArrayFoldings(dimensionsTotal)

	state = pinningFunction2上nDimensional(state, CPUlimit=CPUlimitPinningTests)
	functionName: str = getattr(pinningFunction2上nDimensional, "__name__", pinningFunction2上nDimensional.__class__.__name__)

	assert state.listPermutationSpace, f"{functionName} returned empty listPermutationSpace for {mapShape=} after pinning."

	requiredRowsTotal: int = int(arrayFoldings.shape[0])
	listMaskRequiredRowsMatchThisPermutationSpace: list[numpy.ndarray] = []

	for permutationSpace in state.listPermutationSpace:
		maskRequiredRowsMatchThisPermutationSpace: numpy.ndarray = numpy.ones(requiredRowsTotal, dtype=bool)
		for pile, leafSpace in extractPinnedLeaves(permutationSpace).items():
			if isinstance(leafSpace, int):
				maskRequiredRowsMatchThisPermutationSpace = maskRequiredRowsMatchThisPermutationSpace & (arrayFoldings[:, pile] == leafSpace)
				continue
			if isinstance(leafSpace, mpz):
				allowedLeaves: numpy.ndarray = numpy.fromiter((bool(leafSpace[leaf]) for leaf in range(state.leavesTotal)), dtype=bool, count=state.leavesTotal)
				maskRequiredRowsMatchThisPermutationSpace = maskRequiredRowsMatchThisPermutationSpace & allowedLeaves[arrayFoldings[:, pile]]
		listMaskRequiredRowsMatchThisPermutationSpace.append(maskRequiredRowsMatchThisPermutationSpace)

	masksStacked: numpy.ndarray = numpy.column_stack(listMaskRequiredRowsMatchThisPermutationSpace)
	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	indicesOverlappedRequiredRows: numpy.ndarray = numpy.nonzero(coverageCountPerRow >= 2)[0]

	countOverlappingDictionaries: int = 0
	if indicesOverlappedRequiredRows.size > 0:
		for maskRequiredRowsMatchThisPermutationSpace in listMaskRequiredRowsMatchThisPermutationSpace:
			if bool(maskRequiredRowsMatchThisPermutationSpace[indicesOverlappedRequiredRows].any()):
				countOverlappingDictionaries += 1

	maskRequiredRowsCoveredByAnyPermutationSpace: numpy.ndarray = numpy.logical_or.reduce(listMaskRequiredRowsMatchThisPermutationSpace)
	requiredRowsCoveredTotal: int = int(maskRequiredRowsCoveredByAnyPermutationSpace.sum())

	beansOrCornbread: Callable[[PermutationSpace], bool] = beansWithoutCornbread(state)
	countBeansWithoutCornbread: int = len(list(filter(beansOrCornbread, state.listPermutationSpace)))

	assert requiredRowsCoveredTotal == requiredRowsTotal, f"{functionName} expected all required rows to be covered for {mapShape=}, got requiredRowsCoveredTotal={requiredRowsCoveredTotal}, requiredRowsTotal={requiredRowsTotal}."
	assert countOverlappingDictionaries == 0, f"{functionName} expected no overlapping dictionaries for {mapShape=}, got countOverlappingDictionaries={countOverlappingDictionaries}."
	assert countBeansWithoutCornbread == 0, f"{functionName} expected 0 permutationSpace dictionaries with beans but no cornbread for {mapShape=}, got countBeansWithoutCornbread={countBeansWithoutCornbread}."

