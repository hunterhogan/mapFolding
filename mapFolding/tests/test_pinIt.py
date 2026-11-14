"""Pinning functions partition the foldings of a 2^n-dimensional map among permutation spaces."""
from __future__ import annotations

from functools import partial
from mapFolding.dataBaskets import StateElimination
from mapFolding.filters import beansWithoutCornbread
from mapFolding.pinIt import (
	pin3beans2, pinLeaf首零Plus零, pinLeavesDimension0, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension零,
	pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零, pin首beans)
from mapFolding.tests import assertEqualTo, assertPermutationSpacesPartitionFoldings
from mapFolding.tests.dataSamples.maps2上nDimensional import mapShape2上4Dimensional, mapShape2上5Dimensional, mapShape2上6Dimensional
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from numpy import uint8
	from numpy.typing import NDArray

@pytest.mark.parametrize('mapShape', (
	pytest.param(mapShape2上4Dimensional, id='mapShape2上4Dimensional')
	, pytest.param(mapShape2上5Dimensional, id='mapShape2上5Dimensional')
	, pytest.param(mapShape2上6Dimensional, id='mapShape2上6Dimensional')
))
@pytest.mark.parametrize('pin', (
	pytest.param(pinPilesAtEnds, id='pinPilesAtEnds')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=2), id='pinPilesAtEnds-pileDepth2')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=3), id='pinPilesAtEnds-pileDepth3')
	, pytest.param(pinPile零Ante首零, id='pinPile零Ante首零')
	, pytest.param(pinLeavesDimension0, id='pinLeavesDimension0')
	, pytest.param(pinLeaf首零Plus零, id='pinLeaf首零Plus零')
	, pytest.param(pinLeavesDimension零, id='pinLeavesDimension零')
	, pytest.param(pinLeavesDimension一, id='pinLeavesDimension一')
	, pytest.param(pinLeavesDimensions0零一, id='pinLeavesDimensions0零一')
	, pytest.param(pinLeavesDimension二, id='pinLeavesDimension二')
	, pytest.param(pinLeavesDimension首二, id='pinLeavesDimension首二')
	, pytest.param(pin3beans2, id='pin3beans2')
	, pytest.param(pin首beans, id='pin首beans')
))
def test_pinPartitionsFoldings(arrayFoldings: NDArray[uint8], pin: Callable[..., StateElimination], mapShape: tuple[int, ...]) -> None:
	"""Relation: every enumerated folding agrees with exactly one permutation space, and no permutation space has beans without cornbread.

	The worker count, `CPUlimit`, does not affect the permutation spaces.
	"""
	state: StateElimination = pin(StateElimination(mapShape), CPUlimit=0.75)
	assertPermutationSpacesPartitionFoldings(state.boxOfPermutationSpace, arrayFoldings, 'pin', mapShape)
	assertEqualTo(tuple(filter(partial(beansWithoutCornbread, state), state.boxOfPermutationSpace)), (), 'pin', mapShape)
