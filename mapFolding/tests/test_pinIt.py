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

markDefectDimension4: pytest.MarkDecorator = pytest.mark.xfail(raises=AssertionError, strict=True
	, reason='The function pins from a four-dimensional combined domain that is a known defect in `mapFolding.leafDomains`.')

@pytest.mark.parametrize('pin, mapShape', (
	pytest.param(pinPilesAtEnds, mapShape2上4Dimensional, id='pinPilesAtEnds-mapShape2上4Dimensional')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=2), mapShape2上4Dimensional, id='pinPilesAtEnds-pileDepth2-mapShape2上4Dimensional')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=3), mapShape2上4Dimensional, id='pinPilesAtEnds-pileDepth3-mapShape2上4Dimensional')
	, pytest.param(pinPile零Ante首零, mapShape2上4Dimensional, id='pinPile零Ante首零-mapShape2上4Dimensional')
	, pytest.param(pinLeavesDimension0, mapShape2上4Dimensional, id='pinLeavesDimension0-mapShape2上4Dimensional')
	, pytest.param(pinLeaf首零Plus零, mapShape2上4Dimensional, id='pinLeaf首零Plus零-mapShape2上4Dimensional')
	, pytest.param(pinLeavesDimension零, mapShape2上4Dimensional, id='pinLeavesDimension零-mapShape2上4Dimensional')
	, pytest.param(pinLeavesDimension一, mapShape2上4Dimensional, id='pinLeavesDimension一-mapShape2上4Dimensional')
	, pytest.param(pinLeavesDimensions0零一, mapShape2上4Dimensional, id='pinLeavesDimensions0零一-mapShape2上4Dimensional')
	, pytest.param(pinLeavesDimension二, mapShape2上4Dimensional, marks=markDefectDimension4, id='pinLeavesDimension二-mapShape2上4Dimensional')
	, pytest.param(pinLeavesDimension首二, mapShape2上4Dimensional, marks=markDefectDimension4, id='pinLeavesDimension首二-mapShape2上4Dimensional')
	, pytest.param(pin3beans2, mapShape2上4Dimensional, id='pin3beans2-mapShape2上4Dimensional')
	, pytest.param(pin首beans, mapShape2上4Dimensional, id='pin首beans-mapShape2上4Dimensional')
	, pytest.param(pinPilesAtEnds, mapShape2上5Dimensional, id='pinPilesAtEnds-mapShape2上5Dimensional')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=2), mapShape2上5Dimensional, id='pinPilesAtEnds-pileDepth2-mapShape2上5Dimensional')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=3), mapShape2上5Dimensional, id='pinPilesAtEnds-pileDepth3-mapShape2上5Dimensional')
	, pytest.param(pinPile零Ante首零, mapShape2上5Dimensional, id='pinPile零Ante首零-mapShape2上5Dimensional')
	, pytest.param(pinLeavesDimension0, mapShape2上5Dimensional, id='pinLeavesDimension0-mapShape2上5Dimensional')
	, pytest.param(pinLeaf首零Plus零, mapShape2上5Dimensional, id='pinLeaf首零Plus零-mapShape2上5Dimensional')
	, pytest.param(pinLeavesDimension零, mapShape2上5Dimensional, id='pinLeavesDimension零-mapShape2上5Dimensional')
	, pytest.param(pinLeavesDimension一, mapShape2上5Dimensional, id='pinLeavesDimension一-mapShape2上5Dimensional')
	, pytest.param(pinLeavesDimensions0零一, mapShape2上5Dimensional, id='pinLeavesDimensions0零一-mapShape2上5Dimensional')
	, pytest.param(pinLeavesDimension二, mapShape2上5Dimensional, id='pinLeavesDimension二-mapShape2上5Dimensional')
	, pytest.param(pinLeavesDimension首二, mapShape2上5Dimensional, id='pinLeavesDimension首二-mapShape2上5Dimensional')
	, pytest.param(pin3beans2, mapShape2上5Dimensional, id='pin3beans2-mapShape2上5Dimensional')
	, pytest.param(pin首beans, mapShape2上5Dimensional, id='pin首beans-mapShape2上5Dimensional')
	, pytest.param(pinPilesAtEnds, mapShape2上6Dimensional, id='pinPilesAtEnds-mapShape2上6Dimensional')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=2), mapShape2上6Dimensional, id='pinPilesAtEnds-pileDepth2-mapShape2上6Dimensional')
	, pytest.param(partial(pinPilesAtEnds, pileDepth=3), mapShape2上6Dimensional, id='pinPilesAtEnds-pileDepth3-mapShape2上6Dimensional')
	, pytest.param(pinPile零Ante首零, mapShape2上6Dimensional, id='pinPile零Ante首零-mapShape2上6Dimensional')
	, pytest.param(pinLeavesDimension0, mapShape2上6Dimensional, id='pinLeavesDimension0-mapShape2上6Dimensional')
	, pytest.param(pinLeaf首零Plus零, mapShape2上6Dimensional, id='pinLeaf首零Plus零-mapShape2上6Dimensional')
	, pytest.param(pinLeavesDimension零, mapShape2上6Dimensional, id='pinLeavesDimension零-mapShape2上6Dimensional')
	, pytest.param(pinLeavesDimension一, mapShape2上6Dimensional, id='pinLeavesDimension一-mapShape2上6Dimensional')
	, pytest.param(pinLeavesDimensions0零一, mapShape2上6Dimensional, id='pinLeavesDimensions0零一-mapShape2上6Dimensional')
	, pytest.param(pinLeavesDimension二, mapShape2上6Dimensional, id='pinLeavesDimension二-mapShape2上6Dimensional')
	, pytest.param(pinLeavesDimension首二, mapShape2上6Dimensional, id='pinLeavesDimension首二-mapShape2上6Dimensional')
	, pytest.param(pin3beans2, mapShape2上6Dimensional, id='pin3beans2-mapShape2上6Dimensional')
	, pytest.param(pin首beans, mapShape2上6Dimensional, id='pin首beans-mapShape2上6Dimensional')
))
def test_pinPartitionsFoldings(arrayFoldings: NDArray[uint8], pin: Callable[..., StateElimination], mapShape: tuple[int, ...]) -> None:
	"""Relation: every enumerated folding agrees with exactly one permutation space, and no permutation space has beans without cornbread.

	The worker count, `CPUlimit`, does not affect the permutation spaces. The complete product of functions and maps is enumerated
	because the known defect marks only some combinations.
	"""
	state: StateElimination = pin(StateElimination(mapShape), CPUlimit=0.75)
	assertPermutationSpacesPartitionFoldings(state.boxOfPermutationSpace, arrayFoldings, 'pin', mapShape)
	assertEqualTo(tuple(filter(partial(beansWithoutCornbread, state), state.boxOfPermutationSpace)), (), 'pin', mapShape)
