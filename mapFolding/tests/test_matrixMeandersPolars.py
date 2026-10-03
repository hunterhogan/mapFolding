from __future__ import annotations

from mapFolding.algorithms.matrixMeandersPolars import alignArcs, doTheNeedful, transition
from mapFolding.synthesized.matrixMeanders.polarsWide import integersWidePolars吗
from mapFolding.tests.dataSamples.meandersData import (
	dataframeAligned8, dataframeAligned128, dataframeClosures8, dataframeClosures128, dataframePacked6, dataframePacked14, dataframePacked30,
	dataframePacked62, dataframePacked126, dataframePackedMixed6, dataframeTransition8, dataframeTransition16, dataframeTransition32,
	dataframeTransition64, dataframeTransition128, dataframeTransitionMixed6, dataframeUnsigned8Empty)
from polars.testing import assert_frame_equal
from typing import TYPE_CHECKING
import polars
import pytest

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateMeanders
	from pathlib import Path

@pytest.mark.parametrize(
	'dataframeMeanders, bitWidth, datatypeArcCode, expected'
	, (
	pytest.param(dataframeClosures8.lazy(), 8, polars.UInt8(), dataframeAligned8, id='UInt8-prefixes')
	, pytest.param(dataframeUnsigned8Empty.lazy(), 2, polars.UInt8(), dataframeUnsigned8Empty, id='two-bit-boundary')
		, pytest.param(dataframeClosures128.lazy(), 128, polars.UInt128(), dataframeAligned128, id='UInt128-last-bit')
		, pytest.param(dataframeUnsigned8Empty.lazy(), 8, polars.UInt8(), dataframeUnsigned8Empty, id='empty')
	)
)
def test_alignArcs(dataframeMeanders: polars.LazyFrame, bitWidth: int, datatypeArcCode: polars.DataType, expected: polars.DataFrame) -> None:
	assert_frame_equal(alignArcs(dataframeMeanders, bitWidth, datatypeArcCode).collect(engine='streaming'), expected)

@pytest.mark.parametrize(
	'dataframeMeanders, bitsLocator, arcCodeMAXIMUM, bitWidth, expected'
	, (
		pytest.param(dataframePacked6.lazy(), 0x15, 1 << 256, 6, dataframeTransition8, id='UInt8')
		, pytest.param(dataframePacked14.lazy(), 0x1555, 1 << 256, 14, dataframeTransition16, id='UInt16')
		, pytest.param(dataframePacked30.lazy(), 0x15555555, 1 << 256, 30, dataframeTransition32, id='UInt32')
		, pytest.param(dataframePacked62.lazy(), 0x1555555555555555, 1 << 256, 62, dataframeTransition64, id='UInt64')
		, pytest.param(dataframePacked126.lazy(), 0x15555555555555555555555555555555, 1 << 256, 126, dataframeTransition128, id='UInt128')
		, pytest.param(dataframePacked6.lazy(), 0x15, 256, 6, dataframeTransition8, id='exclusive-dtype-bound')
		, pytest.param(dataframePackedMixed6.lazy(), 0x15, 16, 6, dataframeTransitionMixed6, id='prune-before-shift')
		, pytest.param(dataframePacked6.lazy(), 0x15, 16, 6, dataframeUnsigned8Empty, id='no-completions')
	)
)
def test_transition(dataframeMeanders: polars.LazyFrame, bitsLocator: int, arcCodeMAXIMUM: int, bitWidth: int, expected: polars.DataFrame) -> None:
	assert_frame_equal(transition(dataframeMeanders, bitsLocator, arcCodeMAXIMUM, bitWidth).collect(engine='streaming'), expected, check_row_order=False)

@pytest.mark.parametrize(
	'state, expected'
	, (
		pytest.param((3, 'meanders', {15: (1 << 8) - 1}), {15: 510}, id='UInt8-count')
		, pytest.param((3, 'meanders', {15: (1 << 16) - 1}), {15: 131070}, id='UInt16-count')
		, pytest.param((3, 'meanders', {15: (1 << 32) - 1}), {15: 8589934590}, id='UInt32-count')
		, pytest.param((3, 'meanders', {15: (1 << 64) - 1}), {15: 36893488147419103230}, id='UInt64-count')
		, pytest.param((3, 'meanders', {15: (1 << 128) - 1}), {15: 680564733841876926926749214863536422910}, id='big-integer-count')
	)
	, indirect=['state']
)
def test_doTheNeedful(state: StateMeanders, expected: dict[int, int], tmp_path: Path) -> None:
	state = doTheNeedful(state)
	assert state.boundary == 0, f'{state.boundary=}'
	assert state.lookupMeanders == expected, f'{state.lookupMeanders=}, {expected=}'
	assert tuple(tmp_path.iterdir()) == (), f'{tuple(tmp_path.iterdir())=}'

@pytest.mark.parametrize('state, meandersMaximum, expected', (
	pytest.param((3, 'meanders', {15: 7}), 1 << 125, False, id='128-bit-subtotal')
	, pytest.param((3, 'meanders', {15: 7}), 1 << 126, True, id='129-bit-subtotal')
	, pytest.param((65, 'semi', {(1 << 126) - 1: 7}), 7, False, id='128-bit-transition')
	, pytest.param((65, 'semi', {(1 << 127) - 1: 7}), 7, True, id='129-bit-transition')
	, pytest.param((3, 'meanders', {15: 7}), None, False, id='dictionary-subtotal')
	, pytest.param((3, 'meanders', {15: 1 << 126}), None, True, id='dictionary-wide-subtotal')
), indirect=['state'])
def test_integersWidePolars吗(state: StateMeanders, meandersMaximum: int | None, expected: bool) -> None:
	assert integersWidePolars吗(state, meandersMaximum) == expected, f'{meandersMaximum=}, {expected=}'
