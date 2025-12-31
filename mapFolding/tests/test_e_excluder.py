"""Tests for the excluder system and analysis tools.

This module tests the functionality of the excluder system, including:
1.  The logic for excluding leaves based on pinned piles (`_Z0Z_excludeThisLeaf`, `Z0Z_excluder`).
2.  The generation and analysis of exclusion data (`theExcluderBeast.py`).
3.  The transformation of indices to fraction/addend representations.

The tests use `pytest` fixtures and parametrization to ensure flexibility and coverage.
"""

from fractions import Fraction
from mapFolding._e.Z0Z_analysisPython import theExcluderBeast
from mapFolding._e.Z0Z_analysisPython.theExcluderBeast import (
	_getContiguousEndingAtNegativeOne, _getContiguousFromStart, expressIndexAsFractionAddend, FractionAddend,
	writeAggregatedExclusions, writeExclusionDataCollated, writeExclusionDictionaries)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensionsAnnex import Z0Z_excluder
from pathlib import Path, PurePath
from typing import Any
from unittest.mock import MagicMock, patch
import pandas
import pytest

# ======= Logic Tests (Adapted from test_excluder_logic.py) =======

@pytest.mark.parametrize("dimensionsTotal, pileLast, leavesPinned, expectedResult", [
	(6, 99, {7: 4, 12: 36}, True),
], ids=["2d6_pileLast99_pinned7_4_12_36"])
def test_Z0Z_excluder(dimensionsTotal: int, pileLast: int, leavesPinned: dict[int, int], expectedResult: bool, monkeypatch: pytest.MonkeyPatch) -> None:
	"""Verify Z0Z_excluder correctly identifies invalid states."""
	state = MagicMock(spec=EliminationState)
	state.dimensionsTotal = dimensionsTotal
	state.pileLast = pileLast
	state.leavesPinned = leavesPinned

	stubLookup: dict[int, dict[int, dict[int, list[int]]]] = {
		7: {
			4: {
				12: [36]
			}
		}
	}
	monkeypatch.setattr("mapFolding._e.pin2上nDimensionsAnnex.dictionary2d6AtPileLeafExcludedByPile", stubLookup)

	result = Z0Z_excluder(state)
	assert result == expectedResult

# ======= Transformation Tests =======

@pytest.mark.parametrize("index, pilesTotal, denominators, expected", [
	(0, 10, (), (Fraction(0, 1), 0)),
	(5, 10, (2,), (Fraction(1, 2), 0)),
	(-1, 10, (), (Fraction(0, 1), -1)),
], ids=["index=0", "index=5", "index=-1"])
def test_expressIndexAsFractionAddend(index: int, pilesTotal: int, denominators: tuple[int, ...], expected: FractionAddend) -> None:
	"""Verify index to fraction/addend conversion."""
	assert expressIndexAsFractionAddend(index, pilesTotal, denominators) == expected

# ======= Analysis Method Tests =======

@pytest.mark.parametrize("indices, expected", [
	([0, 1, 2, 5], [0, 1, 2]),
	([1, 2, 3], []),
	([0], []),
], ids=["contiguous-run", "no-zero-start", "single-value"])
def test_getContiguousFromStart(indices: list[int], expected: list[int]) -> None:
	assert _getContiguousFromStart(indices) == expected

@pytest.mark.parametrize("offsets, expected", [
	([-4, -3, -1], []),
	([-3, -2, -1], [-3, -2, -1]),
	([-1], []),
], ids=["missing-terminal", "full-run", "single-value"])
def test_getContiguousEndingAtNegativeOne(offsets: list[int], expected: list[int]) -> None:
	assert _getContiguousEndingAtNegativeOne(offsets) == expected

# ======= File Generation Tests =======

def test_writeExclusionDataCollated_creates_files(path_tmpTesting: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	def stubLeafZero(dimensionsTotal: int) -> int:
		return 0

	def stubLeafOne(dimensionsTotal: int) -> int:
		return 1

	monkeypatch.setattr(theExcluderBeast, "functionsHeadDimensions", [stubLeafZero, stubLeafOne])
	monkeypatch.setattr(theExcluderBeast, "dictionaryFunctionsByName", {stubLeafZero.__name__: stubLeafZero, stubLeafOne.__name__: stubLeafOne})
	monkeypatch.setattr(theExcluderBeast, "pathExclusionData", path_tmpTesting)

	def stubLeafDomains(state: EliminationState) -> dict[int, range]:
		return {0: range(2), 1: range(2)}

	monkeypatch.setattr(theExcluderBeast, "getDictionaryLeafDomains", stubLeafDomains)

	def stubDataFrameFoldings(state: EliminationState) -> pandas.DataFrame:
		return pandas.DataFrame({0: [0, 1], 1: [1, 0]})

	monkeypatch.setattr(theExcluderBeast, "getDataFrameFoldings", stubDataFrameFoldings)

	pathsCreated: list[PurePath] = writeExclusionDataCollated(listDimensions=[5])

	assert pathsCreated
	for pathCreated in pathsCreated:
		assert Path(pathCreated).exists()
		assert Path(pathCreated).parent == path_tmpTesting
		assert "leafExcluderData" in Path(pathCreated).read_text(encoding="utf-8")

def test_writeAggregatedExclusions_creates_files(path_tmpTesting: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	def stubLeaf(dimensionsTotal: int) -> int:
		return 0

	monkeypatch.setattr(theExcluderBeast, "functionsHeadDimensions", [stubLeaf])
	monkeypatch.setattr(theExcluderBeast, "pathExclusionData", path_tmpTesting)

	stub_data: dict[str, dict[str, dict[str, list[tuple[Fraction, int]]]]] = {
		stubLeaf.__name__: {
			stubLeaf.__name__: {
				stubLeaf.__name__: [(Fraction(0, 1), 0), (Fraction(0, 1), 1)]
			}
		}
	}
	def stub_analyze(*args: Any, **kwargs: Any) -> dict[str, dict[str, dict[str, list[tuple[Fraction, int]]]]]:
		return stub_data

	monkeypatch.setattr(theExcluderBeast, "analyzeContiguousEndAbsolute", stub_analyze)
	monkeypatch.setattr(theExcluderBeast, "analyzeContiguousEndRelative", stub_analyze)
	monkeypatch.setattr(theExcluderBeast, "analyzeContiguousStartAbsolute", stub_analyze)
	monkeypatch.setattr(theExcluderBeast, "analyzeContiguousStartRelative", stub_analyze)
	monkeypatch.setattr(theExcluderBeast, "analyzeNonContiguousIndicesRelative", stub_analyze)

	listPathFilenames: list[PurePath] = writeAggregatedExclusions(path_tmpTesting)

	assert listPathFilenames
	for pathCreated in listPathFilenames:
		assert Path(pathCreated).exists()
		assert Path(pathCreated).parent == path_tmpTesting
		assert "dictionaryExclusions" in Path(pathCreated).read_text(encoding="utf-8")

def test_writeExclusionDictionaries_createsFile(path_tmpTesting: Path) -> None:
	"""Verify writeExclusionDictionaries creates the output file."""
	# This function calls loadAggregatedExclusions which looks for files in pathExclusionData.
	# We need to ensure there are files there or mock loadAggregatedExclusions.

	with patch("mapFolding._e.Z0Z_analysisPython.theExcluderBeast.pathExclusionData", path_tmpTesting), \
			patch("mapFolding._e.Z0Z_analysisPython.theExcluderBeast.loadAggregatedExclusions", return_value={}):
		# It also calls restructureAggregatedExclusionsForMapShape which needs to work with empty dict
			pathExclusionsFile: Path = path_tmpTesting / "_exclusions.py"
			pathCreated: PurePath = writeExclusionDictionaries(pathExclusionsFile)

			assert Path(pathCreated).exists()
			assert Path(pathCreated).name == "_exclusions.py"

			content = Path(pathCreated).read_text(encoding="utf-8")
			assert "dictionary2d5LeafExcludedAtPileByPile" in content

