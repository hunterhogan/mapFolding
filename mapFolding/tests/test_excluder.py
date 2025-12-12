"""Tests for the excluder system and analysis tools.

This module tests the functionality of the excluder system, including:
1.  The logic for excluding leaves based on pinned piles (`_Z0Z_excludeThisLeaf`, `Z0Z_excluder`).
2.  The generation and analysis of exclusion data (`theExcluderBeast.py`).
3.  The transformation of indices to fraction/addend representations.

The tests use `pytest` fixtures and parametrization to ensure flexibility and coverage.
"""

from fractions import Fraction
from mapFolding._e.analysisPython import theExcluderBeast
from mapFolding._e.analysisPython.theExcluderBeast import (
	_getContiguousEndingAtNegativeOne, _getContiguousFromStart, expressIndexAsFractionAddend, FractionAddend,
	writeAggregatedExclusions, writeExclusionDataCollated, writeExclusionDictionaries)
from mapFolding._e.pinning2DnAnnex import _Z0Z_excludeThisLeaf, Z0Z_excluder
from mapFolding.dataBaskets import EliminationState
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas
import pytest

# ======= Logic Tests (Adapted from test_excluder_logic.py) =======

@pytest.mark.parametrize("dimensionsTotal, pile, leavesPinned, leafToCheck, expectedResult", [
	(6, 7, {12: 36}, 4, True),
], ids=["2d6_pile7_leaf4_pinned12_36"])
def test_Z0Z_excludeThisLeaf(dimensionsTotal: int, pile: int, leavesPinned: dict[int, int], leafToCheck: int, expectedResult: bool, monkeypatch: pytest.MonkeyPatch) -> None:
	"""Verify _Z0Z_excludeThisLeaf correctly identifies excluded leaves."""
	state = MagicMock(spec=EliminationState)
	state.dimensionsTotal = dimensionsTotal
	state.pile = pile
	state.leavesPinned = leavesPinned

	stubLookup: dict[int, dict[int, dict[int, list[int]]]] = {
		leafToCheck: {
			pile: {
				12: [36]
			}
		}
	}
	monkeypatch.setattr("mapFolding._e.pinning2DnAnnex.dictionary2d6LeafExcludedAtPileByPile", stubLookup)

	result = _Z0Z_excludeThisLeaf(state, leafToCheck)
	assert result == expectedResult

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
	monkeypatch.setattr("mapFolding._e.pinning2DnAnnex.dictionary2d6AtPileLeafExcludedByPile", stubLookup)

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

	pathsCreated = writeExclusionDataCollated(listDimensions=[5])

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

	stub_data = {
		stubLeaf.__name__: {
			stubLeaf.__name__: {
				stubLeaf.__name__: [(Fraction(0, 1), 0), (Fraction(0, 1), 1)]
			}
		}
	}
	monkeypatch.setattr(theExcluderBeast, "analyzeContiguousEndAbsolute", lambda *_: stub_data)
	monkeypatch.setattr(theExcluderBeast, "analyzeContiguousEndRelative", lambda *_: stub_data)
	monkeypatch.setattr(theExcluderBeast, "analyzeContiguousStartAbsolute", lambda *_: stub_data)
	monkeypatch.setattr(theExcluderBeast, "analyzeContiguousStartRelative", lambda *_: stub_data)
	monkeypatch.setattr(theExcluderBeast, "analyzeNonContiguousIndicesRelative", lambda *_: stub_data)

	listPathFilenames = writeAggregatedExclusions(path_tmpTesting)

	assert listPathFilenames
	for pathCreated in listPathFilenames:
		assert Path(pathCreated).exists()
		assert Path(pathCreated).parent == path_tmpTesting
		assert "dictionaryExclusions" in Path(pathCreated).read_text(encoding="utf-8")

def test_writeExclusionDictionaries_createsFile(path_tmpTesting: Path) -> None:
	"""Verify writeExclusionDictionaries creates the output file."""
	# This function calls loadAggregatedExclusions which looks for files in pathExclusionData.
	# We need to ensure there are files there or mock loadAggregatedExclusions.

	with patch("mapFolding._e.analysisPython.theExcluderBeast.pathExclusionData", path_tmpTesting), \
			patch("mapFolding._e.analysisPython.theExcluderBeast.loadAggregatedExclusions", return_value={}):
		# It also calls restructureAggregatedExclusionsForMapShape which needs to work with empty dict
			pathExclusionsFile: Path = path_tmpTesting / "_exclusions.py"
			pathCreated = writeExclusionDictionaries(pathExclusionsFile)

			assert Path(pathCreated).exists()
			assert Path(pathCreated).name == "_exclusions.py"

			content = Path(pathCreated).read_text(encoding="utf-8")
			assert "dictionary2d5LeafExcludedAtPileByPile" in content

