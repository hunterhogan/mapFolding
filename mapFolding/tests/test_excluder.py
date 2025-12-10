"""Tests for the excluder system and analysis tools.

This module tests the functionality of the excluder system, including:
1.  The logic for excluding leaves based on pinned piles (`_Z0Z_excludeThisLeaf`, `Z0Z_excluder`).
2.  The generation and analysis of exclusion data (`theExcluderBeast.py`).
3.  The transformation of indices to fraction/addend representations.

The tests use `pytest` fixtures and parametrization to ensure flexibility and coverage.
"""

from collections.abc import Callable
from fractions import Fraction
from mapFolding._e.analysisPython.excluderValidation import validateAnalysisMethod
from mapFolding._e.analysisPython.theExcluderBeast import (
	_getContiguousFromEnd, _getContiguousFromStart, analyzeContiguousEndAbsolute, analyzeContiguousEndRelative,
	analyzeContiguousStartAbsolute, analyzeContiguousStartRelative, analyzeNonContiguousIndicesRelative,
	expressIndexAsFractionAddend, writeAggregatedExclusions, writeExclusionDataCollated, writeExclusionDictionaries)
from mapFolding._e.pinning2DnAnnex import _Z0Z_excludeThisLeaf, Z0Z_excluder
from mapFolding.dataBaskets import EliminationState
from operator import neg, pos
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# ======= Logic Tests (Adapted from test_excluder_logic.py) =======

@pytest.mark.parametrize("dimensionsTotal, pile, pinnedLeaves, leafToCheck, expectedResult", [
	(6, 7, {12: 36}, 4, True),
], ids=["2d6_pile7_leaf4_pinned12_36"])
def test_Z0Z_excludeThisLeaf(dimensionsTotal: int, pile: int, pinnedLeaves: dict[int, int], leafToCheck: int, expectedResult: bool) -> None:
	"""Verify _Z0Z_excludeThisLeaf correctly identifies excluded leaves."""
	state = MagicMock(spec=EliminationState)
	state.dimensionsTotal = dimensionsTotal
	state.pile = pile
	state.pinnedLeaves = pinnedLeaves

	result = _Z0Z_excludeThisLeaf(state, leafToCheck)
	assert result == expectedResult

@pytest.mark.parametrize("dimensionsTotal, pileLast, pinnedLeaves, expectedResult", [
	(6, 99, {7: 4, 12: 36}, True),
], ids=["2d6_pileLast99_pinned7_4_12_36"])
def test_Z0Z_excluder(dimensionsTotal: int, pileLast: int, pinnedLeaves: dict[int, int], expectedResult: bool) -> None:
	"""Verify Z0Z_excluder correctly identifies invalid states."""
	state = MagicMock(spec=EliminationState)
	state.dimensionsTotal = dimensionsTotal
	state.pileLast = pileLast
	state.pinnedLeaves = pinnedLeaves

	result = Z0Z_excluder(state)
	assert result == expectedResult

# ======= Transformation Tests =======

@pytest.mark.parametrize("index, pilesTotal, expected", [
	(0, 10, (pos, Fraction(0, 1), 0)),
	(5, 10, (pos, Fraction(1, 2), 0)),
	(-1, 10, (neg, Fraction(0, 1), 1)),
], ids=lambda index: f"index={index}")
def test_expressIndexAsFractionAddend(index: int, pilesTotal: int, expected: tuple[Callable[[int], int], Fraction, int]) -> None:
	"""Verify index to fraction/addend conversion."""
	assert expressIndexAsFractionAddend(index, pilesTotal) == expected

# ======= Analysis Method Tests =======

def test_getContiguousFromStart() -> None:
	assert _getContiguousFromStart([0, 1, 2, 5]) == [0, 1, 2]
	assert _getContiguousFromStart([1, 2, 3]) == []
	assert _getContiguousFromStart([0]) == [] # Less than 2

def test_getContiguousFromEnd() -> None:
	assert _getContiguousFromEnd([0, 8, 9], 10) == [8, 9]
	assert _getContiguousFromEnd([0, 1, 2], 10) == []
	assert _getContiguousFromEnd([9], 10) == [] # Less than 2

# ======= File Generation Tests =======

def test_writeExclusionDataCollated_createsFile(path_tmpTesting: Path) -> None:
	"""Verify writeExclusionDataCollated creates the output file."""
	# Patch pathExclusionData to use the temp directory
	with patch("mapFolding._e.analysisPython.theExcluderBeast.pathExclusionData", path_tmpTesting):
		# Use a small dimension list for speed
		pathCreated = writeExclusionDataCollated(listDimensions=[4])

		assert Path(pathCreated).exists()
		assert Path(pathCreated).name == "collated.py"

		# Basic content check
		content = Path(pathCreated).read_text(encoding="utf-8")
		assert "dictionaryExclusionData" in content

def test_writeAggregatedExclusions_createsFiles(path_tmpTesting: Path) -> None:
	"""Verify writeAggregatedExclusions creates output files."""
	with patch("mapFolding._e.analysisPython.theExcluderBeast.pathExclusionData", path_tmpTesting):
		listPathFilenames = writeAggregatedExclusions([])

		for path in listPathFilenames:
			assert Path(path).exists()
			assert Path(path).parent == path_tmpTesting

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

