"""Test suite for the mapFolding package.

This test suite provides comprehensive validation of map folding computations,
file system operations, OEIS integration, task division, and foundational
utilities. The tests are designed to support multiple audiences and use cases.

Test Module Organization (in mapFolding/tests/):
- conftest.py: Testing infrastructure and shared fixtures
- test_computations.py: Core mathematical validation and algorithm testing
- test_filesystem.py: File operations and path management
- test_oeis.py: OEIS sequence integration and caching
- test_other.py: Foundational utilities and data validation
- test_tasks.py: Task division and work distribution

For Contributors:
The test suite follows Domain-Driven Design principles, organizing tests around
mathematical concepts rather than implementation details. Use the existing
patterns as templates when adding new functionality.

For Users Adding Custom Modules:
The test_computations.py module provides the most relevant examples for testing
custom computational approaches. The standardized testing functions in conftest.py
ensure consistent error reporting across all tests.

For AI Assistants:
The testing framework emphasizes readable, predictable patterns that maintain
mathematical correctness while supporting code evolution and optimization.
"""
from __future__ import annotations

from mapFolding.tests.conftestAnnex import (
	assert_allclose as assert_allclose, assert_approx as assert_approx, assert_array_equal as assert_array_equal,
	assertEqualTo as assertEqualTo, messageTestFailure as messageTestFailure, messageTestFailure_ndarray as messageTestFailure_ndarray)
