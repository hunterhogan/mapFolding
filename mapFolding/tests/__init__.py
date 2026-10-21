"""Test suite for the mapFolding package.

This test suite provides comprehensive validation of map folding computations,
file system operations, OEIS integration, task division, and foundational
utilities. The tests are designed to support multiple audiences and use cases.

Test Module Organization (in mapFolding/tests/):
- conftest.py: Testing infrastructure and shared fixtures
- test_algorithms.py: Core mathematical validation and algorithm testing
- test_computations.py: Elimination flows and counts after pinning
- test_dataBaskets.py: Computation state and array-shape behavior
- test_dataDynamic.py: Leaf domains, pile choices, and map-derived lookups
- test_dataStructures.py: Dense and sparse triangle conversions and selections
- test_filesystem.py: File operations and path management
- test_insertion.py: Album insertion and constrained search-space construction
- test_measurements.py: Positional-numeral measurements of leaf and pile labels
- test_oeis.py: OEIS sequence access, triangle layouts, and local b-file updates
- test_oeis_byFormula.py: Formula selection and sequence relationships
- test_parameterValidation.py: Input and processor-limit validation
- test_pinning.py: Pinning and leaf-order restrictions
- test_semiotics.py: Dimension-position ideographs and head-relative labels
- test_taskDivisions.py: Task division and work distribution
- testsNeedAHome/test_generatingFunction.py: Generating-function research checks

For Contributors:
The test suite follows Domain-Driven Design principles, organizing tests around
mathematical concepts rather than implementation details. Use the existing
patterns as templates when adding new functionality.

For Users Adding Custom Modules:
The test_algorithms.py module provides the most relevant examples for testing
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
