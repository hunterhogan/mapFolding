"""Test suite for the mapFolding package.

Test package organization (in mapFolding/tests/):
- conftest.py: Fixtures: parameter dimensions, derived expected values, filesystem isolation, and environment control
- conftestAnnex.py: Assertion helpers, re-exported here
- dataSamples/: Static expected data and pytest-ready parameter rows
- test_algorithms.py: Counts, enumerations, insertion albums, and synthesized jobs from each algorithm
- test_dataDynamic.py: Leaf domains, pile choices, and creases of 2^n-dimensional maps
- test_dataStructures.py: Triangle diagonals and flattening, and array shapes and slicers inside Numba-compiled code
- test_filesystem.py: Filenames, paths, and reading and writing the package's files
- test_measurements.py: Positional-numeral measurements of leaf and pile labels
- test_oeis.py: OEIS sequence access, the command line, triangle layouts, and local b-file updates
- test_oeis_byFormula.py: Each formula in `mapFolding.oeis.byFormula`
- test_parameterValidation.py: Map-shape validation, leaf totals, and the `hunterMakesPy` parsers they rely on
- test_pinning.py: Pinning functions cover every folding exactly once
- test_semiotics.py: Dimension-position ideographs and head-relative labels
- test_taskDivisions.py: Processor limits and task division

Each test module imports its assertion helpers from `mapFolding.tests` and its static data from
`mapFolding.tests.dataSamples`; pytest supplies the fixtures in `conftest.py` by name.
"""
from __future__ import annotations

from mapFolding.tests.conftestAnnex import (
	assert_allclose as assert_allclose, assert_approx as assert_approx, assert_array_equal as assert_array_equal,
	assertEqualTo as assertEqualTo, messageTestFailure as messageTestFailure, messageTestFailure_ndarray as messageTestFailure_ndarray)
