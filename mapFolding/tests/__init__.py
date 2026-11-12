"""Test suite for the mapFolding package.

Test package organization (in mapFolding/tests/):
- conftest.py: Fixtures: job directories, album directories, b-file directories, known values, the processor total, written
	files, Numba jobs, enumerated foldings, and expected values from OEIS
- conftestAnnex.py: Assertion helpers for mapFolding's own domain objects, re-exported here
- dataSamples/: Samples shared by several test modules, and facts about them
- test_A400429.py: Diagonals of the semi-meander triangle, the A005315 anchors, and rational-series coefficients
- test_A400429oeisByFormula.py: Sequence values from the cells of the semi-meander triangle, and the diagonal formulas
- test_basecamp.py: Counting and enumerating foldings and meanders with each algorithm
- test_beDRY.py: Map-shape validation, leaf totals, processor limits, and task divisions
- test_byFormula.py: Each formula in `mapFolding.oeis.byFormula`
- test_choicesLeaf.py: The leaves that can appear at each pile
- test_creases.py: The leaves connected to a leaf by a crease
- test_dataStructures.py: Triangle diagonals and flattening, and array shapes and slicers inside Numba-compiled code
- test_insertion.py: Albums of representative foldings from the insertion algorithm
- test_kitFilesystem.py: Filenames, paths, and reading and writing the package's files
- test_leafDomains.py: The piles at which a leaf, or a group of leaves, can appear
- test_makeJob.py: Generated Numba counting jobs
- test_measure.py: Positional-numeral measurements of leaf and pile labels
- test_oeis.py: OEIS sequence access, the command line, triangle layouts, and local b-file updates
- test_pinIt.py: Pinning functions partition the foldings of a map
- test_semiotics.py: Dimension-position ideographs and head-relative labels

Development tools that are not part of the suite:
- createDataSamples.py: Extracts combined leaf domains from the enumerated foldings into `dataSamples`
- verifyPinning.py: Reports how a pinned state covers the enumerated foldings

Each test module holds its own parametrization, imports its assertion helpers from `mapFolding.tests` and its shared samples
from `mapFolding.tests.dataSamples`, and receives the fixtures in `conftest.py` by name.
"""
from __future__ import annotations

from hunterMakesPy.tests import (
	assert_allclose as assert_allclose, assert_approx as assert_approx, assert_array_equal as assert_array_equal,
	assertEqualTo as assertEqualTo, messageTestFailure as messageTestFailure, messageTestFailure_ndarray as messageTestFailure_ndarray)
from mapFolding.tests.conftestAnnex import assertPermutationSpacesPartitionFoldings as assertPermutationSpacesPartitionFoldings
