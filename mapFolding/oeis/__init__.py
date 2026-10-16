"""Access OEIS sequence computations, known values, and triangle utilities.

You can use this package to calculate supported terms from the On-Line Encyclopedia of Integer
Sequences (OEIS) [1], inspect known values, and compare computed results with stored values.
The package exposes sequence dispatch, map-shape conversion, b-file formatting, triangle
lookups, and the `OEIS_for_n` and `getOEISids` console entry points.

Contents
--------
Functions
	formatBFile
		Serialize sequence indices and values as b-file text.
	getMetadata
		Retrieve a registered sequence's shared metadata dictionary.
	getOEISids
		Print registered sequence descriptions and usage examples.
	getTotalFoldsKnown
		Retrieve a known folding total for an exact map shape.
	getTriangleDiagonal
		Select a diagonal from known values using the metadata row layout.
	getTriangleRows
		Group known sequence values into numbered rows.
	getValuesKnown
		Retrieve shared known values or a sentinel dictionary for an unknown identifier.
	makeMapShape
		Convert a supported sequence identifier and index to map dimensions.
	OEIS_for_n
		Calculate a sequence term from command-line arguments.
	oeisIDfor_n
		Calculate a term using a counting algorithm or a selected formula.
	printEasyRunBenchmark
		Print a comparison between a computed term and a known value.
	printEasyRunHeader
		Print benchmark labels with reproducible terminal colors.

Variables
	oeisIDsImplemented
		List the sequence identifiers registered for metadata loading and sequence selection.
	oeisIDsMapFoldingImplemented
		List the sequence identifiers dispatched to map-folding algorithms.

Modules
-------
byFormula
	Compute sequence terms from related sequences using the generated formula module [2].

References
----------
[1] The On-Line Encyclopedia of Integer Sequences.
	https://oeis.org/
[2] `mapFolding.oeis.byFormula`
"""
from __future__ import annotations

from mapFolding.oeis._beDRY import formatBFile as formatBFile
from mapFolding.oeis._theSSOT import oeisIDsImplemented as oeisIDsImplemented, oeisIDsMapFoldingImplemented as oeisIDsMapFoldingImplemented

# isort: split
from mapFolding.oeis._metadata import (
	getMetadata as getMetadata, getTriangleDiagonal as getTriangleDiagonal, getTriangleIndices as getTriangleIndices,
	getTriangleRows as getTriangleRows, getTriangleTerms as getTriangleTerms, getValuesKnown as getValuesKnown, updateBFile as updateBFile,
	updateTriangle as updateTriangle)

# isort: split
from mapFolding.oeis.__main__ import (
	getTotalFoldsKnown as getTotalFoldsKnown, makeMapShape as makeMapShape, printEasyRunBenchmark as printEasyRunBenchmark,
	printEasyRunHeader as printEasyRunHeader, printEasyRunTriangle as printEasyRunTriangle)

# isort: split
from mapFolding.oeis._byID import oeisIDfor_n as oeisIDfor_n

# isort: split
from mapFolding.oeis._commandLine import getOEISids as getOEISids, OEIS_for_n as OEIS_for_n
