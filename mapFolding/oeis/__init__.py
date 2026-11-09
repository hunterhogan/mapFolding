"""Access OEIS sequence computations, known values, and triangle utilities.

(AI generated docstring)

You can use this package to calculate supported terms from the On-Line Encyclopedia of Integer
Sequences (OEIS), inspect known values, and compare computed results with stored values.
The package exposes sequence dispatch, map-shape conversion, b-file formatting, triangle
lookups, and the `OEIS_for_n` and `getOEISids` console entry points.

Contents
--------
Functions
	format_bFile
		Serialize sequence indices and values as b-file text.
	getMetadata
		Retrieve a registered sequence's shared metadata dictionary.
	getOEISids
		Print registered sequence descriptions and usage examples.
	getTotalFoldsKnown
		Retrieve a known folding total for an exact map shape.
	getTriangle
		Group the consecutive known values of a triangle into numbered rows.
	getTriangleDiagonal
		Select a diagonal of a triangle using the metadata row layout.
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
	printEasyRunTriangle
		Compare a computed row with known cells and align differing rows.
	updateTriangle
		Add triangle rows to a local b-file.
	updateValuesKnown
		Add missing terms to a local b-file and refresh shared known values.

Variables
	settingsOEIS
		Access OEIS package settings, paths, sequence identifiers, and symbolic configuration.

Modules
-------
byFormula
	Compute sequence terms from related sequences using the generated formula module.
"""
from __future__ import annotations

from mapFolding.oeis._beDRY import format_bFile as format_bFile
from mapFolding.oeis._theSSOT import settingsOEIS as settingsOEIS

# isort: split
from mapFolding.oeis._metadata import (
	getMetadata as getMetadata, getTriangle as getTriangle, getTriangleDiagonal as getTriangleDiagonal, getValuesKnown as getValuesKnown,
	updateTriangle as updateTriangle, updateValuesKnown as updateValuesKnown)

# isort: split
from mapFolding.oeis.__main__ import (
	getTotalFoldsKnown as getTotalFoldsKnown, makeMapShape as makeMapShape, printEasyRunBenchmark as printEasyRunBenchmark,
	printEasyRunHeader as printEasyRunHeader, printEasyRunTriangle as printEasyRunTriangle)

# isort: split
from mapFolding.oeis._byID import oeisIDfor_n as oeisIDfor_n

# isort: split
from mapFolding.oeis._commandLine import getOEISids as getOEISids, OEIS_for_n as OEIS_for_n
