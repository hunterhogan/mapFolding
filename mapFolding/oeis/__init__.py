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
	getTriangleCells
		Retrieve the known cells of a triangle keyed by row and column.
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
	printEasyRunTriangle
		Compare a computed row with known cells and align differing rows.
	substituteValuesKnown
		Substitute the known value of each sequence value and triangle cell in a symbolic expression.
	updateTriangle
		Add triangle rows to a local b-file.
	updateValuesKnown
		Add missing terms to a local b-file and refresh shared known values.
	updateValuesSolved
		Add each term that the symbolic relations determine to the terms b-files.
	Z0Z_oeisName
		Look up the OEIS ID of an OEIS ID or common name, ignoring case, whitespace, and punctuation.

Variables
	boxOfOEISidTriangle
		Identify every implemented triangle.
	settingsOEIS
		Access OEIS package settings, paths, sequence identifiers, and symbolic configuration.

Types
	MetadataOEISid
		Describe the metadata that `getMetadata` returns.
	OEISid
		Identify an integer sequence with an OEIS label.

Modules
-------
byFormula
	Compute sequence terms from related sequences using the generated formula module.
"""
from __future__ import annotations

from mapFolding.oeis._beDRY import format_bFile as format_bFile
from mapFolding.oeis._dataBaskets import MetadataOEISid as MetadataOEISid
from mapFolding.oeis._theSSOT import settingsOEIS as settingsOEIS
from mapFolding.oeis._theTypes import OEISid as OEISid

# isort: split
from mapFolding.oeis._metadata import (
	boxOfOEISidTriangle as boxOfOEISidTriangle, getMetadata as getMetadata, getTriangle as getTriangle, getTriangleCells as getTriangleCells,
	getTriangleDiagonal as getTriangleDiagonal, getValuesKnown as getValuesKnown, updateTriangle as updateTriangle,
	updateValuesKnown as updateValuesKnown, Z0Z_oeisName as Z0Z_oeisName)

# isort: split
from mapFolding.oeis._benchmark import printEasyRunBenchmark as printEasyRunBenchmark, printEasyRunTriangle as printEasyRunTriangle
from mapFolding.oeis._mapShape import getTotalFoldsKnown as getTotalFoldsKnown, makeMapShape as makeMapShape

# isort: split
from mapFolding.oeis._byID import oeisIDfor_n as oeisIDfor_n
from mapFolding.oeis._symbolic import substituteValuesKnown as substituteValuesKnown, updateValuesSolved as updateValuesSolved
from mapFolding.oeis.A400429 import (
	A005315of0 as A005315of0, A005315of1 as A005315of1, A005315of2 as A005315of2, A005315of3 as A005315of3, A005315of4 as A005315of4,
	A005315of5 as A005315of5, A005315of6 as A005315of6, A005315of7 as A005315of7, A005315of8 as A005315of8, A005315of9 as A005315of9,
	A005315of10 as A005315of10, A005315of11 as A005315of11, boxOfA005315of as boxOfA005315of, boxOfDiagonals as boxOfDiagonals,
	diagonal1 as diagonal1, diagonal2 as diagonal2, diagonal3 as diagonal3, diagonal4 as diagonal4, diagonal5 as diagonal5,
	diagonal6 as diagonal6, diagonal7 as diagonal7, diagonal8 as diagonal8, diagonal9 as diagonal9, diagonal10 as diagonal10,
	diagonal11 as diagonal11)

# isort: split
from mapFolding.oeis._oeisdataClone import update_oeisdata as update_oeisdata

# isort: split
from mapFolding.oeis._commandLine import getOEISids as getOEISids, OEIS_for_n as OEIS_for_n
