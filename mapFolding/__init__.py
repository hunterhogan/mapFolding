"""Map folding, meanders, stamp folding, semi-meanders. Experiment with algorithm transformations, and analyze computational states."""

from mapFolding._semiotics import (
	asciiColorBackgroundBlack as asciiColorBackgroundBlack, asciiColorBackgroundBlue as asciiColorBackgroundBlue,
	asciiColorBackgroundCyan as asciiColorBackgroundCyan, asciiColorBackgroundGreen as asciiColorBackgroundGreen,
	asciiColorBackgroundMagenta as asciiColorBackgroundMagenta, asciiColorBackgroundRed as asciiColorBackgroundRed,
	asciiColorBackgroundWhite as asciiColorBackgroundWhite, asciiColorBackgroundYellow as asciiColorBackgroundYellow,
	asciiColorBlack as asciiColorBlack, asciiColorBlue as asciiColorBlue, asciiColorCyan as asciiColorCyan,
	asciiColorGreen as asciiColorGreen, asciiColorMagenta as asciiColorMagenta, asciiColorRed as asciiColorRed,
	asciiColorReset as asciiColorReset, asciiColorWhite as asciiColorWhite, asciiColorYellow as asciiColorYellow,
	decreasing as decreasing, inclusive as inclusive)

# isort: split
from mapFolding._theTypes import (
	Array1DElephino as Array1DElephino, Array1DFoldsTotal as Array1DFoldsTotal, Array1DLeavesTotal as Array1DLeavesTotal,
	Array2DLeavesTotal as Array2DLeavesTotal, Array3DLeavesTotal as Array3DLeavesTotal, axisOfLength as axisOfLength,
	DatatypeElephino as DatatypeElephino, DatatypeFoldsTotal as DatatypeFoldsTotal,
	DatatypeLeavesTotal as DatatypeLeavesTotal, LeafOrPileRangeOfLeaves as LeafOrPileRangeOfLeaves,
	MetadataOEISid as MetadataOEISid, MetadataOEISidManuallySet as MetadataOEISidManuallySet,
	MetadataOEISidMapFolding as MetadataOEISidMapFolding,
	MetadataOEISidMapFoldingManuallySet as MetadataOEISidMapFoldingManuallySet, NumPyElephino as NumPyElephino,
	NumPyFoldsTotal as NumPyFoldsTotal, NumPyIntegerType as NumPyIntegerType, NumPyLeavesTotal as NumPyLeavesTotal,
	PinnedLeaves as PinnedLeaves, ShapeArray as ShapeArray, ShapeSlicer as ShapeSlicer)

# isort: split
from mapFolding._theSSOT import packageSettings as packageSettings

# isort: split
from mapFolding.beDRY import (
	between as between, consecutive as consecutive, defineProcessorLimit as defineProcessorLimit, DOTvalues as DOTvalues,
	exclude as exclude, getConnectionGraph as getConnectionGraph, getLeavesTotal as getLeavesTotal,
	getTaskDivisions as getTaskDivisions, makeDataContainer as makeDataContainer, mappingHasKey as mappingHasKey,
	noDuplicates as noDuplicates, reverseLookup as reverseLookup, validateListDimensions as validateListDimensions)

# isort: split
from mapFolding.filesystemToolkit import (
	getFilenameFoldsTotal as getFilenameFoldsTotal, getPathFilenameFoldsTotal as getPathFilenameFoldsTotal,
	getPathRootJobDEFAULT as getPathRootJobDEFAULT, saveFoldsTotal as saveFoldsTotal,
	saveFoldsTotalFAILearly as saveFoldsTotalFAILearly)

# isort: split
from mapFolding.basecamp import countFolds as countFolds, eliminateFolds as eliminateFolds

# isort: split
from mapFolding.oeis import (
	dictionaryOEIS as dictionaryOEIS, dictionaryOEISMapFolding as dictionaryOEISMapFolding,
	getFoldsTotalKnown as getFoldsTotalKnown, getOEISids as getOEISids, OEIS_for_n as OEIS_for_n,
	oeisIDfor_n as oeisIDfor_n)
