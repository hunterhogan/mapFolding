"""Map folding, meanders, stamp folding, semi-meanders. Experiment with algorithm transformations, and analyze computational states."""

# isort: split
from mapFolding._semiotics import (
	ansiColorBlackOnCyan as ansiColorBlackOnCyan, ansiColorBlackOnMagenta as ansiColorBlackOnMagenta,
	ansiColorBlackOnWhite as ansiColorBlackOnWhite, ansiColorBlackOnYellow as ansiColorBlackOnYellow,
	ansiColorBlueOnWhite as ansiColorBlueOnWhite, ansiColorBlueOnYellow as ansiColorBlueOnYellow,
	ansiColorCyanOnBlack as ansiColorCyanOnBlack, ansiColorCyanOnBlue as ansiColorCyanOnBlue,
	ansiColorCyanOnMagenta as ansiColorCyanOnMagenta, ansiColorGreenOnBlack as ansiColorGreenOnBlack,
	ansiColorMagentaOnBlack as ansiColorMagentaOnBlack, ansiColorMagentaOnBlue as ansiColorMagentaOnBlue,
	ansiColorMagentaOnCyan as ansiColorMagentaOnCyan, ansiColorRedOnWhite as ansiColorRedOnWhite,
	ansiColorReset as ansiColorReset, ansiColors as ansiColors, ansiColorWhiteOnBlack as ansiColorWhiteOnBlack,
	ansiColorWhiteOnBlue as ansiColorWhiteOnBlue, ansiColorWhiteOnMagenta as ansiColorWhiteOnMagenta,
	ansiColorWhiteOnRed as ansiColorWhiteOnRed, ansiColorYellowOnBlack as ansiColorYellowOnBlack,
	ansiColorYellowOnBlue as ansiColorYellowOnBlue, ansiColorYellowOnRed as ansiColorYellowOnRed, decreasing as decreasing,
	inclusive as inclusive)

# isort: split
from mapFolding._theTypes import (
	Array1DElephino as Array1DElephino, Array1DFoldsTotal as Array1DFoldsTotal, Array1DLeavesTotal as Array1DLeavesTotal,
	Array2DLeavesTotal as Array2DLeavesTotal, Array3DLeavesTotal as Array3DLeavesTotal, axisOfLength as axisOfLength,
	DatatypeElephino as DatatypeElephino, DatatypeFoldsTotal as DatatypeFoldsTotal,
	DatatypeLeavesTotal as DatatypeLeavesTotal, MetadataOEISid as MetadataOEISid,
	MetadataOEISidManuallySet as MetadataOEISidManuallySet, MetadataOEISidMapFolding as MetadataOEISidMapFolding,
	MetadataOEISidMapFoldingManuallySet as MetadataOEISidMapFoldingManuallySet, NumPyElephino as NumPyElephino,
	NumPyFoldsTotal as NumPyFoldsTotal, NumPyIntegerType as NumPyIntegerType, NumPyLeavesTotal as NumPyLeavesTotal,
	ShapeArray as ShapeArray, ShapeSlicer as ShapeSlicer)

# isort: split
from mapFolding._theSSOT import packageSettings as packageSettings

# isort: split
from mapFolding.beDRY import (
	defineProcessorLimit as defineProcessorLimit, getConnectionGraph as getConnectionGraph,
	getLeavesTotal as getLeavesTotal, getTaskDivisions as getTaskDivisions, makeDataContainer as makeDataContainer,
	validateListDimensions as validateListDimensions)

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
