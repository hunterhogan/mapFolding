"""Map folding, meanders, stamp folding, semi-meanders. Experiment with algorithm transformations, and analyze computational states."""

# isort: split
from mapFolding._semiotics import (
	ansiColorReset as ansiColorReset, ansiColors as ansiColors, decreasing as decreasing, inclusive as inclusive,
	zeroIndexed as zeroIndexed)

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
