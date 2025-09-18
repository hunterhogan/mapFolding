"""Map folding, meanders, stamp folding, semi-meanders. Experiment with algorithm transformations, and analyze computational states."""

from mapFolding._theTypes import (
	Array1DElephino as Array1DElephino,
	Array1DFoldsTotal as Array1DFoldsTotal,
	Array1DLeavesTotal as Array1DLeavesTotal,
	Array3DLeavesTotal as Array3DLeavesTotal,
	DatatypeElephino as DatatypeElephino,
	DatatypeFoldsTotal as DatatypeFoldsTotal,
	DatatypeLeavesTotal as DatatypeLeavesTotal,
	MetadataOEISidMapFolding as MetadataOEISidMapFolding,
	MetadataOEISidMapFoldingManuallySet as MetadataOEISidMapFoldingManuallySet,
	MetadataOEISidMeanders as MetadataOEISidMeanders,
	MetadataOEISidMeandersManuallySet as MetadataOEISidMeandersManuallySet,
	NumPyElephino as NumPyElephino,
	NumPyFoldsTotal as NumPyFoldsTotal,
	NumPyIntegerType as NumPyIntegerType,
	NumPyLeavesTotal as NumPyLeavesTotal)

from mapFolding._theSSOT import packageSettings as packageSettings

from mapFolding.beDRY import (
	getConnectionGraph as getConnectionGraph,
	getLeavesTotal as getLeavesTotal,
	getTaskDivisions as getTaskDivisions,
	makeDataContainer as makeDataContainer,
	setProcessorLimit as setProcessorLimit,
	validateListDimensions as validateListDimensions)

from mapFolding.dataBaskets import (
	ParallelMapFoldingState as ParallelMapFoldingState,
	MapFoldingState as MapFoldingState,
	MatrixMeandersNumPyState as MatrixMeandersNumPyState,
	MatrixMeandersState as MatrixMeandersState)

from mapFolding.filesystemToolkit import (
	getFilenameFoldsTotal as getFilenameFoldsTotal,
	getPathFilenameFoldsTotal as getPathFilenameFoldsTotal,
	getPathRootJobDEFAULT as getPathRootJobDEFAULT,
	saveFoldsTotal as saveFoldsTotal,
	saveFoldsTotalFAILearly as saveFoldsTotalFAILearly)

from mapFolding.basecamp import countFolds as countFolds

from mapFolding.oeis import (
	dictionaryOEISMapFolding as dictionaryOEISMapFolding,
	dictionaryOEISMeanders as dictionaryOEISMeanders,
	getFoldsTotalKnown as getFoldsTotalKnown,
	getOEISids as getOEISids,
	OEIS_for_n as OEIS_for_n,
	oeisIDfor_n as oeisIDfor_n)
