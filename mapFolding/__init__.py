"""Computational toolkit for analyzing multi-dimensional map folding patterns.

(AI generated docstring)

The mapFolding package provides a complete implementation of Lunnon's 1971 algorithm
for counting distinct folding patterns in multi-dimensional maps. This toolkit
transforms the complex combinatorial mathematics of map folding into accessible
computational tools, enabling researchers and practitioners to analyze folding
patterns across dimensions from simple 2D strips to complex multi-dimensional
hypercubes.

The package architecture follows Domain-Driven Design principles, organizing
functionality around mathematical concepts rather than implementation details.
The computational framework integrates type safety, persistent result storage,
and mathematical validation through OEIS sequence integration.

Core Transformation Tools:
    countFolds: Primary interface for computing folding pattern counts
    MapFoldingState: Computational state management for recursive analysis
    Connection graph generation: Mathematical foundation for folding relationships
    Task division utilities: Experimental parallel computation options
    OEIS integration: Mathematical validation and sequence discovery

Primary Use Cases:
    Mathematical research into folding pattern properties and relationships
    Educational exploration of combinatorial mathematics concepts
    Computational validation of theoretical results
    Extension of known mathematical sequences through new discoveries

The package handles the full spectrum of map folding analysis, from simple
educational examples to research-grade computations requiring multi-day processing
time. Results integrate seamlessly with the mathematical community through
comprehensive OEIS connectivity and standardized result persistence.

For researchers: The computational foundation supports both replication of
established results and discovery of new mathematical relationships.

For educators: The clear interfaces and type safety enable confident exploration
of combinatorial concepts without computational complexity barriers.

For practitioners: The robust result persistence and type safety ensure
reliable completion of complex analytical tasks.
"""

from mapFolding._theTypes import (
	Array1DElephino as Array1DElephino,
	Array1DFoldsTotal as Array1DFoldsTotal,
	Array1DLeavesTotal as Array1DLeavesTotal,
	Array3D as Array3D,
	DatatypeElephino as DatatypeElephino,
	DatatypeFoldsTotal as DatatypeFoldsTotal,
	DatatypeLeavesTotal as DatatypeLeavesTotal,
	NumPyElephino as NumPyElephino,
	NumPyFoldsTotal as NumPyFoldsTotal,
	NumPyIntegerType as NumPyIntegerType,
	NumPyLeavesTotal as NumPyLeavesTotal,
)

from mapFolding._theSSOT import packageSettings as packageSettings

from mapFolding.beDRY import (
	getConnectionGraph as getConnectionGraph,
	getLeavesTotal as getLeavesTotal,
	getTaskDivisions as getTaskDivisions,
	makeDataContainer as makeDataContainer,
	setProcessorLimit as setProcessorLimit,
	validateListDimensions as validateListDimensions,
)

from mapFolding.dataBaskets import MapFoldingState as MapFoldingState

from mapFolding.filesystemToolkit import (
	getFilenameFoldsTotal as getFilenameFoldsTotal,
	getPathFilenameFoldsTotal as getPathFilenameFoldsTotal,
	getPathRootJobDEFAULT as getPathRootJobDEFAULT,
	saveFoldsTotal as saveFoldsTotal,
	saveFoldsTotalFAILearly as saveFoldsTotalFAILearly,
)

from mapFolding.basecamp import countFolds as countFolds

from mapFolding.oeis import (
	clearOEIScache as clearOEIScache,
	getFoldsTotalKnown as getFoldsTotalKnown,
	getOEISids as getOEISids,
	OEIS_for_n as OEIS_for_n,
	oeisIDfor_n as oeisIDfor_n,
)
