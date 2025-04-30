from typing import Any, TypeAlias
import sys

yourPythonIsOld: TypeAlias = Any
# ruff: noqa: E402

if sys.version_info >= (3, 11):
	from typing import TypedDict as TypedDict
	from typing import NotRequired as NotRequired
else:
	try:
		from typing_extensions import TypedDict as TypedDict
		from typing_extensions import NotRequired as NotRequired
	except Exception:
		TypedDict = dict[yourPythonIsOld, yourPythonIsOld]
		from collections.abc import Iterable
		NotRequired: TypeAlias = Iterable

from mapFolding.datatypes import (
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

from mapFolding.theSSOT import PackageSettings as PackageSettings, packageSettings as packageSettings

from mapFolding.beDRY import (
	ComputationState as ComputationState,
	getConnectionGraph as getConnectionGraph,
	getLeavesTotal as getLeavesTotal,
	getTaskDivisions as getTaskDivisions,
	makeDataContainer as makeDataContainer,
	outfitCountFolds as outfitCountFolds,
	setProcessorLimit as setProcessorLimit,
	validateListDimensions as validateListDimensions,
)

from mapFolding.dataBaskets import MapFoldingState as MapFoldingState

from mapFolding.infoBooth import (
	PackageInformation as PackageInformation,
	raiseIfNoneGitHubIssueNumber3 as raiseIfNoneGitHubIssueNumber3,
	The as The,
)

from mapFolding.theDao import (
	countInitialize as countInitialize,
	doTheNeedful as doTheNeedful,
)

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
