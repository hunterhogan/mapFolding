"""Centralized configuration and default values."""

from __future__ import annotations

from mapFolding.kitFilesystem import getPathRootJobDEFAULT
from mapFolding.theSSOT import settingsPackage
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from hunterMakesPy import identifierDotAttribute

class Default(TypedDict):
	"""Default values."""

	filesystem: dict[str, PurePosixPath]
	function: dict[str, str]
	logicalPath: dict[str, identifierDotAttribute]
	module: dict[str, str]
	variable: dict[str, str]

default = Default(
	filesystem={
		'jobModule': PurePosixPath(getPathRootJobDEFAULT())
		, 'sourcePackage': PurePosixPath(settingsPackage.pathPackage)
	}
	, function={
		'counting': 'count'
		, 'dispatcher': 'doTheNeedful'
	}
	, logicalPath={
		'algorithm': f'{settingsPackage.identifierPackage}._e.algorithms'
		, 'synthetic': f'{settingsPackage.identifierPackage}._e.syntheticModules'
	}
	, module={
		'algorithm': 'eliminationCrease'
		, 'dataBasket': 'dataBaskets'
	}
	, variable={
		'counting': 'groupsOfFolds'
		, 'stateDataclass': 'EliminationState'
		, 'stateInstance': 'state'
	}
)
