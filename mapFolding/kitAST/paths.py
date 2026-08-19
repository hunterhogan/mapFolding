from __future__ import annotations

from astToolkit import identifierDotAttribute, parseLogicalPath2astModule
from humpy_cytoolz import get_in
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST.theSSOT import default
from mapFolding.theSSOT import settingsPackage
from pathlib import PurePath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import Default
	from os import PathLike
	import ast

def getLogicalPath(identifierPackage: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, *identifierModule: str) -> identifierDotAttribute:
	"""Get logical path from components."""
	return '.'.join(filter(None, [identifierPackage, logicalPathInfix, *identifierModule]))

def getModule(
	identifierModule: str | None = None
	, logicalPathInfix: identifierDotAttribute | None = None
	, identifierPackage: str | None = None
	, identifiers: Default | None = None
) -> ast.Module:
	"""Get Module."""
	identifierPackage = identifierPackage or get_in(('module', 'package'), identifiers or {}) or default['module']['package']
	logicalPathInfix = logicalPathInfix or get_in(('logicalPath', 'default'), identifiers or {}) or default['logicalPath']['default']
	identifierModule = identifierModule or get_in(('module', 'default'), identifiers or {}) or default['module']['default']
	return parseLogicalPath2astModule(getLogicalPath(identifierPackage, logicalPathInfix, raiseIfNone(identifierModule)))

def getPathFilename(
	pathRoot: PathLike[str] | None = settingsPackage.pathPackage
	, logicalPathInfix: identifierDotAttribute | None = None
	, identifierModule: str = ''
	, fileExtension: str = settingsPackage.fileExtension
) -> PurePath:
	"""Construct filesystem path from logical path.

	Parameters
	----------
	pathRoot : PathLike[str] | None = settingsPackage.pathPackage
		Base directory for the package structure.
	logicalPathInfix : identifierDotAttribute | None = None
		Logical path in dot notation.
	identifierModule : str = ''
		Name of the specific module file.
	fileExtension : str = settingsPackage.fileExtension
		File extension for Python modules.

	Returns
	-------
	pathFilename : PurePath
		Complete filesystem path for the generated module file.

	"""
	pathFilename = PurePath(identifierModule + fileExtension)
	if logicalPathInfix:
		pathFilename = PurePath(*(str(logicalPathInfix).split('.')), pathFilename)
	if pathRoot:
		pathFilename = PurePath(pathRoot, pathFilename)
	return pathFilename
