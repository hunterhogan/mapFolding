"""
Map folding AST transformation system: Comprehensive transformation orchestration and module generation.

This module provides the orchestration layer of the map folding AST transformation system,
implementing comprehensive tools that coordinate all transformation stages to generate optimized
implementations with diverse computational strategies and performance characteristics. Building
upon the foundational pattern recognition, structural decomposition, core transformation tools,
Numba integration, and configuration management established in previous layers, this module
executes complete transformation processes that convert high-level dataclass-based algorithms
into specialized variants optimized for specific execution contexts.

The transformation orchestration addresses the full spectrum of optimization requirements for
map folding computational research through systematic application of the complete transformation
toolkit. The comprehensive approach decomposes dataclass parameters into primitive values for
Numba compatibility while removing object-oriented overhead and preserving computational logic,
generates concurrent execution variants using ProcessPoolExecutor with task division and result
aggregation, creates dedicated modules for counting variable setup with transformed loop conditions,
and provides theorem-specific transformations with configurable optimization levels including
trimmed variants and Numba-accelerated implementations.

The orchestration process operates through systematic AST manipulation that analyzes source
algorithms to extract dataclass dependencies, transforms data access patterns, applies performance
optimizations, and generates specialized modules with consistent naming conventions and filesystem
organization. The comprehensive transformation process coordinates pattern recognition for structural
analysis, dataclass decomposition for parameter optimization, function transformation for signature
adaptation, Numba integration for compilation optimization, and configuration management for
systematic generation control.

Generated modules maintain algorithmic correctness while providing significant performance
improvements through just-in-time compilation, parallel execution, and optimized data structures
tailored for specific computational requirements essential to large-scale map folding research.
"""
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
