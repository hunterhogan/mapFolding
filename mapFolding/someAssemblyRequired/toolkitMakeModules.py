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

from astToolkit import Be, DOT, identifierDotAttribute, NodeTourist, parseLogicalPath2astModule, Then
from astToolkit.containers import IngredientsFunction
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default
from os import PathLike
from pathlib import PurePath
import ast

def findDataclass(ingredientsFunction: IngredientsFunction) -> tuple[identifierDotAttribute, str, str]:
	"""Dynamically extract information about a `dataclass`: the instance identifier, the identifier, and the logical path module.

	Like many things in the "IngredientsFunction/IngredientsModule" ecosystem, this has specific requirements.
	`ingredientsFunction` must have the dataclass as its first parameter. The `LedgerOfImports` in `ingredientsFunction` must have
	the import information for the dataclass. If you are not using `IngredientsFunction`, you can still use this function to get
	the information you want.

	```python
	from astToolkit import astModuleToIngredientsFunction

	tupleInformation = findDataclass(astModuleToIngredientsFunction(astAST, identifier))
	```

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container with AST and import information.

	Returns
	-------
	logicalPathDataclass : identifierDotAttribute
		Logical path from which the `dataclass` is imported, which might not be the real source of the `dataclass`.
	identifierDataclass : str
		Identifier of the `dataclass`.
	identifierDataclassInstance : str
		Identifier of the `dataclass` instance.
	"""
	dataclassName: ast.expr = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef))
	identifierDataclass: str = raiseIfNone(NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName))
	logicalPathDataclass = None
	for moduleWithLogicalPath, listNameTuples in ingredientsFunction.imports._dictionaryImportFrom.items():  # noqa: SLF001
		for nameTuple in listNameTuples:
			if nameTuple[0] == identifierDataclass:
				logicalPathDataclass = moduleWithLogicalPath
				break
		if logicalPathDataclass:
			break
	identifierDataclassInstance: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	return raiseIfNone(logicalPathDataclass), identifierDataclass, identifierDataclassInstance

def getLogicalPath(identifierPackage: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, *identifierModule: str | None) -> identifierDotAttribute:
	"""Get logical path from components."""
	listLogicalPathParts: list[str] = []
	if identifierPackage:
		listLogicalPathParts.append(identifierPackage)
	if logicalPathInfix:
		listLogicalPathParts.append(logicalPathInfix)
	if identifierModule:
		listLogicalPathParts.extend([module for module in identifierModule if module is not None])
	return '.'.join(listLogicalPathParts)

def getModule(identifierPackage: str | None = packageSettings.identifierPackage, logicalPathInfix: identifierDotAttribute | None = default['logicalPath']['synthetic'], identifierModule: str | None = default['module']['algorithm']) -> ast.Module:
	"""Get Module."""
	logicalPathSourceModule: identifierDotAttribute = getLogicalPath(identifierPackage, logicalPathInfix, identifierModule)
	astModule: ast.Module = parseLogicalPath2astModule(logicalPathSourceModule)
	return astModule

def getPathFilename(pathRoot: PathLike[str] | PurePath | None = packageSettings.pathPackage, logicalPathInfix: identifierDotAttribute | None = None, identifierModule: str = '', fileExtension: str = packageSettings.fileExtension) -> PurePath:
	"""Construct filesystem path from logical path.

	Parameters
	----------
	pathRoot : PathLike[str] | PurePath | None = packageSettings.pathPackage
		Base directory for the package structure.
	logicalPathInfix : identifierDotAttribute | None = None
		Logical path in dot notation.
	moduleIdentifier : str = ''
		Name of the specific module file.
	fileExtension : str = packageSettings.fileExtension
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
