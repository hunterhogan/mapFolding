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

from astToolkit import (
	Be, DOT, identifierDotAttribute, IngredientsFunction, NodeTourist, parseLogicalPath2astModule, Then)
from autoflake import fix_code as autoflake_fix_code
from hunterMakesPy import raiseIfNone, writeStringToHere
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import identifierModuleSourceAlgorithmDEFAULT, logicalPathInfixDEFAULT
from os import PathLike
from pathlib import PurePath
from typing import Any
import ast
import io

def findDataclass(ingredientsFunction: IngredientsFunction) -> tuple[str, str, str]:
	"""Extract dataclass information from a function's AST for transformation operations.

	(AI generated docstring)

	Analyzes the first parameter of a function to identify the dataclass type annotation
	and instance identifier, then locates the module where the dataclass is defined by
	examining the function's import statements. This information is essential for
	dataclass decomposition and transformation operations.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container with AST and import information.

	Returns
	-------
	dataclassLogicalPathModule : str
		Module logical path where the dataclass is defined.
	dataclassIdentifier : str
		Class name of the dataclass.
	dataclassInstanceIdentifier : str
		Parameter name for the dataclass instance.

	Raises
	------
	ValueError
		If dataclass information cannot be extracted from the function.

	"""
	dataclassName: ast.expr = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef))
	dataclassIdentifier: str = raiseIfNone(NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName))
	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in ingredientsFunction.imports._dictionaryImportFrom.items():  # noqa: SLF001
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclassIdentifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	return raiseIfNone(dataclassLogicalPathModule), dataclassIdentifier, dataclassInstanceIdentifier

def getLogicalPath(identifierPackage: str | None = None, logicalPathInfix: str | None = None, *moduleIdentifier: str | None) -> identifierDotAttribute:
	"""Get logical path from components."""
	listLogicalPathParts: list[str] = []
	if identifierPackage:
		listLogicalPathParts.append(identifierPackage)
	if logicalPathInfix:
		listLogicalPathParts.append(logicalPathInfix)
	if moduleIdentifier:
		listLogicalPathParts.extend([module for module in moduleIdentifier if module is not None])
	return '.'.join(listLogicalPathParts)

def getModule(identifierPackage: str | None = packageSettings.identifierPackage, logicalPathInfix: str | None = logicalPathInfixDEFAULT, moduleIdentifier: str | None = identifierModuleSourceAlgorithmDEFAULT) -> ast.Module:
	"""Get Module."""
	logicalPathSourceModule: identifierDotAttribute = getLogicalPath(identifierPackage, logicalPathInfix, moduleIdentifier)
	astModule: ast.Module = parseLogicalPath2astModule(logicalPathSourceModule)
	return astModule

def getPathFilename(pathRoot: PathLike[str] | PurePath | None = packageSettings.pathPackage, logicalPathInfix: PathLike[str] | PurePath | str | None = None, moduleIdentifier: str = '', fileExtension: str = packageSettings.fileExtension) -> PurePath:
	"""Construct filesystem path from logical path.

	Parameters
	----------
	pathRoot : PathLike[str] | PurePath | None = packageSettings.pathPackage
		Base directory for the package structure.
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
		Subdirectory for organizing generated modules.
	moduleIdentifier : str = ''
		Name of the specific module file.
	fileExtension : str = packageSettings.fileExtension
		File extension for Python modules.

	Returns
	-------
	pathFilename : PurePath
		Complete filesystem path for the generated module file.

	"""
	pathFilename = PurePath(moduleIdentifier + fileExtension)
	if logicalPathInfix:
		pathFilename = PurePath(*(str(logicalPathInfix).split('.')), pathFilename)
	if pathRoot:
		pathFilename = PurePath(pathRoot, pathFilename)
	return pathFilename

def write_astModule(astModule: ast.Module, pathFilename: PathLike[Any] | PurePath | io.TextIOBase, packageName: str | None = None) -> None:
	"""Prototype.

	Parameters
	----------
	astModule : ast.Module
		The AST module to be written to a file.
	pathFilename : PathLike[Any] | PurePath
		The file path where the module should be written.
	packageName : str | None = None
		Optional package name to preserve in import optimization.
	"""
	ast.fix_missing_locations(astModule)
	pythonSource: str = ast.unparse(astModule)
	autoflake_additional_imports: list[str] = []
	if packageName:
		autoflake_additional_imports.append(packageName)
	pythonSource = autoflake_fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=True, remove_duplicate_keys = False, remove_unused_variables = False)
	writeStringToHere(pythonSource, pathFilename)

