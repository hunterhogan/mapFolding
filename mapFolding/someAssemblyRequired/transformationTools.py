"""
Tools for transforming Python code through abstract syntax tree (AST) manipulation.

This module provides a comprehensive set of utilities for programmatically analyzing,
transforming, and generating Python code through AST manipulation. It implements
a highly flexible framework that enables:

1. Precise identification of code patterns through composable predicates
2. Targeted modification of code structures while preserving semantics
3. Code generation with proper syntax and import management
4. Analysis of code dependencies and relationships
5. Clean transformation of one algorithmic implementation to another

The utilities are organized into several key components:
- Predicate factories (ifThis): Create composable functions for matching AST patterns
- Node transformers: Modify AST structures in targeted ways
- Code generation helpers (Make): Create well-formed AST nodes programmatically
- Import tracking: Maintain proper imports during code transformation
- Analysis tools: Extract and organize code information

While these tools were developed to transform the baseline algorithm into optimized formats,
they are designed as general-purpose utilities applicable to a wide range of code
transformation scenarios beyond the scope of this package.
"""
from collections.abc import Callable, Sequence
from copy import deepcopy
from inspect import getsource as inspect_getsource
from mapFolding.someAssemblyRequired import ast_Identifier, ifThis, nameDOTname, nodeType, Then
from os import PathLike
from pathlib import Path, PurePath
from types import ModuleType
from typing import Any, cast, Generic, TypeGuard
import ast
import importlib
import importlib.util

"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
Hunter: ^^ did you do that ^^ ? Are you sure? You just fixed some "Name" identifiers that should have been "_name" because the wrong case confused you.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a `TypeAlias` for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""

# Would `LibCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7

class NodeTourist(Generic[nodeType], ast.NodeVisitor):
	def __init__(self, findThis: Callable[[ast.AST], TypeGuard[nodeType] | bool], doThat):
		self.findThis = findThis
		self.doThat = doThat

	def visit(self, node):
		if self.findThis(node):
			self.doThat(node)
		self.generic_visit(node)

class NodeChanger(Generic[nodeType], ast.NodeTransformer):
	def __init__(self, findThis: Callable[[ast.AST], TypeGuard[nodeType] | bool], doThat: Callable[[nodeType], ast.AST | Sequence[ast.AST] | None]) -> None:
		self.findThis = findThis
		self.doThat = doThat

	def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None:
		if self.findThis(node):
			return self.doThat(cast(nodeType, node))
		return super().visit(node)

def importLogicalPath2Callable(logicalPathModule: nameDOTname, identifier: ast_Identifier, packageIdentifierIfRelative: ast_Identifier | None = None) -> Callable[..., Any]:
	moduleImported: ModuleType = importlib.import_module(logicalPathModule, packageIdentifierIfRelative)
	return getattr(moduleImported, identifier)

def importPathFilename2Callable(pathFilename: str | PathLike[Any] | PurePath, identifier: ast_Identifier, moduleIdentifier: ast_Identifier | None = None) -> Callable[..., Any]:
	pathFilename = Path(pathFilename)

	importlibSpecification = importlib.util.spec_from_file_location(moduleIdentifier or pathFilename.stem, pathFilename)
	if importlibSpecification is None or importlibSpecification.loader is None: raise ImportError(f"I received\n\t`{pathFilename = }`,\n\t`{identifier = }`, and\n\t`{moduleIdentifier = }`.\n\tAfter loading, \n\t`importlibSpecification` {'is `None`' if importlibSpecification is None else 'has a value'} and\n\t`importlibSpecification.loader` {'is `None`' if importlibSpecification.loader is None else 'has a value'}.") # type: ignore [union-attr]

	moduleImported_jk_hahaha: ModuleType = importlib.util.module_from_spec(importlibSpecification)
	importlibSpecification.loader.exec_module(moduleImported_jk_hahaha)
	return getattr(moduleImported_jk_hahaha, identifier)

def parseLogicalPath2astModule(logicalPathModule: nameDOTname, packageIdentifierIfRelative: ast_Identifier | None = None) -> ast.Module:
	moduleImported: ModuleType = importlib.import_module(logicalPathModule, packageIdentifierIfRelative)
	sourcePython: str = inspect_getsource(moduleImported)
	return ast.parse(sourcePython)

def parsePathFilename2astModule(pathFilename: str | PathLike[Any] | PurePath) -> ast.Module:
	return ast.parse(Path(pathFilename).read_text(encoding='utf-8'))

# END of acceptable classes and functions ======================================================

def makeDictionaryFunctionDef(module: ast.Module) -> dict[ast_Identifier, ast.FunctionDef]:
	dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = {}
	NodeTourist(ifThis.isFunctionDef, Then.updateThis(dictionaryFunctionDef)).visit(module)
	return dictionaryFunctionDef

dictionaryEstimates: dict[tuple[int, ...], int] = {
	(2,2,2,2,2,2,2,2): 362794844160000,
	(2,21): 1493028892051200,
	(3,15): 9842024675968800,
	(3,3,3,3): 85109616000000000000000000000000,
	(8,8): 129950723279272000,
}

# END of marginal classes and functions ======================================================

# Start of I HATE PROGRAMMING ==========================================================

def Z0Z_extractClassDef(module: ast.Module, identifier: ast_Identifier) -> ast.ClassDef | None:
	sherpa: list[ast.ClassDef] = []
	NodeTourist(ifThis.isClassDef_Identifier(identifier), Then.appendTo(sherpa)).visit(module)
	return sherpa[0] if sherpa else None

def Z0Z_extractFunctionDef(module: ast.Module, identifier: ast_Identifier) -> ast.FunctionDef | None:
	sherpa: list[ast.FunctionDef] = []
	NodeTourist(ifThis.isFunctionDef_Identifier(identifier), Then.appendTo(sherpa)).visit(module)
	return sherpa[0] if sherpa else None

def Z0Z_makeDictionaryReplacementStatements(module: ast.Module) -> dict[ast_Identifier, ast.stmt | list[ast.stmt]]:
	"""Return a dictionary of function names and their replacement statements."""
	dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = makeDictionaryFunctionDef(module)
	dictionaryReplacementStatements: dict[ast_Identifier, ast.stmt | list[ast.stmt]] = {}
	for name, astFunctionDef in dictionaryFunctionDef.items():
		if ifThis.onlyReturnAnyCompare(astFunctionDef):
			dictionaryReplacementStatements[name] = astFunctionDef.body[0].value # type: ignore
		elif ifThis.onlyReturnUnaryOp(astFunctionDef):
			dictionaryReplacementStatements[name] = astFunctionDef.body[0].value # type: ignore
		else:
			dictionaryReplacementStatements[name] = astFunctionDef.body[0:-1]
	return dictionaryReplacementStatements

def Z0Z_descendantContainsMatchingNode(node: ast.AST, predicateFunction: Callable[[ast.AST], bool]) -> bool:
	"""Return True if any descendant of the node (or the node itself) matches the predicateFunction."""
	matchFound = False

	class DescendantFinder(ast.NodeVisitor):
		def generic_visit(self, node: ast.AST) -> None:
			nonlocal matchFound
			if predicateFunction(node):
				matchFound = True
			else:
				super().generic_visit(node)

	DescendantFinder().visit(node)
	return matchFound

def Z0Z_executeActionUnlessDescendantMatches(exclusionPredicate: Callable[[ast.AST], bool], actionFunction: Callable[[ast.AST], None]) -> Callable[[ast.AST], None]:
	"""Return a new action that will execute actionFunction only if no descendant (or the node itself) matches exclusionPredicate."""
	def wrappedAction(node: ast.AST) -> None:
		if not Z0Z_descendantContainsMatchingNode(node, exclusionPredicate):
			actionFunction(node)
	return wrappedAction

def Z0Z_inlineThisFunctionWithTheseValues(astFunctionDef: ast.FunctionDef, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> ast.FunctionDef:
	class FunctionInliner(ast.NodeTransformer):
		def __init__(self, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> None:
			self.dictionaryReplacementStatements = dictionaryReplacementStatements

		def generic_visit(self, node: ast.AST) -> ast.AST:
			"""Visit all nodes and replace them if necessary."""
			return super().generic_visit(node)

		def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
				return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore[attr-defined]
			return node

		def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
				return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore[attr-defined]
			return node

		def visit_Call(self, node: ast.Call) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node):
				replacement = self.dictionaryReplacementStatements[node.func.id] # type: ignore[attr-defined]
				if not isinstance(replacement, list):
					return replacement
			return node

	keepGoing = True
	ImaInlineFunction = deepcopy(astFunctionDef)
	while keepGoing:
		ImaInlineFunction = deepcopy(astFunctionDef)
		FunctionInliner(deepcopy(dictionaryReplacementStatements)).visit(ImaInlineFunction)
		if ast.unparse(ImaInlineFunction) == ast.unparse(astFunctionDef):
			keepGoing = False
		else:
			astFunctionDef = deepcopy(ImaInlineFunction)
			ast.fix_missing_locations(astFunctionDef)
	return ImaInlineFunction

def Z0Z_lameFindReplace(astTree: ast.AST, mappingFindReplaceNodes: dict[ast.AST, ast.AST]) -> ast.AST:
	keepGoing = True
	newTree = deepcopy(astTree)

	while keepGoing:
		for nodeFind, nodeReplace in mappingFindReplaceNodes.items():
			NodeChanger(ifThis.Z0Z_unparseIs(nodeFind), Then.replaceWith(nodeReplace)).visit(newTree)

		if ast.unparse(newTree) == ast.unparse(astTree):
			keepGoing = False
		else:
			astTree = deepcopy(newTree)
	return newTree
