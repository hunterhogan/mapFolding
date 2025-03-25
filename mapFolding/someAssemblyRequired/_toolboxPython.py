# Would `LibCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7

from collections.abc import Callable, Sequence
from inspect import getsource as inspect_getsource
from mapFolding.someAssemblyRequired import ast_Identifier, nameDOTname, nodeType
from os import PathLike
from pathlib import Path, PurePath
from types import ModuleType
from typing import Any, Generic, Literal, TypeGuard, cast
import ast
import importlib
import importlib.util

class NodeTourist(Generic[nodeType], ast.NodeVisitor):
	def __init__(self, findThis: Callable[[ast.AST], TypeGuard[nodeType] | bool], doThat):
		self.findThis = findThis
		self.doThat = doThat
		self.nodeCaptured = None

	def visit(self, node):
		if self.findThis(node):
			nodeActionOutput = self.doThat(node)
			if nodeActionOutput is not None:
				self.nodeCaptured = nodeActionOutput
		self.generic_visit(node)

	def captureFirstMatch(self, node):
		self.nodeCaptured = None
		self.visit(node)
		return self.nodeCaptured

class NodeChanger(Generic[nodeType], ast.NodeTransformer):
	def __init__(self, findThis: Callable[[ast.AST], TypeGuard[nodeType] | bool], doThat: Callable[[nodeType], ast.AST | Sequence[ast.AST] | None]) -> None:
		self.findThis = findThis
		self.doThat = doThat

	def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None:
		if self.findThis(node):
			return self.doThat(cast(nodeType, node))
		return super().visit(node)

def importLogicalPath2Callable(logicalPathModule: nameDOTname, identifier: ast_Identifier, packageIdentifierIfRelative: ast_Identifier | None = None):
	moduleImported: ModuleType = importlib.import_module(logicalPathModule, packageIdentifierIfRelative)
	return getattr(moduleImported, identifier)

def importPathFilename2Callable(pathFilename: PathLike[Any] | PurePath, identifier: ast_Identifier, moduleIdentifier: ast_Identifier | None = None) -> Callable[..., Any]:
	pathFilename = Path(pathFilename)

	importlibSpecification = importlib.util.spec_from_file_location(moduleIdentifier or pathFilename.stem, pathFilename)
	if importlibSpecification is None or importlibSpecification.loader is None: raise ImportError(f"I received\n\t`{pathFilename = }`,\n\t`{identifier = }`, and\n\t`{moduleIdentifier = }`.\n\tAfter loading, \n\t`importlibSpecification` {'is `None`' if importlibSpecification is None else 'has a value'} and\n\t`importlibSpecification.loader` {'is `None`' if importlibSpecification.loader is None else 'has a value'}.") # type: ignore [union-attr]

	moduleImported_jk_hahaha: ModuleType = importlib.util.module_from_spec(importlibSpecification)
	importlibSpecification.loader.exec_module(moduleImported_jk_hahaha)
	return getattr(moduleImported_jk_hahaha, identifier)

def parseLogicalPath2astModule(logicalPathModule: nameDOTname, packageIdentifierIfRelative: ast_Identifier|None=None, mode:str='exec', optimize:Literal[-1,0,1,2]=2) -> ast.AST:
	moduleImported: ModuleType = importlib.import_module(logicalPathModule, packageIdentifierIfRelative)
	sourcePython: str = inspect_getsource(moduleImported)
	return ast.parse(sourcePython, mode=mode, optimize=optimize)

def parsePathFilename2astModule(pathFilename: PathLike[Any] | PurePath, mode:str='exec', optimize:Literal[-1,0,1,2]=2) -> ast.AST:
	return ast.parse(Path(pathFilename).read_text(), mode=mode, optimize=optimize)
