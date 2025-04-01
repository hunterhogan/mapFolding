from collections.abc import Callable
from inspect import getsource as inspect_getsource
from mapFolding.someAssemblyRequired import ast_Identifier, str_nameDOTname
from os import PathLike
from pathlib import Path, PurePath
from types import ModuleType
from typing import Any
import ast
import importlib
import importlib.util

# TODO Identify the logic that narrows the type and can help the user during static type checking.

class NodeTourist(ast.NodeVisitor):
    def __init__(self, findThis, doThat): # type: ignore
        self.findThis = findThis
        self.doThat = doThat
        self.nodeCaptured = None

    def visit(self, node): # type: ignore
        if self.findThis(node):
            nodeActionReturn = self.doThat(node) # type: ignore
            if nodeActionReturn is not None:
                self.nodeCaptured = nodeActionReturn # type: ignore
        self.generic_visit(node)

    def captureLastMatch(self, node): # type: ignore
        """Capture the last matched node that produces a non-None result.

        This method traverses the entire tree starting at the given node
        and returns the last non-None value produced by applying doThat
        to a matching node. It will continue traversing after finding a match,
        and the value captured can be replaced by later matches.

        Parameters:
            node: The AST node to start traversal from

        Returns:
            The result of applying doThat to the last matching node that returned
            a non-None value, or None if no match found or all matches returned None
        """
        self.nodeCaptured = None
        self.visit(node) # type: ignore
        return self.nodeCaptured

class NodeChanger(ast.NodeTransformer):
    def __init__(self, findThis, doThat): # type: ignore
        self.findThis = findThis
        self.doThat = doThat

    def visit(self, node): # type: ignore
        if self.findThis(node):
            return self.doThat(node) # type: ignore
        return super().visit(node)

def importLogicalPath2Callable(logicalPathModule: str_nameDOTname, identifier: ast_Identifier, packageIdentifierIfRelative: ast_Identifier | None = None) -> Callable[..., Any]:
    moduleImported: ModuleType = importlib.import_module(logicalPathModule, packageIdentifierIfRelative)
    return getattr(moduleImported, identifier)

def importPathFilename2Callable(pathFilename: PathLike[Any] | PurePath, identifier: ast_Identifier, moduleIdentifier: ast_Identifier | None = None) -> Callable[..., Any]:
    pathFilename = Path(pathFilename)

    importlibSpecification = importlib.util.spec_from_file_location(moduleIdentifier or pathFilename.stem, pathFilename)
    if importlibSpecification is None or importlibSpecification.loader is None: raise ImportError(f"I received\n\t`{pathFilename = }`,\n\t`{identifier = }`, and\n\t`{moduleIdentifier = }`.\n\tAfter loading, \n\t`importlibSpecification` {'is `None`' if importlibSpecification is None else 'has a value'} and\n\t`importlibSpecification.loader` is unknown.")

    moduleImported_jk_hahaha: ModuleType = importlib.util.module_from_spec(importlibSpecification)
    importlibSpecification.loader.exec_module(moduleImported_jk_hahaha)
    return getattr(moduleImported_jk_hahaha, identifier)

def parseLogicalPath2astModule(logicalPathModule: str_nameDOTname, packageIdentifierIfRelative: ast_Identifier|None=None, mode:str='exec') -> ast.AST:
    moduleImported: ModuleType = importlib.import_module(logicalPathModule, packageIdentifierIfRelative)
    sourcePython: str = inspect_getsource(moduleImported)
    return ast.parse(sourcePython, mode=mode)

def parsePathFilename2astModule(pathFilename: PathLike[Any] | PurePath, mode:str='exec') -> ast.AST:
    return ast.parse(Path(pathFilename).read_text(), mode=mode)
