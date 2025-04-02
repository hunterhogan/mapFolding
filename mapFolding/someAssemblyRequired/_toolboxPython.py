from collections.abc import Callable, Sequence
from inspect import getsource as inspect_getsource
from mapFolding.someAssemblyRequired import ast_Identifier, str_nameDOTname, 个
from os import PathLike
from pathlib import Path, PurePath
from types import ModuleType
from typing import Any, cast, Generic, TypeGuard
import ast
import importlib
import importlib.util

# TODO Identify the logic that narrows the type and can help the user during static type checking.

class NodeTourist(ast.NodeVisitor, Generic[个]):
    def __init__(self, findThis: Callable[[个], TypeGuard[个] | bool], doThat: Callable[[个], 个 | None]) -> None:
        self.findThis = findThis
        self.doThat = doThat
        self.nodeCaptured: 个 | None = None

    def visit(self, node: 个) -> None: # pyright: ignore [reportGeneralTypeIssues]
        if self.findThis(node):
            nodeActionReturn = self.doThat(node)
            if nodeActionReturn is not None:
                self.nodeCaptured = nodeActionReturn
        self.generic_visit(cast(ast.AST, node))

    def captureLastMatch(self, node: 个) -> 个 | None: # pyright: ignore [reportGeneralTypeIssues]
        self.nodeCaptured = None
        self.visit(node)
        return self.nodeCaptured

class NodeChanger(ast.NodeTransformer, Generic[个]):
    def __init__(self, findThis: Callable[[个], bool], doThat: Callable[[个], Sequence[个] | 个 | None]) -> None:
        self.findThis = findThis
        self.doThat = doThat

    def visit(self, node: 个) -> Sequence[个] | 个 | None: # pyright: ignore [reportGeneralTypeIssues]
        if self.findThis(node):
            return self.doThat(node)
        return super().visit(cast(ast.AST, node))

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
