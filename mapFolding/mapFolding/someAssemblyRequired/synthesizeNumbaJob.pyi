from mapFolding.someAssemblyRequired.synthesizeNumbaReusable import *
import ast
import numpy
from mapFolding import FREAKOUT as FREAKOUT, Z0Z_identifierCountFolds as Z0Z_identifierCountFolds, Z0Z_setDatatypeModuleScalar as Z0Z_setDatatypeModuleScalar, Z0Z_setDecoratorCallable as Z0Z_setDecoratorCallable, computationState as computationState, getFilenameFoldsTotal as getFilenameFoldsTotal, getPathFilenameFoldsTotal as getPathFilenameFoldsTotal, getPathJobRootDEFAULT as getPathJobRootDEFAULT, indexMy as indexMy, setDatatypeElephino as setDatatypeElephino, setDatatypeFoldsTotal as setDatatypeFoldsTotal, setDatatypeLeavesTotal as setDatatypeLeavesTotal
from mapFolding.someAssemblyRequired import makeStateJob as makeStateJob
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any

def Z0Z_gamma(FunctionDefTarget: ast.FunctionDef, astAssignee: ast.Name, statement: ast.Assign | ast.stmt, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]: ...
def insertArrayIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: UniversalImportTracker, unrollSlices: int | None = None) -> tuple[ast.FunctionDef, UniversalImportTracker]: ...
def findAndReplaceTrackArrayIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]: ...
def findAndReplaceArraySubscriptIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]: ...
def removeAssignTargetFrom_body(FunctionDefTarget: ast.FunctionDef, identifier: str) -> ast.FunctionDef: ...
def findAndReplaceAnnAssignIn_body(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]: ...
def findThingyReplaceWithConstantIn_body(FunctionDefTarget: ast.FunctionDef, object: str, value: int) -> ast.FunctionDef:
    """
\tReplaces nodes in astFunction matching the AST of the string `object`
\twith a constant node holding the provided value.
\t"""
def findAstNameReplaceWithConstantIn_body(FunctionDefTarget: ast.FunctionDef, name: str, value: int) -> ast.FunctionDef: ...
def insertReturnStatementIn_body(FunctionDefTarget: ast.FunctionDef, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]:
    """Add multiplication and return statement to function, properly constructing AST nodes."""
def findAndReplaceWhileLoopIn_body(FunctionDefTarget: ast.FunctionDef, iteratorName: str, iterationsTotal: int) -> ast.FunctionDef:
    """
\tUnroll all nested while loops matching the condition that their test uses `iteratorName`.
\t"""
def makeLauncherTqdmJobNumba(callableTarget: str, pathFilenameFoldsTotal: Path, totalEstimated: int) -> ast.Module: ...
def makeLauncherBasicJobNumba(callableTarget: str, pathFilenameFoldsTotal: Path) -> ast.Module: ...
def doUnrollCountGaps(FunctionDefTarget: ast.FunctionDef, stateJob: computationState, allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]:
    """The initial results were very bad."""
def writeJobNumba(mapShape: Sequence[int], algorithmSource: ModuleType, callableTarget: str | None = None, parametersNumba: ParametersNumba | None = None, pathFilenameWriteJob: str | PathLike[str] | None = None, unrollCountGaps: bool | None = False, **keywordArguments: Any | None) -> Path:
    """ Parameters: **keywordArguments: most especially for `computationDivisions` if you want to make a parallel job. Also `CPUlimit`. """
