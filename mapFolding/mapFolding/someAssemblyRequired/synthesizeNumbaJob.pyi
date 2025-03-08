import ast
import numpy
from collections.abc import Sequence
from mapFolding.filesystem import getFilenameFoldsTotal as getFilenameFoldsTotal, getPathFilenameFoldsTotal as getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import LedgerOfImports as LedgerOfImports, Make as Make, NodeReplacer as NodeReplacer, Then as Then, decorateCallableWithNumba as decorateCallableWithNumba, ifThis as ifThis, makeStateJob as makeStateJob, thisIsNumbaDotJit as thisIsNumbaDotJit
from mapFolding.theSSOT import ComputationState as ComputationState, FREAKOUT as FREAKOUT, ParametersNumba as ParametersNumba, Z0Z_getDatatypeModuleScalar as Z0Z_getDatatypeModuleScalar, Z0Z_getDecoratorCallable as Z0Z_getDecoratorCallable, Z0Z_setDatatypeModuleScalar as Z0Z_setDatatypeModuleScalar, Z0Z_setDecoratorCallable as Z0Z_setDecoratorCallable, getPathJobRootDEFAULT as getPathJobRootDEFAULT, parametersNumbaDEFAULT as parametersNumbaDEFAULT
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any

def Z0Z_gamma(FunctionDefTarget: ast.FunctionDef, astAssignee: ast.Name, statement: ast.Assign | ast.stmt, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]: ...
def insertArrayIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports, unrollSlices: int | None = None) -> tuple[ast.FunctionDef, LedgerOfImports]: ...
def findAndReplaceTrackArrayIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]: ...
def findAndReplaceArraySubscriptIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]: ...
def removeAssignmentFrom_body(FunctionDefTarget: ast.FunctionDef, identifier: str) -> ast.FunctionDef: ...
def findAndReplaceAnnAssignIn_body(FunctionDefTarget: ast.FunctionDef, allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
    """Unlike most of the other functions, this is generic: it tries to turn an annotation into a construction call."""
def findThingyReplaceWithConstantIn_body(FunctionDefTarget: ast.FunctionDef, object: str, value: int) -> ast.FunctionDef:
    """
\tReplaces nodes in astFunction matching the AST of the string `object`
\twith a constant node holding the provided value.
\t"""
def findAstNameReplaceWithConstantIn_body(FunctionDefTarget: ast.FunctionDef, name: str, value: int) -> ast.FunctionDef: ...
def insertReturnStatementIn_body(FunctionDefTarget: ast.FunctionDef, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
    """Add multiplication and return statement to function, properly constructing AST nodes."""
def findAndReplaceWhileLoopIn_body(FunctionDefTarget: ast.FunctionDef, iteratorName: str, iterationsTotal: int) -> ast.FunctionDef:
    """
\tUnroll all nested while loops matching the condition that their test uses `iteratorName`.
\t"""
def makeLauncherTqdmJobNumba(callableTarget: str, pathFilenameFoldsTotal: Path, totalEstimated: int, leavesTotal: int) -> ast.Module: ...
def makeLauncherBasicJobNumba(callableTarget: str, pathFilenameFoldsTotal: Path) -> ast.Module: ...
def doUnrollCountGaps(FunctionDefTarget: ast.FunctionDef, stateJob: ComputationState, allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
    """The initial results were very bad."""
def writeJobNumba(mapShape: Sequence[int], algorithmSource: ModuleType, callableTarget: str | None = None, parametersNumba: ParametersNumba | None = None, pathFilenameWriteJob: str | PathLike[str] | None = None, unrollCountGaps: bool | None = False, Z0Z_totalEstimated: int = 0, **keywordArguments: Any | None) -> Path:
    """ Parameters: **keywordArguments: most especially for `computationDivisions` if you want to make a parallel job. Also `CPUlimit`.
\tNotes:
\tHypothetically, everything can now be configured with parameters and functions. And changing how the job is written is relatively easy.

\tOverview
\t- the code starts life in theDao.py, which has many optimizations; `makeNumbaOptimizedFlow` increase optimization especially by using numba; `writeJobNumba` increases optimization especially by limiting its capabilities to just one set of parameters
\t- the synthesized module must run well as a standalone interpreted-Python script
\t- the next major optimization step will (probably) be to use the module synthesized by `writeJobNumba` to compile a standalone executable
\t- Nevertheless, at each major optimization step, the code is constantly being improved and optimized, so everything must be well organized and able to handle upstream and downstream changes

\tMinutia
\t- perf_counter is for testing. When I run a real job, I delete those lines
\t- avoid `with` statement

\tNecessary
\t- Move the function's parameters to the function body,
\t- initialize identifiers with their state types and values,

\tOptimizations
\t- replace static-valued identifiers with their values
\t- narrowly focused imports
\t"""
