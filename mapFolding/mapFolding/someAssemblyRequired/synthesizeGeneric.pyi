import ast
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

ast_Identifier: TypeAlias

class YouOughtaKnow(NamedTuple):
    callableSynthesized: str
    pathFilenameForMe: Path
    astForCompetentProgrammers: ast.ImportFrom

class ifThis:
    @staticmethod
    def nameIs(allegedly: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def subscriptNameIs(allegedly: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def NameReallyIs(allegedly: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def CallAsNameIs(callableName: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def CallAsNameIsIn(container: Iterable[Any]) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def CallAsModuleAttributeIs(moduleName: str, callableName: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def CallReallyIs(moduleName: str, callableName: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def CallDoesNotCallItself(moduleName: str, callableName: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def RecklessCallAsAttributeIs(callableName: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def RecklessCallReallyIs(callableName: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def AssignTo(identifier: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isAnnAssign() -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isAnnAssignTo(identifier: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def AugAssignTo(identifier: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def anyAssignmentTo(identifier: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def anyOf(*predicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def is_dataclassesDOTField() -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isUnpackingAnArray(identifier: str) -> Callable[[ast.AST], bool]: ...

class Then:
    @staticmethod
    def copy_astCallKeywords(astCall: ast.Call) -> dict[str, Any]: ...
    @staticmethod
    def make_astCall(caller: ast.Name | ast.Attribute, args: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None, dictionaryKeywords: dict[str, Any] | None = None) -> ast.Call: ...
    @staticmethod
    def makeName(identifier: ast_Identifier) -> ast.Name: ...
    @staticmethod
    def addDOTname(nameChain: ast.Name | ast.Attribute, dotName: str) -> ast.Attribute: ...
    @staticmethod
    def makeNameDOTname(identifier: ast_Identifier, *dotName: str) -> ast.Name | ast.Attribute: ...
    @staticmethod
    def insertThisAbove(astStatement: ast.AST) -> Callable[[ast.AST], Sequence[ast.stmt]]: ...
    @staticmethod
    def insertThisBelow(astStatement: ast.AST) -> Callable[[ast.AST], Sequence[ast.stmt]]: ...
    @staticmethod
    def appendTo(primitiveList: list[Any]) -> Callable[[ast.AST], None]: ...
    @staticmethod
    def replaceWith(astStatement: ast.AST) -> Callable[[ast.AST], ast.stmt]: ...
    @staticmethod
    def removeThis(astNode: ast.AST) -> None: ...

class NodeReplacer(ast.NodeTransformer):
    findMe: Incomplete
    doThis: Incomplete
    def __init__(self, findMe: Callable[[ast.AST], bool], doThis: Callable[[ast.AST], ast.AST | Sequence[ast.AST] | None]) -> None: ...
    def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None: ...

def shatter_dataclassesDOTdataclass(dataclass: ast.AST) -> list[ast.AnnAssign]: ...

class UniversalImportTracker:
    dictionaryImportFrom: dict[str, list[tuple[str, str | None]]]
    listImport: list[str]
    def __init__(self, startWith: ast.AST | None = None) -> None: ...
    def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None: ...
    def addImportStr(self, module: str) -> None: ...
    def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None: ...
    def makeListAst(self) -> list[ast.ImportFrom | ast.Import]: ...
    def update(self, *fromTracker: UniversalImportTracker) -> None: ...
    def walkThis(self, walkThis: ast.AST) -> None: ...

class FunctionInliner(ast.NodeTransformer):
    dictionaryFunctions: dict[str, ast.FunctionDef]
    def __init__(self, dictionaryFunctions: dict[str, ast.FunctionDef]) -> None: ...
    def inlineFunctionBody(self, callableTargetName: str) -> ast.FunctionDef: ...
    def visit_Call(self, node: ast.Call) -> Any | ast.Constant | ast.Call | ast.AST: ...
    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]: ...
