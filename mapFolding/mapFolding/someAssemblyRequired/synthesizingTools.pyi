import ast
import dataclasses
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
    def RecklessCallAsAttributeIs(callableName: str) -> Callable[[ast.AST], bool]:
        """Warning: You might match more than you want."""
    @staticmethod
    def RecklessCallReallyIs(callableName: str) -> Callable[[ast.AST], bool]:
        """Warning: You might match more than you want."""
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
    def is_dataclassesDOTField() -> None: ...
    @staticmethod
    def isUnpackingAnArray(identifier: str) -> Callable[[ast.AST], bool]: ...

class Make:
    @staticmethod
    def copy_astCallKeywords(astCall: ast.Call) -> dict[str, Any]:
        """Extract keyword parameters from a decorator AST node."""
    @staticmethod
    def astCall(caller: ast.Name | ast.Attribute, args: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call: ...
    @staticmethod
    def astAnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None) -> ast.AnnAssign: ...
    @staticmethod
    def astArgumentsSpecification(posonlyargs: list[ast.arg] = [], args: list[ast.arg] = [], vararg: ast.arg | None = None, kwonlyargs: list[ast.arg] = [], kw_defaults: list[ast.expr | None] = [None], kwarg: ast.arg | None = None, defaults: list[ast.expr] = []) -> ast.arguments: ...
    @staticmethod
    def astFunctionDef(name: ast_Identifier, args: ast.arguments = ..., body: list[ast.stmt] = [], decorator_list: list[ast.expr] = [], returns: ast.expr | None = None, type_comment: str | None = None, type_params: list[ast.type_param] = [], **kwargs: Any) -> ast.FunctionDef: ...
    @staticmethod
    def astName(identifier: ast_Identifier) -> ast.Name: ...
    @staticmethod
    def itDOTname(nameChain: ast.Name | ast.Attribute, dotName: str) -> ast.Attribute: ...
    @staticmethod
    def nameDOTname(identifier: ast_Identifier, *dotName: str) -> ast.Name | ast.Attribute: ...

class Then:
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
    @staticmethod
    def appendAnnAssignOfNameDOTnameTo(instance_Identifier: ast_Identifier, primitiveList: list[Any]) -> Callable[[ast.AST], None]: ...

class NodeReplacer(ast.NodeTransformer):
    """
\tA node transformer that replaces or removes AST nodes based on a condition.
\tThis transformer traverses an AST and for each node checks a predicate. If the predicate
\treturns True, the transformer uses the replacement builder to obtain a new node. Returning
\tNone from the replacement builder indicates that the node should be removed.

\tAttributes:
\t\tfindMe: A function that finds all locations that match a one or more conditions.
\t\tdoThis: A function that does work at each location, such as make a new node, collect information or delete the node.

\tMethods:
\t\tvisit(node: ast.AST) -> Optional[ast.AST]:
\t\t\tVisits each node in the AST, replacing or removing it based on the predicate.
\t"""
    findMe: Incomplete
    doThis: Incomplete
    def __init__(self, findMe: Callable[[ast.AST], bool], doThis: Callable[[ast.AST], ast.AST | Sequence[ast.AST] | None]) -> None: ...
    def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None: ...

def shatter_dataclassesDOTdataclass(dataclass: ast.ClassDef, instance_Identifier: ast_Identifier) -> list[ast.AnnAssign]: ...

class LedgerOfImports:
    dictionaryImportFrom: dict[str, list[tuple[str, str | None]]]
    listImport: list[str]
    def __init__(self, startWith: ast.AST | None = None) -> None: ...
    def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None: ...
    def addImportStr(self, module: str) -> None: ...
    def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None: ...
    def makeListAst(self) -> list[ast.ImportFrom | ast.Import]: ...
    def update(self, *fromTracker: LedgerOfImports) -> None:
        """
\t\tUpdate this tracker with imports from one or more other trackers.

\t\tParameters:
\t\t\t*fromTracker: One or more UniversalImportTracker objects to merge from.
\t\t"""
    def walkThis(self, walkThis: ast.AST) -> None: ...

class FunctionInliner(ast.NodeTransformer):
    dictionaryFunctions: dict[str, ast.FunctionDef]
    def __init__(self, dictionaryFunctions: dict[str, ast.FunctionDef]) -> None: ...
    def inlineFunctionBody(self, callableTargetName: str) -> ast.FunctionDef: ...
    def visit_Call(self, node: ast.Call) -> Any | ast.Constant | ast.Call | ast.AST: ...
    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]: ...

@dataclasses.dataclass
class IngredientsFunction:
    """Everything necessary to integrate a function into a module should be here."""
    FunctionDef: ast.FunctionDef
    imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsModule:
    """Everything necessary to create a module, including the package context, should be here."""
    functions: list[ast.FunctionDef]
    imports: LedgerOfImports
    name: ast_Identifier
    Z0Z_logicalPath: str
    Z0Z_absoluteImport: ast.Import
    Z0Z_absoluteImportFrom: ast.ImportFrom
    Z0Z_pathFilename: Path
    Z0Z_package: str
