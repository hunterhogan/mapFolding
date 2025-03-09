import ast
import dataclasses
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Sequence
from mapFolding.theSSOT import FREAKOUT as FREAKOUT, Z0Z_autoflake_additional_imports as Z0Z_autoflake_additional_imports, getDatatypePackage as getDatatypePackage, thePackageName as thePackageName, thePathPackage as thePathPackage, theFileExtension as theFileExtension
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

ast_Identifier: TypeAlias
strDotStrCuzPyStoopid: TypeAlias
Z0Z_thisCannotBeTheBestWay: TypeAlias

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
    def isUnpackingAnArray(identifier: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isAnnotation_astName() -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isAnnotationAttribute() -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isAnyAnnotation() -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def findAnnotationNames() -> Callable[[ast.AST], bool]: ...

class Make:
    @staticmethod
    def copy_astCallKeywords(astCall: ast.Call) -> dict[str, Any]:
        """Extract keyword parameters from a decorator AST node."""
    @staticmethod
    def astAlias(name: ast_Identifier, asname: ast_Identifier | None = None) -> ast.alias: ...
    @staticmethod
    def astAnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None, **kwargs: Any) -> ast.AnnAssign:
        """ `simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't add it as a parameter."""
    @staticmethod
    def astAssign(listTargets: Any, value: ast.expr, type_comment: str | None = None, **kwargs: Any) -> ast.Assign: ...
    @staticmethod
    def astArg(identifier: ast_Identifier, annotation: ast.expr | None = None, type_comment: str | None = None, **kwargs: Any) -> ast.arg: ...
    @staticmethod
    def astArgumentsSpecification(posonlyargs: list[ast.arg] = [], args: list[ast.arg] = [], vararg: ast.arg | None = None, kwonlyargs: list[ast.arg] = [], kw_defaults: list[ast.expr | None] = [None], kwarg: ast.arg | None = None, defaults: list[ast.expr] = []) -> ast.arguments: ...
    @staticmethod
    def astCall(caller: ast.Name | ast.Attribute, args: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call: ...
    @staticmethod
    def astFunctionDef(name: ast_Identifier, args: ast.arguments = ..., body: list[ast.stmt] = [], decorator_list: list[ast.expr] = [], returns: ast.expr | None = None, type_comment: str | None = None, type_params: list[ast.type_param] = [], **kwargs: Any) -> ast.FunctionDef: ...
    @staticmethod
    def astImport(moduleName: ast_Identifier, asname: ast_Identifier | None = None) -> ast.Import: ...
    @staticmethod
    def astImportFrom(moduleName: ast_Identifier, list_astAlias: list[ast.alias]) -> ast.ImportFrom: ...
    @staticmethod
    def astKeyword(keywordArgument: ast_Identifier, value: ast.expr, **kwargs: Any) -> ast.keyword: ...
    @staticmethod
    def astModule(body: list[ast.stmt], type_ignores: list[ast.TypeIgnore] = []) -> ast.Module: ...
    @staticmethod
    def astName(identifier: ast_Identifier) -> ast.Name: ...
    @staticmethod
    def itDOTname(nameChain: ast.Name | ast.Attribute, dotName: str) -> ast.Attribute: ...
    @staticmethod
    def nameDOTname(identifier: ast_Identifier, *dotName: str) -> ast.Name | ast.Attribute: ...
    @staticmethod
    def astTuple(elements: Sequence[ast.expr], ctx: ast.expr_context | None = None, **kwargs: Any) -> ast.Tuple:
        """Create an AST Tuple node from a list of expressions.

\t\tParameters:
\t\t\telements: List of AST expressions to include in the tuple.
\t\t\tctx: Context for the tuple (Load/Store). Defaults to Load context.
\t\t"""

class Then:
    @staticmethod
    def insertThisAbove(astStatement: ast.AST) -> Callable[[ast.AST], Sequence[ast.stmt]]: ...
    @staticmethod
    def insertThisBelow(astStatement: ast.AST) -> Callable[[ast.AST], Sequence[ast.stmt]]: ...
    @staticmethod
    def appendTo(primitiveList: list[Any]) -> Callable[[ast.AST], None]: ...
    @staticmethod
    def Z0Z_appendAnnotationNameTo(primitiveList: list[Any]) -> Callable[[ast.AST], None]: ...
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
    dictionaryImportFrom: dict[str, list[ast.alias]]
    listImport: list[str]
    def __init__(self, startWith: ast.AST | None = None) -> None: ...
    def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None: ...
    def addImportStr(self, module: str) -> None: ...
    def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None: ...
    def makeListAst(self) -> list[ast.ImportFrom | ast.Import]: ...
    def update(self, *fromLedger: LedgerOfImports) -> None:
        """
\t\tUpdate this ledger with imports from one or more other ledgers.

\t\tParameters:
\t\t\t*fromTracker: One or more other `LedgerOfImports` objects from which to merge.
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
    fileExtension: str = ...
    packageName: ast_Identifier = ...
    Z0Z_logicalPath: ast_Identifier | strDotStrCuzPyStoopid | None = ...
    Z0Z_pathPackage: Path = ...
    def _getLogicalPathParent(self) -> str: ...
    def _getLogicalPathAbsolute(self) -> str: ...
    @property
    def pathFilename(self) -> Path: ...
    @property
    def absoluteImport(self) -> ast.Import: ...
    @property
    def absoluteImportFrom(self) -> ast.ImportFrom: ...
    def addFunction(self, ingredientsFunction: IngredientsFunction) -> None:
        """Add a function to the module and incorporate its imports.

\t\tParameters:
\t\t\tingredientsFunction: Function with its imports to be added to this module.
\t\t"""
    def addFunctions(self, *ingredientsFunctions: IngredientsFunction) -> None:
        """Add multiple functions to the module and incorporate their imports.

\t\tParameters:
\t\t\t*ingredientsFunctions: One or more functions with their imports to be added.
\t\t"""
    def removeSelfReferencingImports(self) -> None:
        """Remove any imports that reference this module itself."""
    def writeModule(self) -> None:
        """Writes the module to disk with proper imports and functions.

\t\tThis method creates a proper AST module with imports and function definitions,
\t\tfixes missing locations, unpacks the AST to Python code, applies autoflake
\t\tto clean up imports, and writes the resulting code to the appropriate file.
\t\t"""
