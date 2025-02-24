import ast
import pathlib
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Sequence
from mapFolding import EnumIndices as EnumIndices
from typing import Any, NamedTuple

class YouOughtaKnow(NamedTuple):
    callableSynthesized: str
    pathFilenameForMe: pathlib.Path
    astForCompetentProgrammers: ast.ImportFrom

class ifThis:
    @staticmethod
    def nameIs(allegedly: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def subscriptNameIs(allegedly: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def NameReallyIs(allegedly: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isCallWithAttribute(moduleName: str, callableName: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isCallWithName(callableName: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def AssignTo(identifier: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def AnnAssignTo(identifier: str) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def anyOf(*predicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]: ...
    @staticmethod
    def isUnpackingAnArray(identifier: str) -> Callable[[ast.AST], bool]: ...

class Then:
    @staticmethod
    def copy_astCallKeywords(astCall: ast.Call) -> dict[str, Any]:
        """Extract keyword parameters from a decorator AST node."""
    @staticmethod
    def make_astCall(name: str, args: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None, dictionaryKeywords: dict[str, Any] | None = None) -> ast.Call: ...
    @staticmethod
    def removeNode(astNode: ast.AST) -> None: ...

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
    findMe: Callable[[ast.AST], bool]
    doThis: Callable[[ast.AST], ast.AST | Sequence[ast.AST] | None]
    def __init__(self, findMe: Callable[[ast.AST], bool], doThis: Callable[[ast.AST], ast.AST | Sequence[ast.AST] | None]) -> None: ...
    def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None: ...

class UniversalImportTracker:
    dictionaryImportFrom: dict[str, set[tuple[str, str | None]]]
    setImport: set[str]
    def __init__(self) -> None: ...
    def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None: ...
    def addImportStr(self, module: str) -> None: ...
    def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None: ...
    def makeListAst(self) -> list[ast.ImportFrom | ast.Import]: ...
    def update(self, *fromTracker: UniversalImportTracker) -> None:
        """
\t\tUpdate this tracker with imports from one or more other trackers.

\t\tParameters:
\t\t\t*fromTracker: One or more UniversalImportTracker objects to merge from.
\t\t"""

class RecursiveInliner(ast.NodeTransformer):
    """
\tClass RecursiveInliner:
\t\tA custom AST NodeTransformer designed to recursively inline function calls from a given dictionary
\t\tof function definitions into the AST. Once a particular function has been inlined, it is marked
\t\tas completed to avoid repeated inlining. This transformation modifies the AST in-place by substituting
\t\teligible function calls with the body of their corresponding function.
\t\tAttributes:
\t\t\tdictionaryFunctions (Dict[str, ast.FunctionDef]):
\t\t\t\tA mapping of function name to its AST definition, used as a source for inlining.
\t\t\tcallablesCompleted (Set[str]):
\t\t\t\tA set to track function names that have already been inlined to prevent multiple expansions.
\t\tMethods:
\t\t\tinlineFunctionBody(callableTargetName: str) -> Optional[ast.FunctionDef]:
\t\t\t\tRetrieves the AST definition for a given function name from dictionaryFunctions
\t\t\t\tand recursively inlines any function calls within it. Returns the function definition
\t\t\t\tthat was inlined or None if the function was already processed.
\t\t\tvisit_Call(callNode: ast.Call) -> ast.AST:
\t\t\t\tInspects calls within the AST. If a function call matches one in dictionaryFunctions,
\t\t\t\tit is replaced by the inlined body. If the last statement in the inlined body is a return
\t\t\t\tor an expression, that value or expression is substituted; otherwise, a constant is returned.
\t\t\tvisit_Expr(node: ast.Expr) -> Union[ast.AST, List[ast.AST]]:
\t\t\t\tHandles expression nodes in the AST. If the expression is a function call from
\t\t\t\tdictionaryFunctions, its statements are expanded in place, effectively inlining
\t\t\t\tthe called function's statements into the surrounding context.
\t"""
    dictionaryFunctions: dict[str, ast.FunctionDef]
    callablesCompleted: set[str]
    def __init__(self, dictionaryFunctions: dict[str, ast.FunctionDef]) -> None: ...
    def inlineFunctionBody(self, callableTargetName: str) -> ast.FunctionDef | None: ...
    def visit_Call(self, node: ast.Call) -> Any | ast.Constant | ast.Call | ast.AST: ...
    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]: ...

class UnpackArrays(ast.NodeTransformer):
    """
\tA class that transforms array accesses using enum indices into local variables.

\tThis AST transformer identifies array accesses using enum indices and replaces them
\twith local variables, adding initialization statements at the start of functions.

\tParameters:
\t\tenumIndexClass (Type[EnumIndices]): The enum class used for array indexing
\t\tarrayName (str): The name of the array being accessed

\tAttributes:
\t\tenumIndexClass (Type[EnumIndices]): Stored enum class for index lookups
\t\tarrayName (str): Name of the array being transformed
\t\tsubstitutions (dict): Tracks variable substitutions and their original nodes

\tThe transformer handles two main cases:
\t1. Scalar array access - array[EnumIndices.MEMBER]
\t2. Array slice access - array[EnumIndices.MEMBER, other_indices...]
\tFor each identified access pattern, it:
\t1. Creates a local variable named after the enum member
\t2. Adds initialization code at function start
\t3. Replaces original array access with the local variable
\t"""
    enumIndexClass: Incomplete
    arrayName: Incomplete
    substitutions: dict[str, Any]
    def __init__(self, enumIndexClass: type[EnumIndices], arrayName: str) -> None: ...
    def extract_member_name(self, node: ast.AST) -> str | None:
        """Recursively extract enum member name from any node in the AST."""
    def transform_slice_element(self, node: ast.AST) -> ast.AST:
        """Transform any enum references within a slice element."""
    def visit_Subscript(self, node: ast.Subscript) -> ast.AST: ...
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef: ...
