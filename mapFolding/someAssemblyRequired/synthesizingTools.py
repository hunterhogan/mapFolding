from collections import defaultdict
from collections.abc import Callable, Container, Iterable, Sequence
from importlib import import_module
from inspect import getsource as inspect_getsource
from mapFolding.theSSOT import FREAKOUT, getDatatypePackage, theFileExtension, thePackageName, thePathPackage
from pathlib import Path
from typing import Any, cast, NamedTuple, TypeAlias, TypeGuard, TypeVar
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import autoflake
import dataclasses
"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a TypeAlias for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""
# TODO consider semiotic usefulness of "namespace" or variations such as "namespaceName", "namespacePath", and "namespace_Identifier"

# TODO learn whether libcst can help

astParameter = TypeVar('astParameter', bound=Any)
ast_Identifier: TypeAlias = str
strDotStrCuzPyStoopid: TypeAlias = str
strORlist_ast_type_paramORintORNone: TypeAlias = Any
strORintORNone: TypeAlias = Any
Z0Z_thisCannotBeTheBestWay: TypeAlias = list[ast.Name] | list[ast.Attribute] | list[ast.Subscript] | list[ast.Name | ast.Attribute] | list[ast.Name | ast.Subscript] | list[ast.Attribute | ast.Subscript] | list[ast.Name | ast.Attribute | ast.Subscript]

# NOTE: the new "Recipe" concept will allow me to remove this
class YouOughtaKnow(NamedTuple):
	callableSynthesized: str
	pathFilenameForMe: Path
	astForCompetentProgrammers: ast.ImportFrom

class NodeCollector(ast.NodeVisitor):
	# A node visitor that collects data via one or more actions when a predicate is met.
	def __init__(self, findPredicate: Callable[[ast.AST], bool], actions: list[Callable[[ast.AST], None]]) -> None:
		self.findPredicate = findPredicate
		self.actions = actions

	def visit(self, node: ast.AST) -> None:
		if self.findPredicate(node):
			for action in self.actions:
				action(node)
		self.generic_visit(node)

class NodeReplacer(ast.NodeTransformer):
	"""
	A node transformer that replaces or removes AST nodes based on a condition.
	This transformer traverses an AST and for each node checks a predicate. If the predicate
	returns True, the transformer uses the replacement builder to obtain a new node. Returning
	None from the replacement builder indicates that the node should be removed.

	Attributes:
		findMe: A function that finds all locations that match a one or more conditions.
		doThis: A function that does work at each location, such as make a new node, collect information or delete the node.

	Methods:
		visit(node: ast.AST) -> Optional[ast.AST]:
			Visits each node in the AST, replacing or removing it based on the predicate.
	"""
	def __init__(self
			, findMe: Callable[[ast.AST], bool]
			, doThis: Callable[[ast.AST], ast.AST | Sequence[ast.AST] | None]
			) -> None:
		self.findMe = findMe
		self.doThis = doThis

	def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None:
		if self.findMe(node):
			return self.doThis(node)
		return super().visit(node)

def descendantContainsMatchingNode(node: ast.AST, predicateFunction: Callable[[ast.AST], bool]) -> bool:
	""" Return True if any descendant of the node (or the node itself) matches the predicateFunction. """
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

def executeActionUnlessDescendantMatches(exclusionPredicate: Callable[[ast.AST], bool], actionFunction: Callable[[ast.AST], None]) -> Callable[[ast.AST], None]:
	"""
	Return a new action that will execute actionFunction only if no descendant (or the node itself)
	matches exclusionPredicate.
	"""
	def wrappedAction(node: ast.AST) -> None:
		if not descendantContainsMatchingNode(node, exclusionPredicate):
			actionFunction(node)
	return wrappedAction

class ifThis:
	@staticmethod
	def anyOf(*somePredicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: any(predicate(node) for predicate in somePredicates)

	@staticmethod
	def ast_IdentifierIsIn(container: Container[ast_Identifier]) -> Callable[[ast_Identifier], bool]:
		return lambda node: node in container

	# TODO is this only useable if namespace is not `None`? Yes, but use "" for namespace if necessary.
	@staticmethod
	def CallDoesNotCallItself(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.CallReallyIs(namespace, identifier)(node) and 1 == sum(1 for descendant in ast.walk(node) if ifThis.CallReallyIs(namespace, identifier)(descendant))

	@staticmethod
	def CallDoesNotCallItselfAndNameDOTidIsIn(container: Container[ast_Identifier]) -> Callable[[ast.AST], bool]:
		return lambda node: (ifThis.isCall(node) and ifThis.isName(node.func) and ifThis.ast_IdentifierIsIn(container)(node.func.id) and ifThis.CallDoesNotCallItself("", node.func.id)(node))

	@staticmethod
	def CallReallyIs(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.isCall_Identifier(identifier), ifThis.isCallNamespace_Identifier(namespace, identifier))

	@staticmethod
	def is_keyword(node: ast.AST) -> TypeGuard[ast.keyword]:
		return isinstance(node, ast.keyword)

	@staticmethod
	def is_keywordAndValueIsConstant(node: ast.AST) -> TypeGuard[ast.keyword]:
		return ifThis.is_keyword(node) and ifThis.isConstant(node.value)

	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.is_keyword(node) and node.arg == identifier
	@staticmethod
	def is_keyword_IdentifierEqualsConstantValue(identifier: ast_Identifier, ConstantValue: Any) -> Callable[[ast.AST], bool]:
		return lambda node: (ifThis.is_keyword_Identifier(identifier)(node) and ifThis.is_keywordAndValueIsConstant(node) and ifThis.isConstantEquals(ConstantValue)(node.value))

	@staticmethod
	def isAnnAssign(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return isinstance(node, ast.AnnAssign)

	@staticmethod
	def isAnnAssignAndAnnotationIsName(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return ifThis.isAnnAssign(node) and ifThis.isName(node.annotation)

	@staticmethod
	def isAnnAssignAndTargetIsName(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return ifThis.isAnnAssign(node) and ifThis.isName(node.target)

	@staticmethod
	def isAnnAssignTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isAnnAssign(node) and ifThis.NameReallyIs(identifier)(node.target)

	@staticmethod
	def isAnyAssignmentTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.isAssignOnlyTo(identifier), ifThis.isAnnAssignTo(identifier), ifThis.isAugAssignTo(identifier))

	@staticmethod
	def isAssign(node: ast.AST) -> TypeGuard[ast.Assign]:
		return isinstance(node, ast.Assign)

	@staticmethod
	def isAssignOnlyTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isAssign(node) and ifThis.NameReallyIs(identifier)(node.targets[0])

	@staticmethod
	def isAttribute(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return isinstance(node, ast.Attribute)

	@staticmethod
	def isAugAssign(node: ast.AST) -> TypeGuard[ast.AugAssign]:
		return isinstance(node, ast.AugAssign)

	@staticmethod
	def isAugAssignTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isAugAssign(node) and ifThis.NameReallyIs(identifier)(node.target)

	@staticmethod
	def isCall(node: ast.AST) -> TypeGuard[ast.Call]:
		return isinstance(node, ast.Call)

	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isCall(node) and ifThis.isName_Identifier(identifier)(node.func)

	# TODO what happens if `None` is passed as the namespace?
	@staticmethod
	def isCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isCall(node) and ifThis.isNameDOTnameNamespace_Identifier(namespace, identifier)(node.func)

	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[ast.Call]:
		return ifThis.isCall(node) and ifThis.isName(node.func)

	@staticmethod
	def isClassDef(node: ast.AST) -> TypeGuard[ast.ClassDef]:
		return isinstance(node, ast.ClassDef)

	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isClassDef(node) and node.name == identifier

	@staticmethod
	def isConstant(node: ast.AST) -> TypeGuard[ast.Constant]:
		return isinstance(node, ast.Constant)

	@staticmethod
	def isConstantEquals(value: Any) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isConstant(node) and node.value == value

	@staticmethod
	def isFunctionDef(node: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return isinstance(node, ast.FunctionDef)

	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isFunctionDef(node) and node.name == identifier

	@staticmethod
	def isImport(node: ast.AST) -> TypeGuard[ast.Import]:
		return isinstance(node, ast.Import)

	@staticmethod
	def isName(node: ast.AST) -> TypeGuard[ast.Name]:
		return isinstance(node, ast.Name)

	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isName(node) and node.id == identifier

	@staticmethod
	def isNameDOTname(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return ifThis.isAttribute(node) and ifThis.isName(node.value)

	@staticmethod
	def isNameDOTnameNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isNameDOTname(node) and ifThis.isName_Identifier(namespace)(node.value) and node.attr == identifier

	@staticmethod
	def isSubscript(node: ast.AST) -> TypeGuard[ast.Subscript]:
		return isinstance(node, ast.Subscript)

	@staticmethod
	def isSubscript_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isSubscript(node) and ifThis.isName_Identifier(identifier)(node.value)

	@staticmethod
	def isSubscript_Identifier_Identifier(identifier: ast_Identifier, sliceIdentifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda node: ifThis.isSubscript(node) and ifThis.isName_Identifier(identifier)(node.value) and ifThis.isName_Identifier(sliceIdentifier)(node.slice) # auto-generated

	@staticmethod
	def NameReallyIs(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.isName_Identifier(identifier), ifThis.isSubscript_Identifier(identifier))

class Make:
	@staticmethod
	def copy_astCallKeywords(astCall: ast.Call) -> dict[str, Any]:
		"""Extract keyword parameters from a decorator AST node."""
		dictionaryKeywords: dict[str, Any] = {}
		for keywordItem in astCall.keywords:
			if isinstance(keywordItem.value, ast.Constant) and keywordItem.arg is not None:
				dictionaryKeywords[keywordItem.arg] = keywordItem.value.value
		return dictionaryKeywords

	@staticmethod
	def astAlias(name: ast_Identifier, asname: ast_Identifier | None = None) -> ast.alias:
		return ast.alias(name=name, asname=asname)

	@staticmethod
	def astAnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None, **keywordArguments: int) -> ast.AnnAssign:
		""" `simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't add it as a parameter."""
		return ast.AnnAssign(target, annotation, value, simple=int(isinstance(target, ast.Name)), **keywordArguments)

	@staticmethod
	def astAssign(listTargets: Any, value: ast.expr, **keywordArguments: strORintORNone) -> ast.Assign:
		"""keywordArguments: type_comment:str|None, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.Assign(targets=listTargets, value=value, **keywordArguments)

	@staticmethod
	def astArg(identifier: ast_Identifier, annotation: ast.expr | None = None, **keywordArguments: strORintORNone) -> ast.arg:
		"""keywordArguments: type_comment:str|None, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.arg(identifier, annotation, **keywordArguments)

	@staticmethod
	def astArgumentsSpecification(posonlyargs: list[ast.arg]=[], args: list[ast.arg]=[], vararg: ast.arg|None=None, kwonlyargs: list[ast.arg]=[], kw_defaults: list[ast.expr|None]=[None], kwarg: ast.arg|None=None, defaults: list[ast.expr]=[]) -> ast.arguments:
		return ast.arguments(posonlyargs=posonlyargs, args=args, vararg=vararg, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults, kwarg=kwarg, defaults=defaults)

	@staticmethod
	def astCall(caller: ast.Name | ast.Attribute, args: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call:
		return ast.Call(func=caller, args=list(args) if args else [], keywords=list(list_astKeywords) if list_astKeywords else [])

	@staticmethod
	def astFunctionDef(name: ast_Identifier, args: ast.arguments=ast.arguments(), body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], returns: ast.expr|None=None, **keywordArguments: strORlist_ast_type_paramORintORNone) -> ast.FunctionDef:
		"""keywordArguments: type_comment:str|None, type_params:list[ast.type_param], lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.FunctionDef(name=name, args=args, body=body, decorator_list=decorator_list, returns=returns, **keywordArguments)

	@staticmethod
	def astImport(moduleName: ast_Identifier, asname: ast_Identifier | None = None, **keywordArguments: int) -> ast.Import:
		return ast.Import(names=[Make.astAlias(moduleName, asname)], **keywordArguments)

	@staticmethod
	def astImportFrom(moduleName: ast_Identifier, list_astAlias: list[ast.alias], **keywordArguments: int) -> ast.ImportFrom:
		return ast.ImportFrom(module=moduleName, names=list_astAlias, level=0, **keywordArguments)

	@staticmethod
	def astKeyword(keywordArgument: ast_Identifier, value: ast.expr, **keywordArguments: int) -> ast.keyword:
		return ast.keyword(arg=keywordArgument, value=value, **keywordArguments)

	@staticmethod
	def astModule(body: list[ast.stmt], type_ignores: list[ast.TypeIgnore] = []) -> ast.Module:
		return ast.Module(body=body, type_ignores=type_ignores)

	@staticmethod
	def astName(identifier: ast_Identifier) -> ast.Name:
		return ast.Name(id=identifier, ctx=ast.Load())

	@staticmethod
	def itDOTname(nameChain: ast.Name | ast.Attribute, dotName: str) -> ast.Attribute:
		return ast.Attribute(value=nameChain, attr=dotName, ctx=ast.Load())

	@staticmethod
	def nameDOTname(identifier: ast_Identifier, *dotName: str) -> ast.Name | ast.Attribute:
		nameDOTname: ast.Name | ast.Attribute = Make.astName(identifier)
		if not dotName:
			return nameDOTname
		for suffix in dotName:
			nameDOTname = Make.itDOTname(nameDOTname, suffix)
		return nameDOTname

	@staticmethod
	def astReturn(value: ast.expr | None = None, **keywordArguments: int) -> ast.Return:
		return ast.Return(value=value, **keywordArguments)

	@staticmethod
	def astTuple(elements: Sequence[ast.expr], context: ast.expr_context | None = None, **keywordArguments: int) -> ast.Tuple:
		"""context: Load/Store/Del"""
		context = context or ast.Load()
		return ast.Tuple(elts=list(elements), ctx=context, **keywordArguments)

class LedgerOfImports:
	def __init__(self, startWith: ast.AST | None = None) -> None:
		self.dictionaryImportFrom: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
		self.listImport: list[str] = []

		if startWith:
			self.walkThis(startWith)

	def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None:
		if not isinstance(astImport_, (ast.Import, ast.ImportFrom)):
			raise ValueError(f"Expected ast.Import or ast.ImportFrom, got {type(astImport_)}")
		if isinstance(astImport_, ast.Import):
			for alias in astImport_.names:
				self.listImport.append(alias.name)
		else:
			if astImport_.module is not None:
				for alias in astImport_.names:
					self.dictionaryImportFrom[astImport_.module].append((alias.name, alias.asname))

	def addImportStr(self, module: str) -> None:
		self.listImport.append(module)

	def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None:
		self.dictionaryImportFrom[module].append((name, asname))

	def makeListAst(self) -> list[ast.ImportFrom | ast.Import]:
		listAstImportFrom: list[ast.ImportFrom] = []

		for module, listOfNameTuples in sorted(self.dictionaryImportFrom.items()):
			listOfNameTuples = sorted(list(set(listOfNameTuples)), key=lambda nameTuple: nameTuple[0])
			listAlias: list[ast.alias] = []
			for name, asname in listOfNameTuples:
				listAlias.append(Make.astAlias(name, asname))
			listAstImportFrom.append(Make.astImportFrom(module, listAlias))

		listAstImport: list[ast.Import] = [Make.astImport(name) for name in sorted(set(self.listImport))]
		return listAstImportFrom + listAstImport

	def update(self, *fromLedger: 'LedgerOfImports') -> None:
		"""
		Update this ledger with imports from one or more other ledgers.

		Parameters:
			*fromTracker: One or more other `LedgerOfImports` objects from which to merge.
		"""
		self.dictionaryImportFrom = updateExtendPolishDictionaryLists(self.dictionaryImportFrom, *(ledger.dictionaryImportFrom for ledger in fromLedger), destroyDuplicates=True, reorderLists=True)

		for ledger in fromLedger:
			self.listImport.extend(ledger.listImport)

	def walkThis(self, walkThis: ast.AST) -> None:
		for smurf in ast.walk(walkThis):
			if isinstance(smurf, (ast.Import, ast.ImportFrom)):
				self.addAst(smurf)

class Then:
	@staticmethod
	def insertThisAbove(astStatement: ast.stmt) -> Callable[[ast.stmt], Sequence[ast.stmt]]:
		return lambda aboveMe: [astStatement, aboveMe]
	@staticmethod
	def insertThisBelow(astStatement: ast.stmt) -> Callable[[ast.stmt], Sequence[ast.stmt]]:
		return lambda belowMe: [belowMe, astStatement]
	@staticmethod
	def replaceWith(astStatement: ast.stmt) -> Callable[[ast.stmt], ast.stmt]:
		return lambda replaceMe: astStatement
	@staticmethod
	def removeThis(astNode: ast.AST) -> None: return None

	@staticmethod
	def Z0Z_ledger(logicalPath: strDotStrCuzPyStoopid, ledger: LedgerOfImports) -> Callable[[ast.AST], None]:
		return lambda node: ledger.addImportFromStr(logicalPath, cast(ast.Name, cast(ast.AnnAssign, node).annotation).id)

	@staticmethod
	def Z0Z_appendKeywordMirroredTo(list_keyword: list[ast.keyword]) -> Callable[[ast.AST], None]:
		return lambda node: list_keyword.append(Make.astKeyword(cast(ast.Name, cast(ast.AnnAssign, node).target).id, cast(ast.Name, cast(ast.AnnAssign, node).target)))

	@staticmethod
	def append_targetTo(listName: list[ast.Name]) -> Callable[[ast.AST], None]:
		return lambda node: listName.append(cast(ast.Name, cast(ast.AnnAssign, node).target))

	@staticmethod
	def appendTo(list_stmt: Sequence[ast.stmt]) -> Callable[[ast.stmt], None]:
		return lambda node: list(list_stmt).append(node)

	@staticmethod
	def Z0Z_appendAnnAssignOfNameDOTnameTo(identifier: ast_Identifier, listNameDOTname: list[ast.AnnAssign]) -> Callable[[ast.AST], None]:
		return lambda node: Then.appendTo(listNameDOTname)(Make.astAnnAssign(cast(ast.AnnAssign, node).target, cast(ast.AnnAssign, node).annotation, Make.nameDOTname(identifier, cast(ast.Name, cast(ast.AnnAssign, node).target).id)))

class FunctionInliner(ast.NodeTransformer):
	def __init__(self, dictionaryFunctions: dict[str, ast.FunctionDef]) -> None:
		self.dictionaryFunctions: dict[str, ast.FunctionDef] = dictionaryFunctions

	def inlineFunctionBody(self, callableTargetName: str) -> ast.FunctionDef:
		inlineDefinition: ast.FunctionDef = self.dictionaryFunctions[callableTargetName]
		# Process nested calls within the inlined function
		for astNode in ast.walk(inlineDefinition):
			self.visit(astNode)
		return inlineDefinition

	def visit_Call(self, node: ast.Call):
		astCall = self.generic_visit(node)
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryFunctions)(astCall):
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(cast(ast.Name, cast(ast.Call, astCall).func).id)

			if (inlineDefinition and inlineDefinition.body):
				statementTerminating: ast.stmt = inlineDefinition.body[-1]

				if (isinstance(statementTerminating, ast.Return)
				and statementTerminating.value is not None):
					return self.visit(statementTerminating.value)
				elif isinstance(statementTerminating, ast.Expr):
					return self.visit(statementTerminating.value)
				else:
					return ast.Constant(value=None)
		return astCall

	def visit_Expr(self, node: ast.Expr):
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryFunctions)(node.value):
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(cast(ast.Name, cast(ast.Call, node.value).func).id)
			return [self.visit(stmt) for stmt in inlineDefinition.body]
		return self.generic_visit(node)

@dataclasses.dataclass
class IngredientsFunction:
	"""Everything necessary to integrate a function into a module should be here."""
	FunctionDef: ast.FunctionDef
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsModule:
	"""Everything necessary to create a module, including the package context, should be here."""
	name: ast_Identifier

	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	prologue: list[ast.stmt] = dataclasses.field(default_factory=list)
	functions: list[ast.FunctionDef | ast.stmt] = dataclasses.field(default_factory=list)
	epilogue: list[ast.stmt] = dataclasses.field(default_factory=list)
	launcher: list[ast.stmt] = dataclasses.field(default_factory=list)

	packageName: ast_Identifier | None= thePackageName
	logicalPathINFIX: ast_Identifier | strDotStrCuzPyStoopid | None = None # module names other than the module itself and the package name
	pathPackage: Path = thePathPackage
	fileExtension: str = theFileExtension
	type_ignores: list[ast.TypeIgnore] = dataclasses.field(default_factory=list)

	def _getLogicalPathParent(self) -> str | None:
		listModules: list[ast_Identifier] = []
		if self.packageName:
			listModules.append(self.packageName)
		if self.logicalPathINFIX:
			listModules.append(self.logicalPathINFIX)
		if listModules:
			return '.'.join(listModules)

	def _getLogicalPathAbsolute(self) -> str:
		listModules: list[ast_Identifier] = []
		logicalPathParent: str | None = self._getLogicalPathParent()
		if logicalPathParent:
			listModules.append(logicalPathParent)
		listModules.append(self.name)
		return '.'.join(listModules)

	@property
	def pathFilename(self) -> Path:
		pathRoot: Path = self.pathPackage
		filename = self.name + self.fileExtension
		if self.logicalPathINFIX:
			whyIsThisStillAThing = self.logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		return pathRoot.joinpath(filename)

	@property
	def absoluteImport(self) -> ast.Import:
		return Make.astImport(self._getLogicalPathAbsolute())

	@property
	def absoluteImportFrom(self) -> ast.ImportFrom:
		""" `from . import theModule` """
		logicalPathParent: str | None = self._getLogicalPathParent()
		if logicalPathParent is None:
			logicalPathParent = '.'
		return Make.astImportFrom(logicalPathParent, [Make.astAlias(self.name)])

	def addFunctions(self, *ingredientsFunction: IngredientsFunction) -> None:
		"""Add one or more `IngredientsFunction`. """
		listLedgers: list[LedgerOfImports] = []
		for definition in ingredientsFunction:
			self.functions.append(definition.FunctionDef)
			listLedgers.append(definition.imports)
		self.imports.update(*listLedgers)

	def _makeModuleBody(self) -> list[ast.stmt]:
		"""Constructs the body of the module, including prologue, functions, epilogue, and launcher."""
		body: list[ast.stmt] = []
		body.extend(self.imports.makeListAst())
		body.extend(self.prologue)
		body.extend(self.functions)
		body.extend(self.epilogue)
		body.extend(self.launcher)
		# TODO `launcher` must start with `if __name__ == '__main__':` and be indented
		return body

	def writeModule(self) -> None:
		"""Writes the module to disk with proper imports and functions.

		This method creates a proper AST module with imports and function definitions,
		fixes missing locations, unpacks the AST to Python code, applies autoflake
		to clean up imports, and writes the resulting code to the appropriate file.
		"""
		astModule = Make.astModule(body=self._makeModuleBody(), type_ignores=self.type_ignores)
		ast.fix_missing_locations(astModule)
		pythonSource: str = ast.unparse(astModule)
		if not pythonSource: raise FREAKOUT
		autoflake_additional_imports: list[str] = []
		if self.packageName:
			autoflake_additional_imports.append(self.packageName)
		# TODO LedgerOfImports method: list of package names. autoflake_additional_imports.extend()
		autoflake_additional_imports.append(getDatatypePackage())
		pythonSource = autoflake.fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=False, remove_duplicate_keys = False, remove_unused_variables = False,)
		self.pathFilename.write_text(pythonSource)

	# TODO When resolving the ledger of imports, remove self-referential imports

def shatter_dataclassesDOTdataclass(logicalPathModule: strDotStrCuzPyStoopid, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier
									) -> tuple[ast.Name, LedgerOfImports, list[ast.AnnAssign], list[ast.Name], list[ast.keyword], ast.Tuple]:
	"""
	Parameters:
		logicalPathModule: gimme string cuz python is stoopid
		dataclass_Identifier: The identifier of the dataclass to be dismantled.
		instance_Identifier: In the synthesized module/function/scope, the identifier that will be used for the instance.
	"""
	module: ast.Module = ast.parse(inspect_getsource(import_module(logicalPathModule)))

	dataclass = next((statement for statement in module.body if ifThis.isClassDef_Identifier(dataclass_Identifier)(statement)), None)
	if not isinstance(dataclass, ast.ClassDef):
		raise ValueError(f"I could not find {dataclass_Identifier=} in {logicalPathModule=}.")

	list_astAnnAssign: list[ast.AnnAssign] = []
	listKeywordForDataclassInitialization: list[ast.keyword] = []
	list_astNameDataclassFragments: list[ast.Name] = []
	ledgerDataclassAndFragments = LedgerOfImports()

	addToLedgerPredicate = ifThis.isAnnAssignAndAnnotationIsName
	addToLedgerAction = Then.Z0Z_ledger(logicalPathModule, ledgerDataclassAndFragments)
	addToLedger = NodeCollector(addToLedgerPredicate, [addToLedgerAction])

	exclusionPredicate = ifThis.is_keyword_IdentifierEqualsConstantValue('init', False)
	appendKeywordAction = Then.Z0Z_appendKeywordMirroredTo(listKeywordForDataclassInitialization)
	filteredAppendKeywordAction = executeActionUnlessDescendantMatches(exclusionPredicate, appendKeywordAction)

	collector = NodeCollector(
			ifThis.isAnnAssignAndTargetIsName,
				[Then.Z0Z_appendAnnAssignOfNameDOTnameTo(instance_Identifier, list_astAnnAssign)
				, Then.append_targetTo(list_astNameDataclassFragments)
				, lambda node: addToLedger.visit(node)
				, filteredAppendKeywordAction
				]
			)

	collector.visit(dataclass)

	ledgerDataclassAndFragments.addImportFromStr(logicalPathModule, dataclass_Identifier)

	astNameDataclass = Make.astName(dataclass_Identifier)
	astTupleForAssignTargetsToFragments: ast.Tuple = Make.astTuple(list_astNameDataclassFragments, ast.Store())
	return astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign, list_astNameDataclassFragments, listKeywordForDataclassInitialization, astTupleForAssignTargetsToFragments
