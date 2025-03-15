""" As of 2025-03-15
Tools for transforming Python code from one format to another.

Scope of the tools:
- What is necessary to transform the baseline algorithm into optimized formats used by the official package.

Aspirations:
- Each tool is abstracted or generic enough to be used beyond the scope of the official package.
- Each tool is designed to be used in a modular fashion, allowing for the creation of new tools by combining existing tools.
- If a tool has a default setting, the setting shall be the setting used by the official package.
- Each tool is designed to be used in a way that is easy to understand and use.

Prior notes I'll integrate and delete:
- Settings for synthesizing the modules used by the package (i.e., the flow for numba)
- Settings for synthesizing modules that could be used by the package (e.g., the flow for JAX)
- Therefore, an abstracted system for creating settings for the package
- And with only a little more effort, an abstracted system for creating settings to synthesize arbitrary subsets of modules for arbitrary packages
"""
from collections import defaultdict
from collections.abc import Callable, Container, Sequence
from inspect import getsource as inspect_getsource
from mapFolding.theSSOT import getSourceAlgorithm, theDataclassIdentifier, theDataclassInstance, theDispatcherCallable, theFileExtension, theFormatStrModuleForCallableSynthetic, theFormatStrModuleSynthetic, theLogicalPathModuleDataclass, theLogicalPathModuleDispatcherSynthetic, theModuleDispatcherSynthetic, theModuleOfSyntheticModules, thePackageName, thePathPackage, theSourceSequentialCallable
from pathlib import PurePosixPath
from types import ModuleType
from typing import Any, cast, TypeAlias, TypeGuard, TypeVar
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
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
list_ast_type_paramORintORNone: TypeAlias = Any
strORintORNone: TypeAlias = Any
Z0Z_thisCannotBeTheBestWay: TypeAlias = list[ast.Name] | list[ast.Attribute] | list[ast.Subscript] | list[ast.Name | ast.Attribute] | list[ast.Name | ast.Subscript] | list[ast.Attribute | ast.Subscript] | list[ast.Name | ast.Attribute | ast.Subscript]

# listAsNode

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

def Z0Z_descendantContainsMatchingNode(node: ast.AST, predicateFunction: Callable[[ast.AST], bool]) -> bool:
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

def Z0Z_executeActionUnlessDescendantMatches(exclusionPredicate: Callable[[ast.AST], bool], actionFunction: Callable[[ast.AST], None]) -> Callable[[ast.AST], None]:
	"""
	Return a new action that will execute actionFunction only if no descendant (or the node itself)
	matches exclusionPredicate.
	"""
	def wrappedAction(node: ast.AST) -> None:
		if not Z0Z_descendantContainsMatchingNode(node, exclusionPredicate):
			actionFunction(node)
	return wrappedAction

class ifThis:
	@staticmethod
	def anyOf(*somePredicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda nodeTarget: any(predicate(nodeTarget) for predicate in somePredicates)

	@staticmethod
	def ast_IdentifierIsIn(container: Container[ast_Identifier]) -> Callable[[ast_Identifier], bool]:
		return lambda node: node in container

	# TODO is this only useable if namespace is not `None`? Yes, but use "" for namespace if necessary.
	@staticmethod
	def CallDoesNotCallItself(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda nodeFocus: ifThis.CallReallyIs(namespace, identifier)(nodeFocus) and 1 == sum(1 for descendant in ast.walk(nodeFocus) if ifThis.CallReallyIs(namespace, identifier)(descendant))

	@staticmethod
	def CallDoesNotCallItselfAndNameDOTidIsIn(container: Container[ast_Identifier]) -> Callable[[ast.AST], bool]:
		return lambda nodeSubject: (ifThis.isCall(nodeSubject) and ifThis.isName(nodeSubject.func) and ifThis.ast_IdentifierIsIn(container)(nodeSubject.func.id) and ifThis.CallDoesNotCallItself("", nodeSubject.func.id)(nodeSubject))

	@staticmethod
	def CallReallyIs(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.isCall_Identifier(identifier), ifThis.isCallNamespace_Identifier(namespace, identifier))

	@staticmethod
	def is_keyword(node: ast.AST) -> TypeGuard[ast.keyword]:
		return isinstance(node, ast.keyword)

	@staticmethod
	def is_keywordAndValueIsConstant(nodeCheck: ast.AST) -> TypeGuard[ast.keyword]:
		return ifThis.is_keyword(nodeCheck) and ifThis.isConstant(nodeCheck.value)

	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda nodeInstant: ifThis.is_keyword(nodeInstant) and nodeInstant.arg == identifier
	@staticmethod
	def is_keyword_IdentifierEqualsConstantValue(identifier: ast_Identifier, ConstantValue: Any) -> Callable[[ast.AST], bool]:
		return lambda astNode: (ifThis.is_keyword_Identifier(identifier)(astNode) and ifThis.is_keywordAndValueIsConstant(astNode) and ifThis.isConstantEquals(ConstantValue)(astNode.value))

	@staticmethod
	def isAnnAssign(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return isinstance(node, ast.AnnAssign)

	@staticmethod
	def isAnnAssignAndAnnotationIsName(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return ifThis.isAnnAssign(node) and ifThis.isName(node.annotation)

	@staticmethod
	def isAnnAssignAndTargetIsName(whatNode: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return ifThis.isAnnAssign(whatNode) and ifThis.isName(whatNode.target)

	@staticmethod
	def isAnnAssignTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda nodeStop: ifThis.isAnnAssign(nodeStop) and ifThis.NameReallyIs(identifier)(nodeStop.target)

	@staticmethod
	def isAnyAssignmentTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.isAssignOnlyTo(identifier), ifThis.isAnnAssignTo(identifier), ifThis.isAugAssignTo(identifier))

	@staticmethod
	def isAssign(node: ast.AST) -> TypeGuard[ast.Assign]:
		return isinstance(node, ast.Assign)

	@staticmethod
	def isAssignOnlyTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda aNode: ifThis.isAssign(aNode) and ifThis.NameReallyIs(identifier)(aNode.targets[0])

	@staticmethod
	def isAttribute(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return isinstance(node, ast.Attribute)

	@staticmethod
	def isAugAssign(node: ast.AST) -> TypeGuard[ast.AugAssign]:
		return isinstance(node, ast.AugAssign)

	@staticmethod
	def isAugAssignTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda nodeQuestion: ifThis.isAugAssign(nodeQuestion) and ifThis.NameReallyIs(identifier)(nodeQuestion.target)

	@staticmethod
	def isCall(node: ast.AST) -> TypeGuard[ast.Call]:
		return isinstance(node, ast.Call)

	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return lambda ImaNode: ifThis.isCall(ImaNode) and ifThis.isName_Identifier(identifier)(ImaNode.func)

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
	def astClassDef(name: ast_Identifier, listBases: list[ast.expr]=[], list_keyword: list[ast.keyword]=[], body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], **keywordArguments: list_ast_type_paramORintORNone) -> ast.ClassDef:
		"""keywordArguments: type_params:list[ast.type_param], lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.ClassDef(name=name, bases=listBases, keywords=list_keyword, body=body, decorator_list=decorator_list, **keywordArguments)

	@staticmethod
	def astFunctionDef(name: ast_Identifier, argumentsSpecification: ast.arguments=ast.arguments(), body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], returns: ast.expr|None=None, **keywordArguments: strORlist_ast_type_paramORintORNone) -> ast.FunctionDef:
		"""keywordArguments: type_comment:str|None, type_params:list[ast.type_param], lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.FunctionDef(name=name, args=argumentsSpecification, body=body, decorator_list=decorator_list, returns=returns, **keywordArguments)

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
		return ast.Module(body, type_ignores)

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
		if not isinstance(astImport_, (ast.Import, ast.ImportFrom)): # pyright: ignore[reportUnnecessaryIsInstance]
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

	def exportListModuleNames(self) -> list[str]:
		listModuleNames: list[str] = list(self.dictionaryImportFrom.keys())
		listModuleNames.extend(self.listImport)
		return sorted(set(listModuleNames))

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
	def removeThis(node: ast.AST) -> None:
		return None
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
	def appendTo(listAST: Sequence[ast.AST]) -> Callable[[ast.AST], None]:
		return lambda node: list(listAST).append(node)
	@staticmethod
	def Z0Z_appendAnnAssignOfNameDOTnameTo(identifier: ast_Identifier, listNameDOTname: list[ast.AnnAssign]) -> Callable[[ast.AST], None]:
		return lambda node: listNameDOTname.append(Make.astAnnAssign(cast(ast.AnnAssign, node).target, cast(ast.AnnAssign, node).annotation, Make.nameDOTname(identifier, cast(ast.Name, cast(ast.AnnAssign, node).target).id)))

	# TODO When resolving the ledger of imports, remove self-referential imports

@dataclasses.dataclass
class IngredientsFunction:
	"""Everything necessary to integrate a function into a module should be here."""
	FunctionDef: ast.FunctionDef # hint `Make.astFunctionDef`
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsModule:
	"""Everything necessary to create one _logical_ `ast.Module` should be here.
	Extrinsic qualities should be handled externally, such as with `RecipeModule`."""
	# If an `ast.Module` had a logical name that would be reasonable, but Python is firmly opposed
	# to a reasonable namespace, therefore, Hunter, you were silly to add a `name` field to this
	# dataclass for building an `ast.Module`.
	# name: ast_Identifier
	# Hey, genius, note that this is dataclasses.InitVar
	ingredientsFunction: dataclasses.InitVar[Sequence[IngredientsFunction] | IngredientsFunction | None] = None

	# `body` attribute of `ast.Module`
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	prologue: list[ast.stmt] = dataclasses.field(default_factory=list)
	functions: list[ast.FunctionDef | ast.stmt] = dataclasses.field(default_factory=list)
	epilogue: list[ast.stmt] = dataclasses.field(default_factory=list)
	launcher: list[ast.stmt] = dataclasses.field(default_factory=list)

	# parameter for `ast.Module` constructor
	type_ignores: list[ast.TypeIgnore] = dataclasses.field(default_factory=list)

	def __post_init__(self, ingredientsFunction: Sequence[IngredientsFunction] | IngredientsFunction | None = None) -> None:
		if ingredientsFunction is not None:
			if isinstance(ingredientsFunction, IngredientsFunction):
				self.addIngredientsFunction(ingredientsFunction)
			else:
				self.addIngredientsFunction(*ingredientsFunction)

	def addIngredientsFunction(self, *ingredientsFunction: IngredientsFunction) -> None:
		"""Add one or more `IngredientsFunction`. """
		listLedgers: list[LedgerOfImports] = []
		for definition in ingredientsFunction:
			self.functions.append(definition.FunctionDef)
			listLedgers.append(definition.imports)
		self.imports.update(*listLedgers)

	def _makeModuleBody(self) -> list[ast.stmt]:
		body: list[ast.stmt] = []
		body.extend(self.imports.makeListAst())
		body.extend(self.prologue)
		body.extend(self.functions)
		body.extend(self.epilogue)
		body.extend(self.launcher)
		# TODO `launcher`, if it exists, must start with `if __name__ == '__main__':` and be indented
		return body

	def export(self) -> ast.Module:
		"""Create a new `ast.Module` from the ingredients."""
		return Make.astModule(self._makeModuleBody(), self.type_ignores)

@dataclasses.dataclass
class Z0Z_RecipeSynthesizeFlow:
	"""Settings for synthesizing flow."""
	# TODO consider `IngredientsFlow` or similar
	# ========================================
	# Source
	sourceAlgorithm: ModuleType = getSourceAlgorithm()
	sourcePython: str = inspect_getsource(sourceAlgorithm)
	source_astModule: ast.Module = ast.parse(sourcePython)
	# https://github.com/hunterhogan/mapFolding/issues/4
	sourceDispatcherCallable: str = theDispatcherCallable
	sourceSequentialCallable: str = theSourceSequentialCallable
	sourceDataclassIdentifier: str = theDataclassIdentifier
	# I still hate the OOP paradigm. But I like this dataclass stuff.

	# ========================================
	# Filesystem
	pathPackage: PurePosixPath = PurePosixPath(thePathPackage)
	fileExtension: str = theFileExtension

	# ========================================
	# Logical identifiers
	# meta
	formatStrModuleSynthetic: str = theFormatStrModuleSynthetic
	formatStrModuleForCallableSynthetic: str = theFormatStrModuleForCallableSynthetic

	# Package
	packageName: ast_Identifier = thePackageName

	# Module
	# https://github.com/hunterhogan/mapFolding/issues/4
	Z0Z_flowLogicalPathRoot: str = theModuleOfSyntheticModules
	moduleDispatcher: str = theModuleDispatcherSynthetic
	logicalPathModuleDataclass: str = theLogicalPathModuleDataclass
	# https://github.com/hunterhogan/mapFolding/issues/4
	# `theLogicalPathModuleDispatcherSynthetic` is a problem. It is defined in theSSOT, but it can also be calculated.
	logicalPathModuleDispatcher: str = theLogicalPathModuleDispatcherSynthetic

	# Function
	sequentialCallable: str = sourceSequentialCallable
	dataclassIdentifier: str = sourceDataclassIdentifier
	dispatcherCallable: str = sourceDispatcherCallable

	# Variable
	dataclassInstance: str = theDataclassInstance

def extractClassDef(identifier: ast_Identifier, module: ast.Module) -> ast.ClassDef | None:
	sherpa: list[ast.ClassDef] = []
	extractor = NodeCollector(ifThis.isClassDef_Identifier(identifier), [Then.appendTo(sherpa)])
	extractor.visit(module)
	astClassDef = sherpa[0] if sherpa else None
	return astClassDef

def extractFunctionDef(identifier: ast_Identifier, module: ast.Module) -> ast.FunctionDef | None:
	sherpa: list[ast.FunctionDef] = []
	extractor = NodeCollector(ifThis.isFunctionDef_Identifier(identifier), [Then.appendTo(sherpa)])
	extractor.visit(module)
	astClassDef = sherpa[0] if sherpa else None
	return astClassDef

