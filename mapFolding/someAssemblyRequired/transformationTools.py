"""As of 2025-03-15
Tools for transforming Python code from one format to another.

Scope:
- What is necessary to transform the baseline algorithm into optimized formats used by the official package.

Aspirations:
- Each tool is abstracted or generic enough to be used beyond the scope of the official package.
- Each tool is designed to be used in a modular fashion, allowing for the creation of new tools by combining existing tools.
- If a tool has a default setting, the setting shall be the setting used by the official package.
"""
from collections import defaultdict
from collections.abc import Callable, Container, Generator, Sequence
from contextlib import contextmanager
from inspect import getsource as inspect_getsource
from mapFolding.theSSOT import (
	getSourceAlgorithm,
	theDataclassIdentifier,
	theDataclassInstance,
	theDataclassInstanceTaskDistribution,
	theDispatcherCallable,
	theFileExtension,
	theFormatStrModuleForCallableSynthetic,
	theFormatStrModuleSynthetic,
	theLogicalPathModuleDataclass,
	theLogicalPathModuleDispatcherSynthetic,
	theModuleDispatcherSynthetic,
	theModuleOfSyntheticModules,
	thePackageName,
	thePathPackage,
	theSourceInitializeCallable,
	theSourceParallelCallable,
	theSourceSequentialCallable,
)
from pathlib import PurePosixPath
from types import ModuleType
from typing import Any, Generic, cast, TypeAlias, TypeGuard, TypeVar
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import dataclasses

"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
Hunter: ^^ did you do that ^^ ? Are you sure? You just fixed some that should have been "_name" because it confused you.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a TypeAlias for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""

# Would `LibCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7

nodeType = TypeVar('nodeType', bound=ast.AST)
ast_Identifier: TypeAlias = str
ast_expr_Slice: TypeAlias = ast.expr
strDotStrCuzPyStoopid: TypeAlias = str
strORlist_ast_type_paramORintORNone: TypeAlias = Any
list_ast_type_paramORintORNone: TypeAlias = Any
strORintORNone: TypeAlias = Any
astClassHasAttributeDOTname: TypeAlias = ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef
astMosDef = TypeVar('astMosDef', bound=astClassHasAttributeDOTname)

@contextmanager
def listAsNode(ImaList: list[ast.stmt]) -> Generator[ast.Module, None, None]:
	"""Convert a list of AST nodes into a module for use in node walking tools."""
	ImaModule = ast.Module(ImaList)
	try:
		yield ImaModule
	finally:
		ImaList = ImaModule.body

class NodeCollector(Generic[nodeType], ast.NodeVisitor):
	"""A node visitor that collects data via one or more actions when a predicate is met."""
	def __init__(self, findThis: Callable[[ast.AST], TypeGuard[nodeType] | bool], doThat: list[Callable[[nodeType], Any]]) -> None:
		self.findThis = findThis
		self.doThat = doThat

	def visit(self, node: ast.AST) -> None:
		if self.findThis(node):
			for action in self.doThat:
				action(cast(nodeType, node))
		self.generic_visit(node)

ImaViolationOfTheLiskovSubstitutionPrinciple: TypeAlias = ast.AST
class NodeReplacer(Generic[nodeType], ast.NodeTransformer):
	"""A node transformer that replaces or removes AST nodes based on a condition."""
	def __init__(self, findThis: Callable[[ast.AST], TypeGuard[nodeType] | bool], doThat: Callable[[nodeType], ast.AST | Sequence[ast.AST] | None]) -> None:
		self.findThis = findThis
		self.doThat = doThat

	def visit(self, node: ImaViolationOfTheLiskovSubstitutionPrinciple) -> ast.AST | Sequence[ast.AST] | None:
		if self.findThis(node):
			return self.doThat(cast(nodeType, node))
		return super().visit(node)

class ifThis:
	@staticmethod
	def ast_IdentifierIsIn(container: Container[ast_Identifier]) -> Callable[[ast_Identifier], TypeGuard[ast_Identifier]]:
		return lambda node: node in container # type: ignore

	@staticmethod
	def CallDoesNotCallItself(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call]]:
		"""If `namespace` is not applicable to your case, then call with `namespace=""`."""
		return lambda node: ifThis.matchesMeButNotAnyDescendant(ifThis.CallReallyIs(namespace, identifier))(node) # type: ignore

	@staticmethod
	def CallDoesNotCallItselfAndNameDOTidIsIn(container: Container[ast_Identifier]) -> Callable[[ast.AST], TypeGuard[ast.Call]]:
		return lambda node: ifThis.isCall(node) and ifThis.isName(node.func) and ifThis.ast_IdentifierIsIn(container)(node.func.id) and ifThis.CallDoesNotCallItself("", node.func.id)(node) # type: ignore

	@staticmethod
	def CallReallyIs(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call]]:
		return ifThis.isAnyOf(ifThis.isCall_Identifier(identifier), ifThis.isCallNamespace_Identifier(namespace, identifier)) # type: ignore

	@staticmethod
	def is_keyword(node: ast.AST) -> TypeGuard[ast.keyword]:
		return isinstance(node, ast.keyword)

	@staticmethod
	def is_keywordAndValueIsConstant(node: ast.AST) -> TypeGuard[ast.keyword]:
		return ifThis.is_keyword(node) and ifThis.isConstant(node.value)

	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.keyword]]:
		return lambda node: ifThis.is_keyword(node) and node.arg == identifier # type: ignore

	@staticmethod
	def is_keyword_IdentifierEqualsConstantValue(identifier: ast_Identifier, ConstantValue: Any) -> Callable[[ast.AST], TypeGuard[ast.keyword]]:
		return lambda node: (ifThis.is_keyword_Identifier(identifier)(node) and ifThis.is_keywordAndValueIsConstant(node) and ifThis.isConstantEquals(ConstantValue)(node.value)) # type: ignore

	@staticmethod
	def isAllOf(*thesePredicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: all(predicate(node) for predicate in thesePredicates)

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
	def isAnnAssignTo(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.AnnAssign]]:
		return lambda node: ifThis.isAnnAssign(node) and ifThis.NameReallyIs_Identifier(identifier)(node.target) # type: ignore

	@staticmethod
	def isAnyAssignmentTo(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		return ifThis.isAnyOf(ifThis.isAssignOnlyTo(identifier), ifThis.isAnnAssignTo(identifier), ifThis.isAugAssignTo(identifier))

	@staticmethod
	def isAnyCompare(node: ast.AST) -> TypeGuard[ast.Compare] | TypeGuard[ast.BoolOp]:
		return ifThis.isCompare(node) or ifThis.isBoolOp(node)

	@staticmethod
	def isAnyOf(*thesePredicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: any(predicate(node) for predicate in thesePredicates)

	@staticmethod
	def isAssign(node: ast.AST) -> TypeGuard[ast.Assign]:
		return isinstance(node, ast.Assign)

	@staticmethod
	def isAssignAndValueIsCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign]]:
		return lambda node: ifThis.isAssign(node) and ifThis.isCall_Identifier(identifier)(node.value) # type: ignore

	# @staticmethod
	# def isAssignAndValueIsCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign]]:
	# 	return lambda node: ifThis.isAssign(node) and ifThis.isCallNamespace_Identifier(namespace, identifier)(node.value)

	@staticmethod
	def isAssignAndValueIsCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign]]:
		return ifThis.isAssignAndValueIs(ifThis.isCallNamespace_Identifier(namespace, identifier))

	@staticmethod
	def isAssignOnlyTo(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign]]:
		return lambda node: ifThis.isAssign(node) and ifThis.NameReallyIs_Identifier(identifier)(node.targets[0]) # type: ignore

	@staticmethod
	def isAssignAndTargets0Is(targets0Predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.Assign]]:
		"""node is Assign and node.targets[0] matches `targets0Predicate`."""
		return lambda node: ifThis.isAssign(node) and targets0Predicate(node.targets[0]) # type: ignore

	@staticmethod
	def isAssignAndValueIs(valuePredicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.Assign]]:
		"""node is Assign and node.value matches `valuePredicate`.

		Parameters:
			valuePredicate: Function that evaluates the value of the assignment

		Returns:
			A predicate that matches assignments with values meeting the criteria
		"""
		return lambda node: ifThis.isAssign(node) and valuePredicate(node.value) # type: ignore

	@staticmethod
	def isAttribute(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return isinstance(node, ast.Attribute)

	@staticmethod
	def isAugAssign(node: ast.AST) -> TypeGuard[ast.AugAssign]:
		return isinstance(node, ast.AugAssign)

	@staticmethod
	def isAugAssignTo(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.AugAssign]]:
		return lambda node: ifThis.isAugAssign(node) and ifThis.NameReallyIs_Identifier(identifier)(node.target) # type: ignore

	@staticmethod
	def isBoolOp(node: ast.AST) -> TypeGuard[ast.BoolOp]:
		return isinstance(node, ast.BoolOp)

	@staticmethod
	def isCall(node: ast.AST) -> TypeGuard[ast.Call]:
		return isinstance(node, ast.Call)

	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call]]:
		return lambda node: ifThis.isCall(node) and ifThis.isName_Identifier(identifier)(node.func) # type: ignore

	@staticmethod
	def isCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call]]:
		return lambda node: ifThis.isCall(node) and ifThis.is_nameDOTnameNamespace_Identifier(namespace, identifier)(node.func) # type: ignore

	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[ast.Call]:
		return ifThis.isCall(node) and ifThis.isName(node.func)

	@staticmethod
	def isClassDef(node: ast.AST) -> TypeGuard[ast.ClassDef]:
		return isinstance(node, ast.ClassDef)

	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.ClassDef]]:
		return lambda node: ifThis.isClassDef(node) and node.name == identifier # type: ignore

	@staticmethod
	def isCompare(node: ast.AST) -> TypeGuard[ast.Compare]:
		return isinstance(node, ast.Compare)

	@staticmethod
	def isConstant(node: ast.AST) -> TypeGuard[ast.Constant]:
		return isinstance(node, ast.Constant)

	@staticmethod
	def isConstantEquals(value: Any) -> Callable[[ast.AST], TypeGuard[ast.Constant]]:
		return lambda node: ifThis.isConstant(node) and node.value == value # type: ignore

	@staticmethod
	def isExpr(node: ast.AST) -> TypeGuard[ast.Expr]:
		return isinstance(node, ast.Expr)

	@staticmethod
	def isFunctionDef(node: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return isinstance(node, ast.FunctionDef)

	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.FunctionDef]]:
		return lambda node: ifThis.isFunctionDef(node) and node.name == identifier # type: ignore

	@staticmethod
	def isImport(node: ast.AST) -> TypeGuard[ast.Import]:
		return isinstance(node, ast.Import)

# A "name" is roughly ast._Identifier

# A Name can be a "name"
# ast.Name()

# These classes can be a "name": I will call the group 'ClassQuizas'
# The technical mechanism is `type[ast.ClassQuizas.value] == ast.Name`
# or, and this is the annoying part,
# `type[ast.ClassQuizas.value] == ast.ClassQuizas and type[ast.ClassQuizas.value.value] == ast.ClassQuizas and ... == ast.Name`
# ast.Attribute()
# ast.Subscript()
# ast.Starred()

	@staticmethod
	def isName(node: ast.AST) -> TypeGuard[ast.Name]:
		return isinstance(node, ast.Name)

	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Name]]:
		return lambda node: ifThis.isName(node) and node.id == identifier # type: ignore

	@staticmethod
	def is_nameDOTname(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return ifThis.isAttribute(node) and ifThis.isName(node.value)

	@staticmethod
	def is_nameDOTnameNamespace(namespace: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute]]:
		return lambda node: (ifThis.is_nameDOTname(node) and ifThis.isName_Identifier(namespace)(node.value)) # type: ignore

	@staticmethod
	def is_nameDOTnameNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute]]:
		return lambda node: (ifThis.is_nameDOTnameNamespace(namespace)(node) and node.attr == identifier) # type: ignore

	@staticmethod
	def isAnyNestedName(node: ast.AST):
		return ifThis.isAnyOf()

	@staticmethod
	def NameReallyIs_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], bool]:
		# The following logic is incomplete.
		return ifThis.isAnyOf(ifThis.isName_Identifier(identifier), ifThis.isSubscriptIsName_Identifier(identifier))

	@staticmethod
	def isReturn(node: ast.AST) -> TypeGuard[ast.Return]:
		return isinstance(node, ast.Return)

	@staticmethod
	def isReturnAnyCompare(node: ast.AST) -> TypeGuard[ast.Return]:
		return ifThis.isReturn(node) and node.value is not None and ifThis.isAnyCompare(node.value)

	@staticmethod
	def isReturnUnaryOp(node: ast.AST) -> TypeGuard[ast.Return]:
		return ifThis.isReturn(node) and node.value is not None and ifThis.isUnaryOp(node.value)

	@staticmethod
	def isSubscript(node: ast.AST) -> TypeGuard[ast.Subscript]:
		return isinstance(node, ast.Subscript)

	@staticmethod
	def isSubscript_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript]]:
		"""node is `ast.Subscript` and the top-level `ast.Name` is `identifier`
		Parameters:
			identifier: The identifier to look for in the value chain

		Returns:
			A predicate function that checks if a node matches the criteria
		"""
		def predicate(node: ast.AST) -> TypeGuard[ast.Subscript]:
			if not ifThis.isSubscript(node):
				return False
			def checkNodeDOTvalue(nodeDOTvalue: ast.AST) -> bool:
				if ifThis.isName(nodeDOTvalue):
					if nodeDOTvalue.id == identifier:
						return True
				elif hasattr(nodeDOTvalue, "value"):
					return checkNodeDOTvalue(nodeDOTvalue.value) # type: ignore
				return False
			return checkNodeDOTvalue(node.value)
		return predicate

	@staticmethod
	def isSubscriptIsName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript]]:
		return lambda node: ifThis.isSubscript(node) and ifThis.isName_Identifier(identifier)(node.value) # type: ignore

	@staticmethod
	def isSubscript_Identifier_Identifier(identifier: ast_Identifier, sliceIdentifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript]]:
		return lambda node: ifThis.isSubscript(node) and ifThis.isName_Identifier(identifier)(node.value) and ifThis.isName_Identifier(sliceIdentifier)(node.slice) # type: ignore

	@staticmethod
	def isUnaryOp(node: ast.AST) -> TypeGuard[ast.UnaryOp]:
		return isinstance(node, ast.UnaryOp)

	# Does this work?
	@staticmethod
	def matchesAtLeast1Descendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if any descendant of the node matches the given predicate."""
		return lambda node: not ifThis.matchesNoDescendant(predicate)(node)

	# Does this work?
	@staticmethod
	def matchesMeAndMyDescendantsExactlyNTimes(predicate: Callable[[ast.AST], bool], nTimes: int) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if exactly 'count' nodes in the tree match the predicate."""
		def countMatchingNodes(node: ast.AST) -> bool:
			matches = sum(1 for descendant in ast.walk(node) if predicate(descendant))
			return matches == nTimes
		return countMatchingNodes

	@staticmethod
	def matchesMeButNotAnyDescendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if the node matches but none of its descendants match the predicate."""
		return lambda node: predicate(node) and ifThis.matchesNoDescendant(predicate)(node)

	@staticmethod
	def matchesNoDescendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if no descendant of the node matches the given predicate."""
		def checkNoMatchingDescendant(node: ast.AST) -> bool:
			for descendant in ast.walk(node):
				if descendant is not node and predicate(descendant):
					return False
			return True
		return checkNoMatchingDescendant

	@staticmethod
	def OnlyReturnAnyCompare(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return ifThis.isFunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnAnyCompare(astFunctionDef.body[0])

	@staticmethod
	def OnlyReturnUnaryOp(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return ifThis.isFunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnUnaryOp(astFunctionDef.body[0])

class Make:
	@staticmethod
	def ast_arg(identifier: ast_Identifier, annotation: ast.expr | None = None, **keywordArguments: strORintORNone) -> ast.arg:
		"""keywordArguments: type_comment:str|None, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.arg(identifier, annotation, **keywordArguments)

	@staticmethod
	def ast_keyword(keywordArgument: ast_Identifier, value: ast.expr, **keywordArguments: int) -> ast.keyword:
		return ast.keyword(arg=keywordArgument, value=value, **keywordArguments)

	@staticmethod
	def astAlias(name: ast_Identifier, asname: ast_Identifier | None = None) -> ast.alias:
		return ast.alias(name, asname)

	@staticmethod
	def astAnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None, **keywordArguments: int) -> ast.AnnAssign:
		"""`simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't add it as a parameter."""
		return ast.AnnAssign(target, annotation, value, simple=int(isinstance(target, ast.Name)), **keywordArguments)

	@staticmethod
	def astAssign(listTargets: Any, value: ast.expr, **keywordArguments: strORintORNone) -> ast.Assign:
		"""keywordArguments: type_comment:str|None, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.Assign(targets=listTargets, value=value, **keywordArguments)

	@staticmethod
	def astArgumentsSpecification(posonlyargs: list[ast.arg]=[], args: list[ast.arg]=[], vararg: ast.arg|None=None, kwonlyargs: list[ast.arg]=[], kw_defaults: list[ast.expr|None]=[None], kwarg: ast.arg|None=None, defaults: list[ast.expr]=[]) -> ast.arguments:
		return ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)

	@staticmethod
	def astAttribute(value: ast.expr, attribute: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:
		"""
		Parameters:
			value: the part before the dot (hint `ast.Name` for nameDOTname)
			attribute: the `str` after the dot
			context (ast.Load()): Load/Store/Del"""
		return ast.Attribute(value, attribute, context, **keywordArguments)

	@staticmethod
	def astCall(caller: ast.Name | ast.Attribute, listArguments: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call:
		return ast.Call(func=caller, args=list(listArguments) if listArguments else [], keywords=list(list_astKeywords) if list_astKeywords else [])

	@staticmethod
	def astClassDef(name: ast_Identifier, listBases: list[ast.expr]=[], list_keyword: list[ast.keyword]=[], body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], **keywordArguments: list_ast_type_paramORintORNone) -> ast.ClassDef:
		"""keywordArguments: type_params:list[ast.type_param], lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.ClassDef(name=name, bases=listBases, keywords=list_keyword, body=body, decorator_list=decorator_list, **keywordArguments)

	@staticmethod
	def astConstant(value: Any, **keywordArguments: strORintORNone) -> ast.Constant:
		"""value: str|int|float|bool|None|bytes|bytearray|memoryview|complex|list|tuple|dict|set, or any other type that can be represented as a constant in Python.
		keywordArguments: kind:str, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.Constant(value, **keywordArguments)

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
	def astModule(body: list[ast.stmt], type_ignores: list[ast.TypeIgnore] = []) -> ast.Module:
		return ast.Module(body, type_ignores)

	@staticmethod
	def astName(identifier: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Name:
		return ast.Name(identifier, context, **keywordArguments)

	@staticmethod
	def itDOTname(nameChain: ast.Name | ast.Attribute, dotName: str) -> ast.Attribute:
		return ast.Attribute(value=nameChain, attr=dotName, ctx=ast.Load())

	@staticmethod
	# TODO rewrite with all parameters
	def nameDOTname(identifier: ast_Identifier, *dotName: str) -> ast.Name | ast.Attribute:
		nameDOTname: ast.Name | ast.Attribute = Make.astName(identifier)
		if not dotName:
			return nameDOTname
		for suffix in dotName:
			nameDOTname = Make.itDOTname(nameDOTname, suffix)
		return nameDOTname

	@staticmethod
	def astReturn(value: ast.expr | None = None, **keywordArguments: int) -> ast.Return:
		return ast.Return(value, **keywordArguments)

	@staticmethod
	def astSubscript(value: ast.expr, slice: ast_expr_Slice, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Subscript:
		return ast.Subscript(value, slice, ctx=context, **keywordArguments)

	@staticmethod
	def astTuple(elements: Sequence[ast.expr], context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Tuple:
	# def astTuple(elements: Sequence[ast.expr], context: ast.expr_context | None = None, **keywordArguments: int) -> ast.Tuple:
		"""context: Load/Store/Del"""
		# context = context or ast.Load()
		return ast.Tuple(elts=list(elements), ctx=context, **keywordArguments)

class LedgerOfImports:
	# TODO When resolving the ledger of imports, remove self-referential imports

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
		"""Update this ledger with imports from one or more other ledgers.
		Parameters:
			*fromLedger: One or more other `LedgerOfImports` objects from which to merge.
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
	def append_targetTo(listName: list[ast.AST]) -> Callable[[ast.AnnAssign], None]:
		return lambda node: listName.append(node.target)
	@staticmethod
	def appendTo(listOfAny: list[Any]) -> Callable[[ast.AST], None]:
		return lambda node: listOfAny.append(node)
	@staticmethod
	def insertThisAbove(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda aboveMe: [*list_astAST, aboveMe]
	@staticmethod
	def insertThisBelow(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda belowMe: [belowMe, *list_astAST]
	@staticmethod
	def removeThis(_node: ast.AST) -> None:
		return None
	@staticmethod
	def replaceWith(astAST: ast.AST) -> Callable[[ast.AST], ast.AST]:
		return lambda _replaceMe: astAST
	@staticmethod
	def updateThis(dictionaryOf_astMosDef: dict[ast_Identifier, astMosDef]) -> Callable[[astMosDef], astMosDef]:
		return lambda node: dictionaryOf_astMosDef.setdefault(node.name, node)
	@staticmethod
	def Z0Z_ledger(logicalPath: strDotStrCuzPyStoopid, ledger: LedgerOfImports) -> Callable[[ast.AnnAssign], None]:
		return lambda node: ledger.addImportFromStr(logicalPath, node.annotation.id) # type: ignore
	@staticmethod
	def Z0Z_appendKeywordMirroredTo(list_keyword: list[ast.keyword]) -> Callable[[ast.AnnAssign], None]:
		return lambda node: list_keyword.append(Make.ast_keyword(node.target.id, node.target)) # type: ignore
	@staticmethod
	def Z0Z_appendAnnAssignOf_nameDOTnameTo(identifier: ast_Identifier, list_nameDOTname: list[ast.AnnAssign]) -> Callable[[ast.AnnAssign], None]:
		return lambda node: list_nameDOTname.append(Make.astAnnAssign(node.target, node.annotation, Make.nameDOTname(identifier, node.target.id))) # type: ignore

	# TODO When resolving the ledger of imports, remove self-referential imports

@dataclasses.dataclass
class IngredientsFunction:
	"""Everything necessary to integrate a function into a module should be here."""
	astFunctionDef: ast.FunctionDef # hint `Make.astFunctionDef`
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsModule:
	"""Everything necessary to create one _logical_ `ast.Module` should be here.
	Extrinsic qualities should _probably_ be handled externally."""
	# If an `ast.Module` had a logical name that would be reasonable, but Python is firmly opposed
	# to a reasonable namespace, therefore, Hunter, you were silly to add a `name` field to this
	# dataclass for building an `ast.Module`.
	# name: ast_Identifier
	# Hey, genius, note that this is dataclasses.InitVar
	ingredientsFunction: dataclasses.InitVar[Sequence[IngredientsFunction] | IngredientsFunction | None] = None

	# init var with an existing module? method to deconstruct an existing module?

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
		"""Add one or more `IngredientsFunction`."""
		listLedgers: list[LedgerOfImports] = []
		for definition in ingredientsFunction:
			self.functions.append(definition.astFunctionDef)
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
class RecipeSynthesizeFlow:
	"""Settings for synthesizing flow."""
	# ========================================
	# Source
	sourceAlgorithm: ModuleType = getSourceAlgorithm()
	sourcePython: str = inspect_getsource(sourceAlgorithm)
	source_astModule: ast.Module = ast.parse(sourcePython)

	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	sourceDispatcherCallable: str = theDispatcherCallable
	sourceInitializeCallable: str = theSourceInitializeCallable
	sourceParallelCallable: str = theSourceParallelCallable
	sourceSequentialCallable: str = theSourceSequentialCallable

	sourceDataclassIdentifier: str = theDataclassIdentifier
	sourceDataclassInstance: str = theDataclassInstance
	sourceDataclassInstanceTaskDistribution: str = theDataclassInstanceTaskDistribution
	sourcePathModuleDataclass: str = theLogicalPathModuleDataclass

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
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	Z0Z_flowLogicalPathRoot: str = theModuleOfSyntheticModules
	moduleDispatcher: str = theModuleDispatcherSynthetic
	logicalPathModuleDataclass: str = sourcePathModuleDataclass
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	# `theLogicalPathModuleDispatcherSynthetic` is a problem. It is defined in theSSOT, but it can also be calculated.
	logicalPathModuleDispatcher: str = theLogicalPathModuleDispatcherSynthetic

	# Function
	dispatcherCallable: str = sourceDispatcherCallable
	initializeCallable: str = sourceInitializeCallable
	parallelCallable: str = sourceParallelCallable
	sequentialCallable: str = sourceSequentialCallable

	dataclassIdentifier: str = sourceDataclassIdentifier

	# Variable
	dataclassInstance: str = sourceDataclassInstance

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

def makeDictionaryFunctionDef(module: ast.Module) -> dict[ast_Identifier, ast.FunctionDef]:
	dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = {}
	NodeCollector(ifThis.isFunctionDef, [Then.updateThis(dictionaryFunctionDef)]).visit(module)
	return dictionaryFunctionDef

def makeDictionaryReplacementStatements(module: ast.Module) -> dict[ast_Identifier, ast.stmt | list[ast.stmt]]:
	"""Return a dictionary of function names and their replacement statements."""
	dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = makeDictionaryFunctionDef(module)
	dictionaryReplacementStatements: dict[ast_Identifier, ast.stmt | list[ast.stmt]] = {}
	for name, astFunctionDef in dictionaryFunctionDef.items():
		if ifThis.OnlyReturnAnyCompare(astFunctionDef):
			dictionaryReplacementStatements[name] = astFunctionDef.body[0].value # type: ignore
		elif ifThis.OnlyReturnUnaryOp(astFunctionDef):
			dictionaryReplacementStatements[name] = astFunctionDef.body[0].value # type: ignore
		else:
			dictionaryReplacementStatements[name] = astFunctionDef.body[0:-1]
	return dictionaryReplacementStatements

def Z0Z_descendantContainsMatchingNode(node: ast.AST, predicateFunction: Callable[[ast.AST], bool]) -> bool:
	"""Return True if any descendant of the node (or the node itself) matches the predicateFunction."""
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
	"""Return a new action that will execute actionFunction only if no descendant (or the node itself) matches exclusionPredicate."""
	def wrappedAction(node: ast.AST) -> None:
		if not Z0Z_descendantContainsMatchingNode(node, exclusionPredicate):
			actionFunction(node)
	return wrappedAction
