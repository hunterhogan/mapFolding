from collections.abc import Callable, Container
from mapFolding.someAssemblyRequired import ast_Identifier, astClassHasDOTnameNotName, astClassHasDOTtarget, astClassHasDOTvalue, nodeType, ast_expr_Slice
from typing import Any, TypeAlias, TypeGuard
import ast
"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
Hunter: ^^ did you do that ^^ ? Are you sure? You just fixed some "Name" identifiers that should have been "_name" because the wrong case confused you.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a `TypeAlias` for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""

Z0Z_got_Identifier: TypeAlias = ast.alias | ast.arg | ast.Attribute | ast.ImportFrom | ast.keyword | ast.Name | astClassHasDOTnameNotName
class 又:
	@staticmethod
	def annotation(predicate: Callable[[nodeType | None], TypeGuard[nodeType] | bool]) -> Callable[[ast.AnnAssign | ast.arg], TypeGuard[ast.AnnAssign | ast.arg] | bool]:
		return lambda node: predicate(node.annotation)
	@staticmethod
	def arg(predicate: Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.arg | ast.keyword], TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool]:
		return lambda node: predicate(node.arg)
	@staticmethod
	def asname(predicate: Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.alias], TypeGuard[ast.alias] | bool]:
		return lambda node: predicate(node.asname)
	@staticmethod
	def attr(predicate: Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.Attribute], TypeGuard[ast.Attribute] | bool]:
		return lambda node: predicate(node.attr)
	@staticmethod
	def func(predicate: Callable[[nodeType], TypeGuard[nodeType] | bool]) -> Callable[[ast.Call], TypeGuard[ast.Call] | bool]:
		return lambda node: predicate(node.func)
	@staticmethod
	def id(predicate: Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.Name], TypeGuard[ast.Name] | bool]:
		return lambda node: predicate(node.id)
	@staticmethod
	def module(predicate: Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.ImportFrom], TypeGuard[ast.ImportFrom] | bool]:
		return lambda node: predicate(node.module)
	@staticmethod
	def name(predicate: Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]) -> Callable[[astClassHasDOTnameNotName], TypeGuard[astClassHasDOTnameNotName] | bool]:
		return lambda node: predicate(node.name)
	@staticmethod
	def slice(predicate: Callable[[ast_expr_Slice], TypeGuard[ast_expr_Slice] | bool]) -> Callable[[ast.Subscript], TypeGuard[ast.Subscript] | bool]:
		return lambda node: predicate(node.slice)
	@staticmethod
	def target(predicate: Callable[[nodeType], TypeGuard[nodeType] | bool]) -> Callable[[astClassHasDOTtarget], TypeGuard[astClassHasDOTtarget] | bool]:
		return lambda node: predicate(node.target)
	@staticmethod
	def value(predicate: Callable[[nodeType | Any | None], TypeGuard[nodeType] | bool]) -> Callable[[astClassHasDOTvalue], TypeGuard[astClassHasDOTvalue] | bool]:
		return lambda node: predicate(node.value)
	@staticmethod
	def isAllOf(*thesePredicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: all(predicate(node) for predicate in thesePredicates)
	@staticmethod
	def isAnyOf(*thesePredicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: any(predicate(node) for predicate in thesePredicates)

class ifThis:
	@staticmethod
	def isAssignAndTargets0Is(targets0Predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		"""node is Assign and node.targets[0] matches `targets0Predicate`."""
		return lambda node: ifThis.isAssign(node) and targets0Predicate(node.targets[0])
	@staticmethod
	def isAssignAndValueIs(valuePredicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		"""node is ast.Assign and node.value matches `valuePredicate`.
		Parameters:
			valuePredicate: Function that evaluates the value of the assignment
		Returns:
			predicate: matches assignments with values meeting the criteria
		"""
		return lambda node: ifThis.isAssign(node) and 又.value(valuePredicate)(node)
	@staticmethod
	def _Identifier(identifier: ast_Identifier) -> Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node == identifier
	@staticmethod
	def equals(this: Any) -> Callable[[Any], TypeGuard[Any] | bool]:
		return lambda node: node == this
	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.FunctionDef] | bool]:
		return lambda node: ifThis.isFunctionDef(node) and 又.name(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Name] | bool]:
		return lambda node: ifThis.isName(node) and 又.id(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isArgument_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool]:
		return lambda node: (ifThis.is_arg(node) or ifThis.is_keyword(node)) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: ifThis.is_keyword(node) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def is_arg_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: ifThis.is_arg(node) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.ClassDef] | bool]:
		return lambda node: ifThis.isClassDef(node) and 又.name(ifThis._Identifier(identifier))(node)
	@staticmethod
	def is_nameDOTnameNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		return lambda node: ifThis.is_nameDOTname(node) and 又.value(ifThis.isName_Identifier(namespace))(node) and 又.attr(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: ifThis.isCall(node) and 又.func(ifThis.isName_Identifier(identifier))(node)
	@staticmethod
	def isCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: ifThis.isCall(node) and 又.func(ifThis.is_nameDOTnameNamespace_Identifier(namespace, identifier))(node)
	@staticmethod
	def isAssignAndValueIsCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return lambda node: ifThis.isAssign(node) and 又.value(ifThis.isCall_Identifier(identifier))(node)
	@staticmethod
	def isAssignAndValueIsCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return ifThis.isAssignAndValueIs(ifThis.isCallNamespace_Identifier(namespace, identifier))
	@staticmethod
	def CallReallyIs(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return 又.isAnyOf(ifThis.isCall_Identifier(identifier), ifThis.isCallNamespace_Identifier(namespace, identifier))
	@staticmethod
	def ast_IdentifierIn(container: Container[ast_Identifier]) -> Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node in container
	@staticmethod
	def CallDoesNotCallItself(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		"""If `namespace` is not applicable to your case, then call with `namespace=""`."""
		return lambda node: ifThis.matchesMeButNotAnyDescendant(ifThis.CallReallyIs(namespace, identifier))(node)
	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[ast.Call]:
		return ifThis.isCall(node) and 又.func(ifThis.isName)(node)
	@staticmethod
	def CallDoesNotCallItselfAndNameDOTidIsIn(container: Container[ast_Identifier]) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: ifThis.isCallToName(node) and 又.func(又.id(ifThis.ast_IdentifierIn(container)))(node) and ifThis.CallDoesNotCallItself("", node.func.id)(node)
	@staticmethod
	def is_arg(node: ast.AST) -> TypeGuard[ast.arg]:
		return isinstance(node, ast.arg)
	@staticmethod
	def is_keyword(node: ast.AST) -> TypeGuard[ast.keyword]:
		return isinstance(node, ast.keyword)
	@staticmethod
	def is_keywordAndValueIsConstant(node: ast.AST) -> TypeGuard[ast.keyword]:
		return ifThis.is_keyword(node) and 又.value(ifThis.isConstant)(node)
	@staticmethod
	def is_keyword_IdentifierEqualsConstantValue(identifier: ast_Identifier, ConstantValue: Any) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		return lambda node: ifThis.is_keyword_Identifier(identifier)(node) and ifThis.is_keywordAndValueIsConstant(node) and 又.value(ifThis.isConstantEquals(ConstantValue))(node)
	@staticmethod
	def is_nameDOTname(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return ifThis.isAttribute(node) and 又.value(ifThis.isName)(node)
	@staticmethod
	def isAnnAssign(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return isinstance(node, ast.AnnAssign)
	@staticmethod
	def isAnnAssignAndAnnotationIsName(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return ifThis.isAnnAssign(node) and 又.annotation(ifThis.isName)(node)
	@staticmethod
	def isAnnAssignAndTargetIsName(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return ifThis.isAnnAssign(node) and 又.target(ifThis.isName)(node)
	@staticmethod
	def isAnyCompare(node: ast.AST) -> TypeGuard[ast.Compare] | TypeGuard[ast.BoolOp]:
		return ifThis.isCompare(node) or ifThis.isBoolOp(node)
	@staticmethod
	def isAssign(node: ast.AST) -> TypeGuard[ast.Assign]:
		return isinstance(node, ast.Assign)
	@staticmethod
	def isAttribute(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return isinstance(node, ast.Attribute)
	@staticmethod
	def isAugAssign(node: ast.AST) -> TypeGuard[ast.AugAssign]:
		return isinstance(node, ast.AugAssign)
	@staticmethod
	def isBoolOp(node: ast.AST) -> TypeGuard[ast.BoolOp]:
		return isinstance(node, ast.BoolOp)
	@staticmethod
	def isCall(node: ast.AST) -> TypeGuard[ast.Call]:
		return isinstance(node, ast.Call)
	@staticmethod
	def isClassDef(node: ast.AST) -> TypeGuard[ast.ClassDef]:
		return isinstance(node, ast.ClassDef)
	@staticmethod
	def isCompare(node: ast.AST) -> TypeGuard[ast.Compare]:
		return isinstance(node, ast.Compare)
	@staticmethod
	def isConstant(node: ast.AST) -> TypeGuard[ast.Constant]:
		return isinstance(node, ast.Constant)
	@staticmethod
	def isConstantEquals(value: Any) -> Callable[[ast.AST], TypeGuard[ast.Constant] | bool]:
		return lambda node: ifThis.isConstant(node) and 又.value(ifThis.equals(value))(node)
	@staticmethod
	def isExpr(node: ast.AST) -> TypeGuard[ast.Expr]:
		return isinstance(node, ast.Expr)
	@staticmethod
	def isFunctionDef(node: ast.AST) -> TypeGuard[ast.FunctionDef]: return isinstance(node, ast.FunctionDef)
	@staticmethod
	def isImport(node: ast.AST) -> TypeGuard[ast.Import]:
		return isinstance(node, ast.Import)
	@staticmethod
	def isReturn(node: ast.AST) -> TypeGuard[ast.Return]:
		return isinstance(node, ast.Return)
	@staticmethod
	def isReturnAnyCompare(node: ast.AST) -> TypeGuard[ast.Return]:
		return ifThis.isReturn(node) and node.value is not None and 又.value(ifThis.isAnyCompare)(node)
	@staticmethod
	def isReturnUnaryOp(node: ast.AST) -> TypeGuard[ast.Return]:
		return ifThis.isReturn(node) and node.value is not None and 又.value(ifThis.isUnaryOp)(node)
	@staticmethod
	def isSubscript(node: ast.AST) -> TypeGuard[ast.Subscript]:
		return isinstance(node, ast.Subscript)
	@staticmethod
	def isUnaryOp(node: ast.AST) -> TypeGuard[ast.UnaryOp]:
		return isinstance(node, ast.UnaryOp)
	@staticmethod
	def isName(node: ast.AST) -> TypeGuard[ast.Name]:
		return isinstance(node, ast.Name)
	@staticmethod
	def isStarred(node: ast.AST) -> TypeGuard[ast.Starred]:
		return isinstance(node, ast.Starred)
	@staticmethod
	def _nestedJunction_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | TypeGuard[ast.Subscript] | TypeGuard[ast.Starred] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute] | TypeGuard[ast.Subscript] | TypeGuard[ast.Starred] | bool:
			return ifThis.isName_Identifier(identifier)(node) or ifThis.isAttribute_Identifier(identifier)(node) or ifThis.isSubscript_Identifier(identifier)(node) or ifThis.isStarred_Identifier(identifier)(node)
		return workhorse
	@staticmethod
	def isAttribute_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		"""node is `ast.Attribute` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute]:
			return ifThis.isAttribute(node) and 又.value(ifThis._nestedJunction_Identifier(identifier))(node)
		return workhorse
	@staticmethod
	def isStarred_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Starred] | bool]:
		"""node is `ast.Starred` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Starred]:
			return ifThis.isStarred(node) and 又.value(ifThis._nestedJunction_Identifier(identifier))(node)
		return workhorse
	@staticmethod
	def isSubscript_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript] | bool]:
		"""node is `ast.Subscript` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Subscript]:
			return ifThis.isSubscript(node) and 又.value(ifThis._nestedJunction_Identifier(identifier))(node)
		return workhorse
	# TODO Does this work?
	@staticmethod
	def Z0Z_matchesAtLeast1Descendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if any descendant of the node matches the given predicate."""
		return lambda node: not ifThis.matchesNoDescendant(predicate)(node)
	# TODO Does this work?
	@staticmethod
	def Z0Z_matchesMeAndMyDescendantsExactlyNTimes(predicate: Callable[[ast.AST], bool], nTimes: int) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if exactly 'nTimes' nodes in the tree match the predicate."""
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
		def workhorse(node: ast.AST) -> bool:
			for descendant in ast.walk(node):
				if descendant is not node and predicate(descendant):
					return False
			return True
		return workhorse

	# For function inlining
	@staticmethod
	def onlyReturnAnyCompare(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return ifThis.isFunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnAnyCompare(astFunctionDef.body[0])
	# For function inlining
	@staticmethod
	def onlyReturnUnaryOp(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return ifThis.isFunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnUnaryOp(astFunctionDef.body[0])
	@staticmethod
	def Z0Z_unparseIs(astAST: ast.AST) -> Callable[[ast.AST], bool]:
		def workhorse(node: ast.AST) -> bool: return ast.unparse(node) == ast.unparse(astAST)
		return workhorse
