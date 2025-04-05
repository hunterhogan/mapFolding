from collections.abc import Callable
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	astClassHasDOTnameNotName,
	astClassHasDOTtarget,
	astClassHasDOTtargetAttributeNameSubscript,
	astClassHasDOTtarget_expr,
	astClassHasDOTvalue,
	astClassHasDOTvalue_expr,
	astClassOptionallyHasDOTnameNotName,
	astClassHasDOTvalue_exprNone,
	TypeCertified,
)
from typing import Any, overload, TypeGuard
import ast

class DOT:
	"""
	class `DOT` : give only the attribute or sub-node to anything, including a `Callable`. Usable anywhere.
	class `grab`: bring the `Callable` to the node and its attribute or sub-node. Not for antecedents.
	"""
	@staticmethod
	@overload
	def annotation(node: ast.AnnAssign) -> ast.expr:...
	@staticmethod
	@overload
	def annotation(node: ast.arg) -> ast.expr | None:...
	@staticmethod
	def annotation(node: ast.AnnAssign | ast.arg) -> ast.expr | None:
		return node.annotation

	@staticmethod
	@overload
	def arg(node: ast.arg) -> ast_Identifier:...
	@staticmethod
	@overload
	def arg(node: ast.keyword) -> ast_Identifier | None:...
	@staticmethod
	def arg(node: ast.arg | ast.keyword) -> ast_Identifier | None:
		return node.arg

	@staticmethod
	def attr(node: ast.Attribute) -> ast_Identifier:
		return node.attr
	@staticmethod
	def func(node: ast.Call) -> ast.expr:
		return node.func
	@staticmethod
	def id(node: ast.Name) -> ast_Identifier:
		return node.id

	@staticmethod
	@overload
	def name(node: astClassHasDOTnameNotName) -> ast_Identifier:...
	@staticmethod
	@overload
	def name(node: astClassOptionallyHasDOTnameNotName) -> ast_Identifier | None:...
	@staticmethod
	def name(node: astClassHasDOTnameNotName | astClassOptionallyHasDOTnameNotName) -> ast_Identifier | None:
		return node.name

	@staticmethod
	@overload
	def target(node: ast.NamedExpr) -> ast.Name:...
	@staticmethod
	@overload
	def target(node: astClassHasDOTtarget_expr) -> ast.expr:...
	@staticmethod
	@overload
	def target(node: astClassHasDOTtargetAttributeNameSubscript) -> ast.Attribute | ast.Name | ast.Subscript:...
	@staticmethod
	def target(node: astClassHasDOTtarget) -> ast.Attribute | ast.expr | ast.Name | ast.Subscript:
		return node.target

	@staticmethod
	@overload
	def value(node: ast.Constant) -> Any:...
	@staticmethod
	@overload
	def value(node: ast.MatchSingleton) -> bool | None:...
	@staticmethod
	@overload
	def value(node: astClassHasDOTvalue_expr) -> ast.expr:...
	@staticmethod
	@overload
	def value(node: astClassHasDOTvalue_exprNone) -> ast.expr | None:...
	@staticmethod
	def value(node: astClassHasDOTvalue) -> Any | ast.expr | bool | None:
		return node.value

class be:
	@staticmethod
	def _typeCertified(antecedent: type[TypeCertified]) -> Callable[[ast.AST | None], TypeGuard[TypeCertified]]:
		def workhorse(node: ast.AST | None) -> TypeGuard[TypeCertified]:
			return isinstance(node, antecedent)
		return workhorse
	@staticmethod
	def AnnAssign(node: ast.AST) -> TypeGuard[ast.AnnAssign]: return be._typeCertified(ast.AnnAssign)(node)
	@staticmethod
	def arg(node: ast.AST) -> TypeGuard[ast.arg]: return be._typeCertified(ast.arg)(node)
	@staticmethod
	def Assign(node: ast.AST) -> TypeGuard[ast.Assign]: return be._typeCertified(ast.Assign)(node)
	@staticmethod
	def Attribute(node: ast.AST) -> TypeGuard[ast.Attribute]: return be._typeCertified(ast.Attribute)(node)
	@staticmethod
	def AugAssign(node: ast.AST) -> TypeGuard[ast.AugAssign]: return be._typeCertified(ast.AugAssign)(node)
	@staticmethod
	def BoolOp(node: ast.AST) -> TypeGuard[ast.BoolOp]: return be._typeCertified(ast.BoolOp)(node)
	@staticmethod
	def Call(node: ast.AST) -> TypeGuard[ast.Call]: return be._typeCertified(ast.Call)(node)
	@staticmethod
	def ClassDef(node: ast.AST) -> TypeGuard[ast.ClassDef]: return be._typeCertified(ast.ClassDef)(node)
	@staticmethod
	def Compare(node: ast.AST) -> TypeGuard[ast.Compare]: return be._typeCertified(ast.Compare)(node)
	@staticmethod
	def Constant(node: ast.AST) -> TypeGuard[ast.Constant]: return be._typeCertified(ast.Constant)(node)
	@staticmethod
	def Expr(node: ast.AST) -> TypeGuard[ast.Expr]: return be._typeCertified(ast.Expr)(node)
	@staticmethod
	def FunctionDef(node: ast.AST) -> TypeGuard[ast.FunctionDef]: return be._typeCertified(ast.FunctionDef)(node)
	@staticmethod
	def Import(node: ast.AST) -> TypeGuard[ast.Import]: return be._typeCertified(ast.Import)(node)
	@staticmethod
	def ImportFrom(node: ast.AST) -> TypeGuard[ast.ImportFrom]: return be._typeCertified(ast.ImportFrom)(node)
	@staticmethod
	def keyword(node: ast.AST) -> TypeGuard[ast.keyword]: return be._typeCertified(ast.keyword)(node)
	@staticmethod
	def Module(node: ast.AST) -> TypeGuard[ast.Module]: return be._typeCertified(ast.Module)(node)
	@staticmethod
	def Name(node: ast.AST) -> TypeGuard[ast.Name]: return be._typeCertified(ast.Name)(node)
	@staticmethod
	def Return(node: ast.AST) -> TypeGuard[ast.Return]: return be._typeCertified(ast.Return)(node)
	@staticmethod
	def Starred(node: ast.AST) -> TypeGuard[ast.Starred]: return be._typeCertified(ast.Starred)(node)
	@staticmethod
	def Subscript(node: ast.AST) -> TypeGuard[ast.Subscript]: return be._typeCertified(ast.Subscript)(node)
	@staticmethod
	def UnaryOp(node: ast.AST) -> TypeGuard[ast.UnaryOp]: return be._typeCertified(ast.UnaryOp)(node)

class ifThis:
	@staticmethod
	def _Identifier(identifier: ast_Identifier) -> Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node == identifier
	@staticmethod
	def _nested_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute | ast.Starred | ast.Subscript] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute | ast.Starred | ast.Subscript] | bool:
			return ifThis.isName_Identifier(identifier)(node) or ifThis.isAttribute_Identifier(identifier)(node) or ifThis.isSubscript_Identifier(identifier)(node) or ifThis.isStarred_Identifier(identifier)(node)
		return workhorse
	@staticmethod
	def is_arg_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.arg(node) and ifThis._Identifier(identifier)(DOT.arg(node))
	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.keyword(node) and ifThis._Identifier(identifier)(DOT.arg(node))
	@staticmethod
	def isAnnAssign_targetIs(targetPredicate: Callable[[ast.expr], TypeGuard[ast.expr] | bool]) -> Callable[[ast.AST], TypeGuard[ast.AnnAssign] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.AnnAssign] | bool:
			return be.AnnAssign(node) and targetPredicate(DOT.target(node))
		return workhorse
	@staticmethod
	def isAnnAssignAndAnnotationIsName(node: ast.AST) -> TypeGuard[ast.AnnAssign] | bool:
		return be.AnnAssign(node) and be.Name(DOT.annotation(node))
	@staticmethod
	def isArgument_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg | ast.keyword] | bool]:
		return lambda node: (be.arg(node) or be.keyword(node)) and ifThis._Identifier(identifier)(DOT.arg(node))
	@staticmethod
	def isAssignAndTargets0Is(targets0Predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.AnnAssign] | bool]:
		"""node is Assign and node.targets[0] matches `targets0Predicate`."""
		return lambda node: be.Assign(node) and targets0Predicate(node.targets[0])
	@staticmethod
	def isAssignAndValueIs(valuePredicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		"""node is ast.Assign and node.value matches `valuePredicate`. """
		return lambda node: be.Assign(node) and valuePredicate(DOT.value(node))
	@staticmethod
	def isAssignAndValueIsCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return lambda node: be.Assign(node) and ifThis.isCall_Identifier(identifier)(DOT.value(node))
	@staticmethod
	def isAssignAndValueIsCallAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return ifThis.isAssignAndValueIs(ifThis.isCallAttributeNamespace_Identifier(namespace, identifier))
	@staticmethod
	def isAttribute_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		"""node is `ast.Attribute` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute]:
			return be.Attribute(node) and ifThis._nested_Identifier(identifier)(DOT.value(node))
		return workhorse
	@staticmethod
	def isAttributeName(node: ast.AST) -> TypeGuard[ast.Attribute]:
		""" Displayed as Name.attribute."""
		return be.Attribute(node) and be.Name(DOT.value(node))
	@staticmethod
	def isAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		return lambda node: ifThis.isAttributeName(node) and ifThis.isName_Identifier(namespace)(DOT.value(node)) and ifThis._Identifier(identifier)(DOT.attr(node))
	@staticmethod
	def isAugAssign_targetIs(targetPredicate: Callable[[ast.expr], TypeGuard[ast.expr] | bool]) -> Callable[[ast.AST], TypeGuard[ast.AugAssign] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.AugAssign] | bool:
			return be.AugAssign(node) and targetPredicate(DOT.target(node))
		return workhorse
	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: be.Call(node) and ifThis.isName_Identifier(identifier)(DOT.func(node))
	@staticmethod
	def isCallAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: be.Call(node) and ifThis.isAttributeNamespace_Identifier(namespace, identifier)(DOT.func(node))
	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[ast.Call]:
		return be.Call(node) and be.Name(DOT.func(node))
	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.ClassDef] | bool]:
		return lambda node: be.ClassDef(node) and ifThis._Identifier(identifier)(DOT.name(node))
	@staticmethod
	def isConstantEquals(value: Any) -> Callable[[ast.AST], TypeGuard[ast.Constant] | bool]:
		return lambda node: be.Constant(node) and DOT.value(node) == value
	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.FunctionDef] | bool]:
		return lambda node: be.FunctionDef(node) and ifThis._Identifier(identifier)(DOT.name(node))
	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Name] | bool]:
		return lambda node: be.Name(node) and ifThis._Identifier(identifier)(DOT.id(node))
	@staticmethod
	def isStarred_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Starred] | bool]:
		"""node is `ast.Starred` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Starred]:
			return be.Starred(node) and ifThis._nested_Identifier(identifier)(DOT.value(node))
		return workhorse
	@staticmethod
	def isSubscript_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript] | bool]:
		"""node is `ast.Subscript` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Subscript]:
			return be.Subscript(node) and ifThis._nested_Identifier(identifier)(DOT.value(node))
		return workhorse
	@staticmethod
	def equals(this: Any) -> Callable[[Any], TypeGuard[Any] | bool]:
		return lambda node: node == this
	@staticmethod
	def matchesAtLeast1Descendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if any descendant of the node matches the given predicate."""
		return lambda node: not ifThis.matchesNoDescendant(predicate)(node)
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
	@staticmethod
	def Z0Z_unparseIs(astAST: ast.AST) -> Callable[[ast.AST], bool]:
		def workhorse(node: ast.AST) -> bool: return ast.unparse(node) == ast.unparse(astAST)
		return workhorse
