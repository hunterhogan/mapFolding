from collections.abc import Callable, Container
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	astClassHasDOTnameNotName,
	astClassHasDOTtarget,
	astClassHasDOTvalue,
	astClassOptionallyHasDOTnameNotName,
	astClassOptionallyHasDOTvalue,
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
	def target(node: astClassHasDOTtarget) -> ast.Name | ast.Attribute | ast.Subscript | ast.expr:
		return node.target
	@staticmethod
	@overload
	def value(node: astClassHasDOTvalue) -> ast.expr:...
	@staticmethod
	@overload
	def value(node: astClassOptionallyHasDOTvalue) -> ast.expr | Any | None:...
	@staticmethod
	def value(node: astClassHasDOTvalue | astClassOptionallyHasDOTvalue) -> ast.expr | Any | None | bool:
		return node.value

class be:
	@staticmethod
	def _typeCertified(antecedent: type[TypeCertified]) -> Callable[[Any | None], TypeGuard[TypeCertified]]:
		def workhorse(node: Any | None) -> TypeGuard[TypeCertified]:
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
	def equals(this: Any) -> Callable[[Any], TypeGuard[Any] | bool]:
		return lambda node: node == this
	@staticmethod
	def isAssignAndTargets0Is(targets0Predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.AnnAssign] | bool]:
		"""node is Assign and node.targets[0] matches `targets0Predicate`."""
		return lambda node: be.Assign(node) and targets0Predicate(node.targets[0])
	@staticmethod
	def isAssignAndValueIs(valuePredicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		"""node is ast.Assign and node.value matches `valuePredicate`.
		Parameters:
			valuePredicate: Function that evaluates the value of the assignment
		Returns:
			predicate: matches assignments with values meeting the criteria
		"""
		return lambda node: be.Assign(node) and valuePredicate(DOT.value(node))
	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.FunctionDef] | bool]:
		return lambda node: be.FunctionDef(node) and ifThis._Identifier(identifier)(DOT.name(node))
	@staticmethod
	def isArgument_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg | ast.keyword] | bool]:
		return lambda node: (be.arg(node) or be.keyword(node)) and ifThis._Identifier(identifier)(DOT.arg(node))
	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.keyword(node) and ifThis._Identifier(identifier)(DOT.arg(node))
	@staticmethod
	def is_arg_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.arg(node) and ifThis._Identifier(identifier)(DOT.arg(node))
	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.ClassDef] | bool]:
		return lambda node: be.ClassDef(node) and ifThis._Identifier(identifier)(DOT.name(node))
	@staticmethod
	def isAssignAndValueIsCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return lambda node: be.Assign(node) and ifThis.isCall_Identifier(identifier)(DOT.value(node))
	@staticmethod
	def isAssignAndValueIsCallAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return ifThis.isAssignAndValueIs(ifThis.isCallAttributeNamespace_Identifier(namespace, identifier))
	@staticmethod
	def is_keywordAndValueIsConstant(node: ast.AST) -> TypeGuard[ast.keyword]:
		return be.keyword(node) and be.Constant(DOT.value(node))
	@staticmethod
	def is_keyword_IdentifierEqualsConstantValue(identifier: ast_Identifier, ConstantValue: Any) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		return lambda node: ifThis.is_keyword_Identifier(identifier)(node) and ifThis.is_keywordAndValueIsConstant(node) and ifThis.isConstantEquals(ConstantValue)(DOT.value(node))
	@staticmethod
	def isAnnAssign_targetIs(targetPredicate: Callable[[ast.expr], TypeGuard[ast.expr] | bool]) -> Callable[[ast.AST], TypeGuard[ast.AnnAssign] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.AnnAssign] | bool:
			return be.AnnAssign(node) and targetPredicate(DOT.target(node))
		return workhorse
	@staticmethod
	def isAnnAssignAndAnnotationIsName(node: ast.AST) -> TypeGuard[ast.AnnAssign] | bool:
		return be.AnnAssign(node) and be.Name(DOT.annotation(node))
	@staticmethod
	def isAugAssign_targetIs(targetPredicate: Callable[[ast.expr], TypeGuard[ast.expr] | bool]) -> Callable[[ast.AST], TypeGuard[ast.AugAssign] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.AugAssign] | bool:
			return be.AugAssign(node) and targetPredicate(DOT.target(node))
		return workhorse

	@staticmethod
	def isAnyCompare(node: ast.AST) -> TypeGuard[ast.BoolOp | ast.Compare]:
		return be.BoolOp(node) or be.Compare(node)
	@staticmethod
	def isConstantEquals(value: Any) -> Callable[[ast.AST], TypeGuard[ast.Constant] | bool]:
		return lambda node: be.Constant(node) and DOT.value(node) == value
	@staticmethod
	def isReturnAnyCompare(node: ast.AST) -> TypeGuard[ast.Return] | bool:
		return be.Return(node) and ifThis.isAnyCompare(DOT.value(node)) # type: ignore
	@staticmethod
	def isReturnUnaryOp(node: ast.AST) -> TypeGuard[ast.Return] | bool:
		return be.Return(node) and be.UnaryOp(DOT.value(node)) # type: ignore

	# ================================================================
	# Nested identifier
	@staticmethod
	def _nestedJunction_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute | ast.Starred | ast.Subscript] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute | ast.Starred | ast.Subscript] | bool:
			return ifThis.isName_Identifier(identifier)(node) or ifThis.isAttribute_Identifier(identifier)(node) or ifThis.isSubscript_Identifier(identifier)(node) or ifThis.isStarred_Identifier(identifier)(node)
		return workhorse
	@staticmethod
	def isAttribute_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		"""node is `ast.Attribute` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute]:
			return be.Attribute(node) and ifThis._nestedJunction_Identifier(identifier)(DOT.value(node))
		return workhorse
	@staticmethod
	def isStarred_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Starred] | bool]:
		"""node is `ast.Starred` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Starred]:
			return be.Starred(node) and ifThis._nestedJunction_Identifier(identifier)(DOT.value(node))
		return workhorse
	@staticmethod
	def isSubscript_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript] | bool]:
		"""node is `ast.Subscript` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Subscript]:
			return be.Subscript(node) and ifThis._nestedJunction_Identifier(identifier)(DOT.value(node))
		return workhorse
	# ================================================================

	@staticmethod
	def Z0Z_unparseIs(astAST: ast.AST) -> Callable[[ast.AST], bool]:
		def workhorse(node: ast.AST) -> bool: return ast.unparse(node) == ast.unparse(astAST)
		return workhorse

	# ================================================================
	# NOT used
	@staticmethod
	def matchesAtLeast1Descendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if any descendant of the node matches the given predicate."""
		return lambda node: not ifThis.matchesNoDescendant(predicate)(node)
	# ================================================================
	# MORE function inlining
	@staticmethod
	def onlyReturnAnyCompare(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef] | bool:
		return be.FunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnAnyCompare(astFunctionDef.body[0])
	# For function inlining
	@staticmethod
	def onlyReturnUnaryOp(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef] | bool:
		return be.FunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnUnaryOp(astFunctionDef.body[0])
	# ================================================================
	# These are used by other functions
	@staticmethod
	def isCallAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: be.Call(node) and ifThis.isAttributeNamespace_Identifier(namespace, identifier)(DOT.func(node))
	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Name] | bool]:
		return lambda node: be.Name(node) and ifThis._Identifier(identifier)(DOT.id(node))
	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: be.Call(node) and ifThis.isName_Identifier(identifier)(DOT.func(node))
	# ================================================================
	@staticmethod
	def CallDoesNotCallItself(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		"""If `namespace` is not applicable to your case, then call with `namespace=""`."""
		return lambda node: ifThis.matchesMeButNotAnyDescendant(ifThis.CallReallyIs(namespace, identifier))(node)
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
	def CallReallyIs(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return ifThis.isCall_Identifier(identifier) or ifThis.isCallAttributeNamespace_Identifier(namespace, identifier)
	@staticmethod
	def isAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		return lambda node: ifThis.isAttributeName(node) and ifThis.isName_Identifier(namespace)(DOT.value(node)) and ifThis._Identifier(identifier)(DOT.attr(node))
	@staticmethod
	def _Identifier(identifier: ast_Identifier) -> Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node == identifier
	@staticmethod
	def isAttributeName(node: ast.AST) -> TypeGuard[ast.Attribute]:
		""" Displayed as Name.attribute."""
		return be.Attribute(node) and be.Name(DOT.value(node))

	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[ast.Call]:
		return be.Call(node) and be.Name(DOT.func(node))
	@staticmethod
	def ast_IdentifierIn(container: Container[ast_Identifier]) -> Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node in container
	# This bullshit is for the crappy function inliner I made.
	@staticmethod
	def CallDoesNotCallItselfAndNameDOTidIsIn(container: Container[ast_Identifier]) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: ifThis.isCallToName(node) and ifThis.ast_IdentifierIn(container)(DOT.id(DOT.func(node))) and ifThis.CallDoesNotCallItself("", node.func.id)(node) # type: ignore
