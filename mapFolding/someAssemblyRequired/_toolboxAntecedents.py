from collections.abc import Callable, Container
from mapFolding.someAssemblyRequired import ast_expr_Slice, ast_Identifier, astClassHasDOTnameNotName, astClassHasDOTtarget, astClassHasDOTvalue, ImaAnnotationType, typeCertified
from typing import Any, overload, TypeGuard
import ast

Ima_targetType = ast.AST

class 又:
	@staticmethod
	@overload
	def annotation(predicate: Callable[[ImaAnnotationType], ast.AST | ast_Identifier]) -> Callable[[ast.AnnAssign | ast.arg], ast.AST | ast_Identifier]:...
	@staticmethod
	@overload
	def annotation(predicate: Callable[[ImaAnnotationType], TypeGuard[ImaAnnotationType] | bool]) -> Callable[[ast.AnnAssign | ast.arg], TypeGuard[ast.AnnAssign] | TypeGuard[ast.arg] | bool]:...
	@staticmethod
	def annotation(predicate: Callable[[ImaAnnotationType], TypeGuard[ImaAnnotationType] | ast.AST | ast_Identifier | bool]) -> Callable[[ast.AnnAssign | ast.arg], TypeGuard[ast.AnnAssign] | TypeGuard[ast.arg] | ast.AST | ast_Identifier | bool]:
		@overload
		def workhorse(node: ast.AnnAssign | ast.arg) -> ast.AST | ast_Identifier:...
		@overload
		def workhorse(node: ast.AnnAssign | ast.arg) -> TypeGuard[ast.AnnAssign] | TypeGuard[ast.arg] | bool:...
		def workhorse(node: ast.AnnAssign | ast.arg) -> TypeGuard[ast.AnnAssign] | TypeGuard[ast.arg] | ast.AST | ast_Identifier | bool:
			ImaAnnotation = node.annotation
			if ImaAnnotation is None: return False
			assert be.Attribute(ImaAnnotation) or be.Constant(ImaAnnotation) or be.Name(ImaAnnotation) or be.Subscript(ImaAnnotation)
			# assert be.Annotation(ImaAnnotation)
			return predicate(ImaAnnotation)
		return workhorse
	@staticmethod
	@overload
	def arg(predicate: Callable[[ast_Identifier], ast.AST | ast_Identifier]) -> Callable[[ast.arg | ast.keyword], ast.AST | ast_Identifier]:...
	@staticmethod
	@overload
	def arg(predicate: Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.arg | ast.keyword], TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool]:...
	@staticmethod
	def arg(predicate: Callable[[ast_Identifier], TypeGuard[ast_Identifier] | ast.AST | ast_Identifier | bool]) -> Callable[[ast.arg | ast.keyword], TypeGuard[ast.arg] | TypeGuard[ast.keyword] | ast.AST | ast_Identifier | bool]:
		@overload
		def workhorse(node: ast.arg | ast.keyword) -> ast.AST | ast_Identifier:...
		@overload
		def workhorse(node: ast.arg | ast.keyword) -> TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool:...
		def workhorse(node: ast.arg | ast.keyword) -> TypeGuard[ast.arg] | TypeGuard[ast.keyword] | ast.AST | ast_Identifier | bool:
			Ima_arg = node.arg
			if Ima_arg is None: return False
			return predicate(Ima_arg)
		return workhorse
	@staticmethod
	def asname(predicate: Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.alias], TypeGuard[ast.alias] | bool]:
		return lambda node: predicate(node.asname)
	@staticmethod
	def attr(predicate: Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.Attribute], TypeGuard[ast.Attribute] | bool]:
		return lambda node: predicate(node.attr)
	@staticmethod
	def func(predicate: Callable[[ast.AST], TypeGuard[ast.AST] | bool]) -> Callable[[ast.Call], TypeGuard[ast.Call] | bool]:
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
	def target(predicate: Callable[[ast.AST], TypeGuard[ast.AST] | bool]) -> Callable[[astClassHasDOTtarget], TypeGuard[astClassHasDOTtarget] | bool]:
		return lambda node: predicate(node.target)
	@staticmethod
	def value(predicate: Callable[[ast.AST], TypeGuard[ast.AST] | bool]) -> Callable[[astClassHasDOTvalue], TypeGuard[astClassHasDOTvalue] | bool]:
		def workhorse(node: astClassHasDOTvalue) -> TypeGuard[astClassHasDOTvalue] | bool:
			ImaValue = node.value
			if ImaValue is None: return False
			return predicate(ImaValue)
		return workhorse

class be:
	@staticmethod
	def _typeCertified(antecedent: type[typeCertified]) -> Callable[[Any | None], TypeGuard[typeCertified]]:
		def workhorse(node: Any | None) -> TypeGuard[typeCertified]:
			return isinstance(node, antecedent)
		return workhorse
	@staticmethod
	def AnnAssign(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.AnnAssign)(node)
# 'TypeVar "typeCertified" appears only once in generic function signature. Use "object" instead Pylance(reportInvalidTypeVarUse)"' HOW THE FUCK IS THAT INVALID WHEN IT IS WORKING PERFECTLY TO PASS THE TYPE INFORMATION--IN YOUR FUCKING STATIC TYPE CHECKER, PYLANCE!!!! Fuck you, and fuck your pretentious language.
	@staticmethod
	def arg(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.arg)(node)

	# @staticmethod
	# def Annotation(node: ast.AST) -> TypeGuard[object] | bool:
	# 	if be.Attribute(node):
	# 		return be.Attribute(node)
	# 	elif be.Constant(node):
	# 		return be.Constant(node)
	# 	elif be.Name(node):
	# 		return be.Name(node)
	# 	elif be.Subscript(node):
	# 		return be.Subscript(node)
	# 	else:
	# 		return False
		# return be.Attribute(node) or be.Constant(node) or be.Name(node) or be.Subscript(node)

	@staticmethod
	def Assign(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Assign)(node)
	@staticmethod
	def Attribute(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Attribute)(node)
	@staticmethod
	def AugAssign(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.AugAssign)(node)
	@staticmethod
	def BoolOp(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.BoolOp)(node)
	@staticmethod
	def Call(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Call)(node)
	@staticmethod
	def ClassDef(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.ClassDef)(node)
	@staticmethod
	def Compare(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Compare)(node)
	@staticmethod
	def Constant(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Constant)(node)
	@staticmethod
	def Expr(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Expr)(node)
	@staticmethod
	def FunctionDef(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.FunctionDef)(node)
	@staticmethod
	def Import(node: ast.AST) -> TypeGuard[ast.Import]: return be._typeCertified(ast.Import)(node)
	@staticmethod
	def ImportFrom(node: ast.AST) -> TypeGuard[ast.ImportFrom]: return be._typeCertified(ast.ImportFrom)(node)
	@staticmethod
	def keyword(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.keyword)(node)
	@staticmethod
	def Module(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Module)(node)
	@staticmethod
	def Name(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Name)(node)
	@staticmethod
	def Return(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Return)(node)
	@staticmethod
	def Starred(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Starred)(node)
	@staticmethod
	def Subscript(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.Subscript)(node)
	@staticmethod
	def UnaryOp(node: ast.AST) -> TypeGuard[object]: return be._typeCertified(ast.UnaryOp)(node)

class ifThis:
	@staticmethod
	def equals(this: Any) -> Callable[[Any], TypeGuard[Any] | bool]:
		return lambda node: node == this
	@staticmethod
	def isAssignAndTargets0Is(targets0Predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		"""node is Assign and node.targets[0] matches `targets0Predicate`."""
		return lambda node: be.Assign(node) and targets0Predicate(node.targets[0])
	@staticmethod
	def isAssignAndValueIs(valuePredicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		"""node is ast.Assign and node.value matches `valuePredicate`.
		Parameters:
			valuePredicate: Function that evaluates the value of the assignment
		Returns:
			predicate: matches assignments with values meeting the criteria
		"""
		return lambda node: be.Assign(node) and 又.value(valuePredicate)(node)
	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: be.FunctionDef(node) and 又.name(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isArgument_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: (be.arg(node) or be.keyword(node)) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.keyword(node) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def is_arg_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.arg(node) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: be.ClassDef(node) and 又.name(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isAssignAndValueIsCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: be.Assign(node) and 又.value(ifThis.isCall_Identifier(identifier))(node)
	@staticmethod
	def isAssignAndValueIsCallAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return ifThis.isAssignAndValueIs(ifThis.isCallAttributeNamespace_Identifier(namespace, identifier))
	@staticmethod
	def is_keywordAndValueIsConstant(node: ast.AST) -> TypeGuard[object]:
		return be.keyword(node) and 又.value(be.Constant)(node)
	@staticmethod
	def is_keyword_IdentifierEqualsConstantValue(identifier: ast_Identifier, ConstantValue: Any) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: ifThis.is_keyword_Identifier(identifier)(node) and ifThis.is_keywordAndValueIsConstant(node) and 又.value(ifThis.isConstantEquals(ConstantValue))(node)
	"""
Argument of type "typeCertified@isAnnAssign_targetIs" cannot be assigned to parameter of type "astClassHasDOTtarget"
	Type "typeCertified@isAnnAssign_targetIs" is not assignable to type "astClassHasDOTtarget"
		"object*" is not assignable to "AnnAssign"
		"object*" is not assignable to "AsyncFor"
		"object*" is not assignable to "AugAssign"
		"object*" is not assignable to "comprehension"
		"object*" is not assignable to "For"
		"object*" is not assignable to "NamedExpr"
	"""
	@staticmethod
	def isAnnAssign_targetIs(targetPredicate: Callable[[Ima_targetType], TypeGuard[Ima_targetType] | bool]) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[object] | bool:
			return be.AnnAssign(node) and 又.target(targetPredicate)(node)
		return workhorse
	@staticmethod
	def isAnnAssignAndAnnotationIsName(node: ast.AST) -> TypeGuard[object] | bool:
		return be.AnnAssign(node) and 又.annotation(be.Name)(node)

	@staticmethod
	def isAnyCompare(node: ast.AST) -> TypeGuard[object]:
		return be.Compare(node) or be.BoolOp(node)
	@staticmethod
	def isConstantEquals(value: Any) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: be.Constant(node) and 又.value(ifThis.equals(value))(node)
	@staticmethod
	def isReturnAnyCompare(node: ast.AST) -> TypeGuard[object] | bool:
		return be.Return(node) and 又.value(ifThis.isAnyCompare)(node)
	@staticmethod
	def isReturnUnaryOp(node: ast.AST) -> TypeGuard[object] | bool:
		return be.Return(node) and 又.value(be.UnaryOp)(node)

	# ================================================================
	# Nested identifier
	@staticmethod
	def _nestedJunction_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[object] | bool:
			return ifThis.isName_Identifier(identifier)(node) or ifThis.isAttribute_Identifier(identifier)(node) or ifThis.isSubscript_Identifier(identifier)(node) or ifThis.isStarred_Identifier(identifier)(node)
		return workhorse
	@staticmethod
	def isAttribute_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		"""node is `ast.Attribute` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[object]:
			return be.Attribute(node) and 又.value(ifThis._nestedJunction_Identifier(identifier))(node)
		return workhorse
	@staticmethod
	def isStarred_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		"""node is `ast.Starred` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[object]:
			return be.Starred(node) and 又.value(ifThis._nestedJunction_Identifier(identifier))(node)
		return workhorse
	@staticmethod
	def isSubscript_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		"""node is `ast.Subscript` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[object]:
			return be.Subscript(node) and 又.value(ifThis._nestedJunction_Identifier(identifier))(node)
		return workhorse
	# ================================================================

	@staticmethod
	def Z0Z_unparseIs(astAST: ast.AST) -> Callable[[ast.AST], bool]:
		def workhorse(node: ast.AST) -> bool: return ast.unparse(node) == ast.unparse(astAST)
		return workhorse

	# ================================================================
	# NOT used
	# TODO Does this work?
	@staticmethod
	def Z0Z_matchesAtLeast1Descendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if any descendant of the node matches the given predicate."""
		return lambda node: not ifThis.matchesNoDescendant(predicate)(node)
	# ================================================================
	# MORE function inlining
	@staticmethod
	def onlyReturnAnyCompare(astFunctionDef: ast.AST) -> TypeGuard[object]:
		return be.FunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnAnyCompare(astFunctionDef.body[0])
	# For function inlining
	@staticmethod
	def onlyReturnUnaryOp(astFunctionDef: ast.AST) -> TypeGuard[object]:
		return be.FunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnUnaryOp(astFunctionDef.body[0])
	# ================================================================
	# These are used by other functions
	@staticmethod
	def isCallAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: be.Call(node) and 又.func(ifThis.isAttributeNamespace_Identifier(namespace, identifier))(node)
	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: be.Name(node) and 又.id(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: be.Call(node) and 又.func(ifThis.isName_Identifier(identifier))(node)
	# ================================================================
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
	def CallDoesNotCallItself(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		"""If `namespace` is not applicable to your case, then call with `namespace=""`."""
		return lambda node: ifThis.matchesMeButNotAnyDescendant(ifThis.CallReallyIs(namespace, identifier))(node)
	@staticmethod
	def CallReallyIs(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return ifThis.isCall_Identifier(identifier) or ifThis.isCallAttributeNamespace_Identifier(namespace, identifier)
	@staticmethod
	def isAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: ifThis.isAttributeName(node) and 又.value(ifThis.isName_Identifier(namespace))(node) and 又.attr(ifThis._Identifier(identifier))(node)
	@staticmethod
	def _Identifier(identifier: ast_Identifier) -> Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node == identifier
	@staticmethod
	def isAttributeName(node: ast.AST) -> TypeGuard[object]:
		""" Displayed as Name.attribute."""
		return be.Attribute(node) and 又.value(be.Name)(node)

	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[object]:
		return be.Call(node) and 又.func(be.Name)(node)
	@staticmethod
	def ast_IdentifierIn(container: Container[ast_Identifier]) -> Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node in container
	# This bullshit is for the crappy function inliner I made.
	@staticmethod
	def CallDoesNotCallItselfAndNameDOTidIsIn(container: Container[ast_Identifier]) -> Callable[[ast.AST], TypeGuard[object] | bool]:
		return lambda node: ifThis.isCallToName(node) and 又.func(又.id(ifThis.ast_IdentifierIn(container)))(node) and ifThis.CallDoesNotCallItself("", node.func.id)(node)
