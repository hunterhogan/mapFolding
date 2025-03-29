from collections.abc import Callable, Container
from mapFolding.someAssemblyRequired import ast_expr_Slice, ast_Identifier, astClassHasDOTnameNotName, astClassHasDOTtarget, astClassHasDOTvalue, ImaAnnotationType, typeCertified
from typing import Any, TypeGuard
import ast

Ima_targetType = ast.AST
ImaAnnAssign_targetType = ast.AST
ImaAnnAugAssign_targetType = ast.AST

class 又:
	@staticmethod
	def annotation(predicate: Callable[[ImaAnnotationType], TypeGuard[ImaAnnotationType] | bool | ast.AST]) -> Callable[[ast.AnnAssign | ast.arg], TypeGuard[ast.AnnAssign] | TypeGuard[ast.arg] | bool]:
		def workhorse(node: ast.AnnAssign | ast.arg) -> TypeGuard[ast.AnnAssign] | TypeGuard[ast.arg] | bool:
			return predicate(node.annotation) # pyright: ignore [reportArgumentType, reportReturnType]
			# return be.Annotation(ImaAnnotation) and predicate(ImaAnnotation) # This does not help because `be.Annotation` does not change the type of `ImaAnnotation` from `ast.expr` to something else. And I'm not sure why it doesn't change.
		return workhorse
	@staticmethod
	def arg(predicate: Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]) -> Callable[[ast.arg | ast.keyword], TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool]:
		# return lambda node: predicate(node.arg)
		def workhorse(node: ast.arg | ast.keyword) -> TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool:
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
	def AnnAssign(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.AnnAssign)(node) # pyright: ignore [reportInvalidTypeVarUse]
# 'TypeVar "typeCertified" appears only once in generic function signature. Use "object" instead Pylance(reportInvalidTypeVarUse)"' HOW THE FUCK IS THAT INVALID WHEN IT IS WORKING PERFECTLY TO PASS THE TYPE INFORMATION--IN YOUR FUCKING STATIC TYPE CHECKER, PYLANCE!!!! Fuck you, and fuck your pretentious language.
	@staticmethod
	def arg(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.arg)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Annotation(node: ast.AST) -> TypeGuard[typeCertified] | bool: # pyright: ignore [reportInvalidTypeVarUse]
		return be.Attribute(node) or be.Constant(node) or be.Name(node) or be.Subscript(node)
	@staticmethod
	def Assign(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Assign)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Attribute(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Attribute)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def AugAssign(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.AugAssign)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def BoolOp(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.BoolOp)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Call(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Call)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def ClassDef(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.ClassDef)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Compare(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Compare)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Constant(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Constant)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Expr(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Expr)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def FunctionDef(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.FunctionDef)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Import(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Import)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def keyword(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.keyword)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Name(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Name)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Return(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Return)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Starred(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Starred)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def Subscript(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.Subscript)(node) # pyright: ignore [reportInvalidTypeVarUse]
	@staticmethod
	def UnaryOp(node: ast.AST) -> TypeGuard[typeCertified]: return be._typeCertified(ast.UnaryOp)(node) # pyright: ignore [reportInvalidTypeVarUse]

class ifThis:
	@staticmethod
	def equals(this: Any) -> Callable[[Any], TypeGuard[Any] | bool]:
		return lambda node: node == this
	@staticmethod
	def isAssignAndTargets0Is(targets0Predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
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
		return lambda node: be.Assign(node) and 又.value(valuePredicate)(node)
	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.FunctionDef] | bool]:
		return lambda node: be.FunctionDef(node) and 又.name(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isArgument_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool]:
		return lambda node: (be.arg(node) or be.keyword(node)) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.keyword(node) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def is_arg_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.arg(node) and 又.arg(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.ClassDef] | bool]:
		return lambda node: be.ClassDef(node) and 又.name(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isAssignAndValueIsCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return lambda node: be.Assign(node) and 又.value(ifThis.isCall_Identifier(identifier))(node)
	@staticmethod
	def isAssignAndValueIsCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return ifThis.isAssignAndValueIs(ifThis.isCallNamespace_Identifier(namespace, identifier))
	@staticmethod
	def is_keywordAndValueIsConstant(node: ast.AST) -> TypeGuard[ast.keyword]:
		return be.keyword(node) and 又.value(be.Constant)(node)
	@staticmethod
	def is_keyword_IdentifierEqualsConstantValue(identifier: ast_Identifier, ConstantValue: Any) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		return lambda node: ifThis.is_keyword_Identifier(identifier)(node) and ifThis.is_keywordAndValueIsConstant(node) and 又.value(ifThis.isConstantEquals(ConstantValue))(node)

	@staticmethod
	def isAnnAssign_targetIs(targetPredicate: Callable[[typeCertified|Ima_targetType|ImaAnnAssign_targetType|ImaAnnAugAssign_targetType], TypeGuard[typeCertified|Ima_targetType|ImaAnnAssign_targetType|ImaAnnAugAssign_targetType] | bool]) -> Callable[[ast.AST], TypeGuard[typeCertified] | bool]:
		return lambda node: be.AnnAssign(node) and 又.target(targetPredicate)(node) # type: ignore
	@staticmethod
	def isAnnAssignAndAnnotationIsName(node: ast.AST) -> TypeGuard[ast.AnnAssign]:
		return be.AnnAssign(node) and 又.annotation(be.Name)(node)

	@staticmethod
	def isAnyCompare(node: ast.AST) -> TypeGuard[ast.Compare] | TypeGuard[ast.BoolOp]:
		return be.Compare(node) or be.BoolOp(node)
	@staticmethod
	def isConstantEquals(value: Any) -> Callable[[ast.AST], TypeGuard[ast.Constant] | bool]:
		return lambda node: be.Constant(node) and 又.value(ifThis.equals(value))(node)
	@staticmethod
	def isReturnAnyCompare(node: ast.AST) -> TypeGuard[ast.Return] | bool:
		return be.Return(node) and 又.value(ifThis.isAnyCompare)(node)
	@staticmethod
	def isReturnUnaryOp(node: ast.AST) -> TypeGuard[ast.Return] | bool:
		return be.Return(node) and 又.value(be.UnaryOp)(node)

	# ================================================================
	# Nested identifier
	@staticmethod
	def _nestedJunction_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | TypeGuard[ast.Subscript] | TypeGuard[ast.Starred] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute] | TypeGuard[ast.Subscript] | TypeGuard[ast.Starred] | bool:
			return ifThis.isName_Identifier(identifier)(node) or ifThis.isAttribute_Identifier(identifier)(node) or ifThis.isSubscript_Identifier(identifier)(node) or ifThis.isStarred_Identifier(identifier)(node)
		return workhorse
	@staticmethod
	def isAttribute_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		"""node is `ast.Attribute` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute]:
			return be.Attribute(node) and 又.value(ifThis._nestedJunction_Identifier(identifier))(node)
		return workhorse
	@staticmethod
	def isStarred_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Starred] | bool]:
		"""node is `ast.Starred` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Starred]:
			return be.Starred(node) and 又.value(ifThis._nestedJunction_Identifier(identifier))(node)
		return workhorse
	@staticmethod
	def isSubscript_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript] | bool]:
		"""node is `ast.Subscript` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Subscript]:
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
	def onlyReturnAnyCompare(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return be.FunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnAnyCompare(astFunctionDef.body[0])
	# For function inlining
	@staticmethod
	def onlyReturnUnaryOp(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return be.FunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnUnaryOp(astFunctionDef.body[0])
	# ================================================================
	# These are used by other functions
	@staticmethod
	def isCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: be.Call(node) and 又.func(ifThis.is_nameDOTnameNamespace_Identifier(namespace, identifier))(node)
	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Name] | bool]:
		return lambda node: be.Name(node) and 又.id(ifThis._Identifier(identifier))(node)
	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
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
	def CallDoesNotCallItself(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		"""If `namespace` is not applicable to your case, then call with `namespace=""`."""
		return lambda node: ifThis.matchesMeButNotAnyDescendant(ifThis.CallReallyIs(namespace, identifier))(node)
	@staticmethod
	def CallReallyIs(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return ifThis.isCall_Identifier(identifier) or ifThis.isCallNamespace_Identifier(namespace, identifier)
	@staticmethod
	def is_nameDOTnameNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		return lambda node: ifThis.is_nameDOTname(node) and 又.value(ifThis.isName_Identifier(namespace))(node) and 又.attr(ifThis._Identifier(identifier))(node)
	@staticmethod
	def _Identifier(identifier: ast_Identifier) -> Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node == identifier
	@staticmethod
	def is_nameDOTname(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return be.Attribute(node) and 又.value(be.Name)(node)

	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[ast.Call]:
		return be.Call(node) and 又.func(be.Name)(node)
	@staticmethod
	def ast_IdentifierIn(container: Container[ast_Identifier]) -> Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node in container
	# This bullshit is for the crappy function inliner I made.
	@staticmethod
	def CallDoesNotCallItselfAndNameDOTidIsIn(container: Container[ast_Identifier]) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: ifThis.isCallToName(node) and 又.func(又.id(ifThis.ast_IdentifierIn(container)))(node) and ifThis.CallDoesNotCallItself("", node.func.id)(node) # type: ignore
