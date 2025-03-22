from collections.abc import Callable, Container
from mapFolding.someAssemblyRequired import ast_Identifier
from typing import Any, TypeGuard
import ast

class ifThis:
	@staticmethod
	def ast_IdentifierIsIn(container: Container[ast_Identifier]) -> Callable[[ast_Identifier], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node in container
	@staticmethod
	def CallDoesNotCallItself(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		"""If `namespace` is not applicable to your case, then call with `namespace=""`."""
		return lambda node: ifThis.matchesMeButNotAnyDescendant(ifThis.CallReallyIs(namespace, identifier))(node)
	@staticmethod
	def CallDoesNotCallItselfAndNameDOTidIsIn(container: Container[ast_Identifier]) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: ifThis.isCall(node) and ifThis.isName(node.func) and ifThis.ast_IdentifierIsIn(container)(node.func.id) and ifThis.CallDoesNotCallItself("", node.func.id)(node)
	@staticmethod
	def CallReallyIs(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return ifThis.isAnyOf(ifThis.isCall_Identifier(identifier), ifThis.isCallNamespace_Identifier(namespace, identifier))
	@staticmethod
	def is_arg(node: ast.AST) -> TypeGuard[ast.arg]:
		return isinstance(node, ast.arg)
	@staticmethod
	def is_arg_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.arg] | bool: return ifThis.is_arg(node) and node.arg == identifier
		return workhorse
	@staticmethod
	def is_keyword(node: ast.AST) -> TypeGuard[ast.keyword]:
		return isinstance(node, ast.keyword)
	@staticmethod
	def is_keywordAndValueIsConstant(node: ast.AST) -> TypeGuard[ast.keyword]:
		return ifThis.is_keyword(node) and ifThis.isConstant(node.value)
	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.keyword] | bool: return ifThis.is_keyword(node) and node.arg == identifier
		return workhorse
	@staticmethod
	def isArgument_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.arg] | TypeGuard[ast.keyword] | bool:
			return (ifThis.is_arg(node) or ifThis.is_keyword(node)) and node.arg == identifier
		return workhorse
	@staticmethod
	def is_keyword_IdentifierEqualsConstantValue(identifier: ast_Identifier, ConstantValue: Any) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		return lambda node: ifThis.is_keyword_Identifier(identifier)(node) and ifThis.is_keywordAndValueIsConstant(node) and ifThis.isConstantEquals(ConstantValue)(node.value)
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
	def isAnnAssignTo(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.AnnAssign] | bool]:
		return lambda node: ifThis.isAnnAssign(node) and ifThis.NameReallyIs_Identifier(identifier)(node.target)
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
	def isAssignAndValueIsCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return lambda node: ifThis.isAssign(node) and ifThis.isCall_Identifier(identifier)(node.value)
	@staticmethod
	def isAssignAndValueIsCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return ifThis.isAssignAndValueIs(ifThis.isCallNamespace_Identifier(namespace, identifier))
	@staticmethod
	def isAssignOnlyTo(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		return lambda node: ifThis.isAssign(node) and ifThis.NameReallyIs_Identifier(identifier)(node.targets[0])
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
		return lambda node: ifThis.isAssign(node) and valuePredicate(node.value)
	@staticmethod
	def isAttribute(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return isinstance(node, ast.Attribute)
	@staticmethod
	def isAugAssign(node: ast.AST) -> TypeGuard[ast.AugAssign]:
		return isinstance(node, ast.AugAssign)
	@staticmethod
	def isAugAssignTo(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.AugAssign] | bool]:
		return lambda node: ifThis.isAugAssign(node) and ifThis.NameReallyIs_Identifier(identifier)(node.target)
	@staticmethod
	def isBoolOp(node: ast.AST) -> TypeGuard[ast.BoolOp]:
		return isinstance(node, ast.BoolOp)
	@staticmethod
	def isCall(node: ast.AST) -> TypeGuard[ast.Call]:
		return isinstance(node, ast.Call)
	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Call] | bool: return ifThis.isCall(node) and ifThis.isName_Identifier(identifier)(node.func)
		return workhorse
	@staticmethod
	def isCallNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		return lambda node: ifThis.isCall(node) and ifThis.is_nameDOTnameNamespace_Identifier(namespace, identifier)(node.func)
	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[ast.Call]:
		return ifThis.isCall(node) and ifThis.isName(node.func)
	@staticmethod
	def isClassDef(node: ast.AST) -> TypeGuard[ast.ClassDef]:
		return isinstance(node, ast.ClassDef)
	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.ClassDef] | bool]:
		return lambda node: ifThis.isClassDef(node) and node.name == identifier
	@staticmethod
	def isCompare(node: ast.AST) -> TypeGuard[ast.Compare]:
		return isinstance(node, ast.Compare)
	@staticmethod
	def isConstant(node: ast.AST) -> TypeGuard[ast.Constant]:
		return isinstance(node, ast.Constant)
	@staticmethod
	def isConstantEquals(value: Any) -> Callable[[ast.AST], TypeGuard[ast.Constant] | bool]:
		return lambda node: ifThis.isConstant(node) and node.value == value
	@staticmethod
	def isExpr(node: ast.AST) -> TypeGuard[ast.Expr]:
		return isinstance(node, ast.Expr)
	@staticmethod
	def isFunctionDef(node: ast.AST) -> TypeGuard[ast.FunctionDef]: return isinstance(node, ast.FunctionDef)
	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.FunctionDef] | bool]:
		return lambda node: ifThis.isFunctionDef(node) and node.name == identifier
	@staticmethod
	def isImport(node: ast.AST) -> TypeGuard[ast.Import]:
		return isinstance(node, ast.Import)
	@staticmethod
	def isName(node: ast.AST) -> TypeGuard[ast.Name]:
		"""TODO
		ast.Name()
		ast.Attribute()
		ast.Subscript()
		ast.Starred()
		"""
		return isinstance(node, ast.Name)
	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Name] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Name] | bool: return ifThis.isName(node) and node.id == identifier
		return workhorse
	@staticmethod
	def is_nameDOTname(node: ast.AST) -> TypeGuard[ast.Attribute]:
		return ifThis.isAttribute(node) and ifThis.isName(node.value)
	@staticmethod
	def is_nameDOTnameNamespace(namespace: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute] | bool: return ifThis.is_nameDOTname(node) and ifThis.isName_Identifier(namespace)(node.value)
		return workhorse
	@staticmethod
	def is_nameDOTnameNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		return lambda node: ifThis.is_nameDOTname(node) and ifThis.isName_Identifier(namespace)(node.value) and node.attr == identifier
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
			workhorse: function that checks if a node matches the criteria
		"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Subscript]:
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
		return workhorse
	@staticmethod
	def isSubscriptIsName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript] | bool]:
		return lambda node: ifThis.isSubscript(node) and ifThis.isName_Identifier(identifier)(node.value)
	@staticmethod
	def isSubscript_Identifier_Identifier(identifier: ast_Identifier, sliceIdentifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript] | bool]:
		return lambda node: ifThis.isSubscript(node) and ifThis.isName_Identifier(identifier)(node.value) and ifThis.isName_Identifier(sliceIdentifier)(node.slice)
	@staticmethod
	def isUnaryOp(node: ast.AST) -> TypeGuard[ast.UnaryOp]:
		return isinstance(node, ast.UnaryOp)
	# TODO Does this work?
	@staticmethod
	def matchesAtLeast1Descendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		"""Create a predicate that returns True if any descendant of the node matches the given predicate."""
		return lambda node: not ifThis.matchesNoDescendant(predicate)(node)
	# TODO Does this work?
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
		def workhorse(node: ast.AST) -> bool:
			for descendant in ast.walk(node):
				if descendant is not node and predicate(descendant):
					return False
			return True
		return workhorse
	@staticmethod
	def onlyReturnAnyCompare(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return ifThis.isFunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnAnyCompare(astFunctionDef.body[0])
	@staticmethod
	def onlyReturnUnaryOp(astFunctionDef: ast.AST) -> TypeGuard[ast.FunctionDef]:
		return ifThis.isFunctionDef(astFunctionDef) and len(astFunctionDef.body) == 1 and ifThis.isReturnUnaryOp(astFunctionDef.body[0])
